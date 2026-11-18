//! Every two-plane colour texture this device converts on the CPU, as one type.
//!
//! Two families reach a type-12 object or a two-plane surface, and both are one
//! `MTLTexture` whose texels are a function of two planes:
//!
//! - the two-plane Y′CbCr formats ([`crate::ycbcr`]), a luma and a chroma plane;
//! - Apple's two-plane extended-range format, `RGB10A8_2P_XR10` (550): a colour
//!   plane and an alpha plane.
//!
//! [`two_plane_conversion`] parses an ordinal once into a [`TwoPlaneConversion`]
//! that says what each plane holds and what the texels read. Every other
//! ordinal is not a two-plane colour format — notably `Depth32Float_Stencil8`
//! (260), which also arrives as a type-12 object with a depth plane and a
//! stencil plane and is no colour texture to convert.
//!
//! # `RGB10A8_2P_XR10` (550)
//!
//! Measured by the metal2vulkan project on an Apple M3 Max (macOS 26.7) and an
//! Apple M4 (macOS 27.0.1), every code and channel, and stated in its
//! `metal::format::ExtendedRange` (`Planes`), `lower::arith::extended_range`
//! and `metal-runtime::two_planes`:
//!
//! - **Planes.** Plane 0 holds each texel's colour word, 4 bytes, laid out as
//!   `BGR10_XR`'s: a little-endian `u32` with blue in bits 0-9, green 10-19 and
//!   red 20-29; bits 30-31 are not read. Plane 1 holds each texel's alpha, **one
//!   byte**. (Over an IOSurface the pair is a `'b3a8'` surface's two planes; in
//!   Metal's own memory plane 0's rows are `align(4 w, 16)` bytes apart, the
//!   alpha plane starts at plane 0 rounded up to 16 KiB, and its rows are
//!   `align(w, 16)` bytes apart.)
//! - **Codes.** A colour channel's 10-bit code `c` reads `(c - 384) / 510`,
//!   correctly rounded to `f32`: the extended range `[-0.752941, 1.252941]`,
//!   with codes 384 and 894 reading 0 and 1. The alpha byte `b` reads `b / 255`,
//!   which is within `[0, 1]`. The value is linear: the format has no transfer
//!   function.
//! - **The `_sRGB` twin** (551) reads the same codes and passes each colour
//!   channel's magnitude through sRGB decoding, signed, by Metal's own tables,
//!   which are measured to within half a `2^-13` step and not recovered exactly.
//!   It refuses as [`TwoPlaneRefusal::ExtendedSrgbTable`].
//!
//! The conversion writes `RGBA16Float`: each channel the half float nearest the
//! exact value. None of these values is a tie on the half grid (a tie would
//! need `255 | (c - 384)`, which only the exactly representable 0, ±0.5 and 1
//! satisfy), so nearest is unambiguous, and it is what Metal's own `half` read
//! of the texture rounds the same value to.

use alloc::vec;
use alloc::vec::Vec;

use crate::pixel_format::{
    TexelLayout, MTL_FORMAT_R32_UINT, MTL_FORMAT_R8_UNORM, MTL_FORMAT_RGBA16_FLOAT,
};
use crate::ycbcr::{self, CodeLayout, Matrix, PlaneRows, Range, TexelRead, TwoPlaneRefusal};

/// `MTLPixelFormatRGB10A8_2P_XR10`.
pub const MTL_FORMAT_RGB10A8_2P_XR10: u16 = 550;
/// `MTLPixelFormatRGB10A8_2P_XR10_sRGB`.
pub const MTL_FORMAT_RGB10A8_2P_XR10_SRGB: u16 = 551;

/// The code an extended-range channel reads as 0.
pub const XR10_ZERO_CODE: u32 = 384;
/// Codes per unit of an extended-range channel.
pub const XR10_CODES_PER_UNIT: u32 = 510;

/// What a two-plane colour texture's planes hold, and what its texels read.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum TwoPlaneConversion {
    /// A two-plane Y′CbCr format and how its codes read.
    YCbCr { layout: CodeLayout, read: TexelRead },
    /// `RGB10A8_2P_XR10`: a `BGR10_XR` colour word and an alpha byte.
    ExtendedRange,
}

/// The conversion `pixel_format` names, or `None` when it is not a two-plane
/// colour format at all (`Depth32Float_Stencil8` among them).
///
/// `range` and `matrix` are what is known of the surface a Y′CbCr texture is
/// over; [`CodeLayout::texel_read`] says which a format needs.
pub fn two_plane_conversion(
    pixel_format: u16,
    range: Option<Range>,
    matrix: Option<Matrix>,
) -> Option<Result<TwoPlaneConversion, TwoPlaneRefusal>> {
    if let Some(format) = ycbcr::two_plane_format(pixel_format) {
        return Some(format.convertible().and_then(|layout| {
            let read = layout.texel_read(range, matrix)?;
            Ok(TwoPlaneConversion::YCbCr { layout, read })
        }));
    }
    match pixel_format {
        MTL_FORMAT_RGB10A8_2P_XR10 => Some(Ok(TwoPlaneConversion::ExtendedRange)),
        MTL_FORMAT_RGB10A8_2P_XR10_SRGB => {
            Some(Err(TwoPlaneRefusal::ExtendedSrgbTable { pixel_format }))
        }
        _ => None,
    }
}

impl TwoPlaneConversion {
    /// The `MTLPixelFormat` ordinal it was parsed from.
    pub const fn pixel_format(self) -> u16 {
        match self {
            Self::YCbCr { layout, .. } => layout.pixel_format(),
            Self::ExtendedRange => MTL_FORMAT_RGB10A8_2P_XR10,
        }
    }

    /// Bytes of one element of plane 0 and of plane 1.
    pub const fn plane_bytes_per_texel(self) -> [u32; 2] {
        match self {
            Self::YCbCr { layout, .. } => [
                layout.luma_bytes_per_texel(),
                layout.chroma_bytes_per_texel(),
            ],
            Self::ExtendedRange => [4, 1],
        }
    }

    /// The single-plane formats whose texels are one element of each plane, as
    /// a surface's plane records describe them.
    pub const fn plane_formats(self) -> [u16; 2] {
        match self {
            Self::YCbCr { layout, .. } => {
                [layout.luma_plane_format(), layout.chroma_plane_format()]
            }
            Self::ExtendedRange => [MTL_FORMAT_R32_UINT, MTL_FORMAT_R8_UNORM],
        }
    }

    /// Plane 0's and plane 1's extent for a texture of `width` x `height`.
    pub const fn plane_extents(self, width: u32, height: u32) -> [(u32, u32); 2] {
        match self {
            Self::YCbCr { layout, .. } => [
                (width, height),
                layout.subsampling().chroma_extent(width, height),
            ],
            Self::ExtendedRange => [(width, height), (width, height)],
        }
    }

    /// The layout [`Self::convert`] writes.
    pub const fn output_layout(self) -> TexelLayout {
        match self {
            Self::YCbCr { layout, .. } => layout.output_layout(),
            Self::ExtendedRange => TexelLayout::Rgba16Float,
        }
    }

    /// The single-plane `MTLPixelFormat` whose texels are [`Self::output_layout`].
    pub const fn output_pixel_format(self) -> u16 {
        match self {
            Self::YCbCr { layout, .. } => layout.output_pixel_format(),
            Self::ExtendedRange => MTL_FORMAT_RGBA16_FLOAT,
        }
    }

    /// The texels a `width` x `height` texture reads from its planes, tightly
    /// packed in [`Self::output_layout`].
    pub fn convert(
        self,
        width: u32,
        height: u32,
        plane0: PlaneRows<'_>,
        plane1: PlaneRows<'_>,
    ) -> Result<Vec<u8>, TwoPlaneRefusal> {
        match self {
            Self::YCbCr { layout, read } => {
                ycbcr::convert(layout, read, width, height, plane0, plane1)
            }
            Self::ExtendedRange => convert_extended_range(width, height, plane0, plane1),
        }
    }
}

/// The half float nearest `numerator / denominator`, for a value whose
/// magnitude is zero or within the half's normal range, never a tie.
const fn half_of_ratio(numerator: i32, denominator: u32) -> u16 {
    if numerator == 0 {
        return 0;
    }
    let sign: u16 = if numerator < 0 { 0x8000 } else { 0 };
    let n = numerator.unsigned_abs() as u64;
    let d = denominator as u64;
    // The exponent `e` with `2^e <= n / d < 2^(e + 1)`, over the half's normal
    // exponents.
    let mut e: i32 = 15;
    while e > -14 {
        let at_least = if e >= 0 { n >= d << e } else { n << -e >= d };
        if at_least {
            break;
        }
        e -= 1;
    }
    // Eleven significant bits: `n / d * 2^(10 - e)`, rounded to nearest even.
    let shift = 10 - e;
    let scaled = if shift >= 0 { n << shift } else { n >> -shift };
    let (mut q, r) = (scaled / d, scaled % d);
    if 2 * r > d || (2 * r == d && q & 1 == 1) {
        q += 1;
    }
    if q == 2048 {
        q = 1024;
        e += 1;
    }
    sign | (((e + 15) as u16) << 10) | ((q - 1024) as u16)
}

/// The half float an extended-range channel code reads.
pub const fn xr10_code_half(code: u32) -> u16 {
    half_of_ratio(code as i32 - XR10_ZERO_CODE as i32, XR10_CODES_PER_UNIT)
}

/// The half float an alpha byte reads.
pub const fn alpha_byte_half(byte: u8) -> u16 {
    half_of_ratio(byte as i32, 255)
}

fn convert_extended_range(
    width: u32,
    height: u32,
    color: PlaneRows<'_>,
    alpha: PlaneRows<'_>,
) -> Result<Vec<u8>, TwoPlaneRefusal> {
    if width == 0 || height == 0 {
        return Err(TwoPlaneRefusal::Extent { width, height });
    }
    let reach = |plane: &'static str, rows: PlaneRows<'_>, bpe: u64| {
        let tight = u64::from(width) * bpe;
        if (rows.pitch as u64) < tight {
            return Err(TwoPlaneRefusal::PitchShort {
                plane,
                pitch: rows.pitch as u64,
                tight,
            });
        }
        let need = rows.pitch as u64 * u64::from(height - 1) + tight;
        if (rows.bytes.len() as u64) < need {
            return Err(TwoPlaneRefusal::PlaneShort {
                plane,
                need,
                have: rows.bytes.len() as u64,
            });
        }
        Ok(())
    };
    reach("color", color, 4)?;
    reach("alpha", alpha, 1)?;
    let mut out = vec![0u8; width as usize * height as usize * 8];
    for y in 0..height as usize {
        let color_row = &color.bytes[y * color.pitch..];
        let alpha_row = &alpha.bytes[y * alpha.pitch..];
        for x in 0..width as usize {
            let w = u32::from_le_bytes([
                color_row[4 * x],
                color_row[4 * x + 1],
                color_row[4 * x + 2],
                color_row[4 * x + 3],
            ]);
            let texel = [
                xr10_code_half((w >> 20) & 0x3ff),
                xr10_code_half((w >> 10) & 0x3ff),
                xr10_code_half(w & 0x3ff),
                alpha_byte_half(alpha_row[x]),
            ];
            let at = (y * width as usize + x) * 8;
            for (i, h) in texel.into_iter().enumerate() {
                out[at + 2 * i..at + 2 * i + 2].copy_from_slice(&h.to_le_bytes());
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A half's value, exactly, as an `f64`.
    fn half_value(h: u16) -> f64 {
        let sign = if h & 0x8000 != 0 { -1.0 } else { 1.0 };
        let e = i32::from((h >> 10) & 0x1f);
        let m = f64::from(h & 0x3ff);
        assert!(e != 31, "no infinity or NaN here");
        if e == 0 {
            sign * m * 2f64.powi(-24)
        } else {
            sign * (1.0 + m / 1024.0) * 2f64.powi(e - 15)
        }
    }

    /// The half's two neighbours on the grid are no closer to `v` than it is.
    fn assert_nearest(h: u16, v: f64, what: &str) {
        let d = (half_value(h) - v).abs();
        for n in [h.wrapping_sub(1), h.wrapping_add(1)] {
            if (n & 0x7c00) == 0x7c00 || (n & 0x7fff) == 0 && n != h {
                continue;
            }
            assert!(
                (half_value(n) - v).abs() > d,
                "{what}: {h:#06x} is not the nearest half to {v}"
            );
        }
    }

    #[test]
    fn every_extended_range_code_reads_its_value_nearest() {
        for code in 0..1024u32 {
            let v = (f64::from(code) - 384.0) / 510.0;
            assert_nearest(xr10_code_half(code), v, &alloc::format!("code {code}"));
        }
        assert_eq!(xr10_code_half(384), 0, "384 reads 0");
        assert_eq!(xr10_code_half(894), 0x3c00, "894 reads 1");
        assert_eq!(xr10_code_half(639), 0x3800, "639 reads 0.5");
        assert_eq!(xr10_code_half(129), 0xb800, "129 reads -0.5");
        assert!(
            half_value(xr10_code_half(1023)) > 1.25,
            "the top of the range"
        );
        assert!(
            half_value(xr10_code_half(0)) < -0.75,
            "the bottom of the range"
        );
    }

    #[test]
    fn every_alpha_byte_reads_its_value_nearest() {
        for b in 0..=255u8 {
            assert_nearest(alpha_byte_half(b), f64::from(b) / 255.0, "alpha");
        }
        assert_eq!(alpha_byte_half(0), 0);
        assert_eq!(alpha_byte_half(255), 0x3c00);
    }

    #[test]
    fn the_ordinals_parse_and_depth_stencil_is_not_a_color_format() {
        assert_eq!(
            two_plane_conversion(550, None, None),
            Some(Ok(TwoPlaneConversion::ExtendedRange))
        );
        assert_eq!(
            two_plane_conversion(551, None, None),
            Some(Err(TwoPlaneRefusal::ExtendedSrgbTable {
                pixel_format: 551
            }))
        );
        assert_eq!(
            two_plane_conversion(
                crate::pixel_format::MTL_FORMAT_DEPTH32_FLOAT_STENCIL8,
                None,
                None
            ),
            None
        );
        assert!(matches!(
            two_plane_conversion(542, None, None),
            Some(Ok(TwoPlaneConversion::YCbCr {
                read: TexelRead::Codes,
                ..
            }))
        ));
        let xr = TwoPlaneConversion::ExtendedRange;
        assert_eq!(xr.plane_bytes_per_texel(), [4, 1]);
        assert_eq!(xr.plane_extents(320, 128), [(320, 128), (320, 128)]);
        assert_eq!(xr.output_layout(), TexelLayout::Rgba16Float);
    }

    /// Red in bits 20-29, green 10-19, blue 0-9; bits 30-31 unread; alpha the
    /// byte at the texel's column of its own row.
    #[test]
    fn an_extended_range_texel_is_its_bgr10_word_and_its_alpha_byte() {
        let word = |r: u32, g: u32, b: u32, junk: u32| (junk << 30) | (r << 20) | (g << 10) | b;
        let words = [word(894, 384, 0, 3), word(1023, 639, 129, 0)];
        // Rows of 3 texels' room, 2 texels each; the third word is padding.
        let mut color = Vec::new();
        for row in 0..2 {
            for (x, w) in words.iter().enumerate() {
                color.extend_from_slice(&(w + (row * x as u32) % 2).to_le_bytes());
            }
            color.extend_from_slice(&[0xee; 4]);
        }
        let alpha = [255u8, 128, 0xee, 0, 64, 0xee];
        let out = TwoPlaneConversion::ExtendedRange
            .convert(
                2,
                2,
                PlaneRows {
                    bytes: &color,
                    pitch: 12,
                },
                PlaneRows {
                    bytes: &alpha,
                    pitch: 3,
                },
            )
            .unwrap();
        let halves: Vec<u16> = out
            .chunks_exact(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]))
            .collect();
        assert_eq!(
            halves[0..4],
            [0x3c00, 0, xr10_code_half(0), 0x3c00],
            "white-red: 1, 0, the bottom of the range; alpha 1 (junk bits unread)"
        );
        assert_eq!(
            halves[4..8],
            [xr10_code_half(1023), 0x3800, 0xb800, alpha_byte_half(128)]
        );
        // Row 1, texel 1: blue's code is 130 (the row's +1), alpha 64.
        assert_eq!(
            halves[12..16],
            [
                xr10_code_half(1023),
                0x3800,
                xr10_code_half(130),
                alpha_byte_half(64)
            ]
        );
        assert_eq!(halves[11], 0, "row 1 texel 0's alpha byte is 0");
    }
}
