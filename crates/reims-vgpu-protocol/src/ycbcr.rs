//! Two-plane Y′CbCr pixel formats: which ordinals they are, what each one's
//! planes hold, and the texels Metal reads out of them.
//!
//! # What the guest is handed
//!
//! One `MTLTexture` whose pixel format names a luma plane (plane 0, one code a
//! texel) and a chroma plane (plane 1, a Cb and then a Cr code for each 2 x 2,
//! 2 x 1 or single texel). It is **not** two plane textures: a shader samples it
//! as RGB, and what that RGB is depends on the format's family:
//!
//! - **Codes** (`RGB8_420_2P` .. `RGB10_444_2P`, 540 to 545): the codes
//!   themselves after the chroma filter, red `Cr`, green `Y`, blue `Cb`, each
//!   `code / (2^bits - 1)`, alpha 1. Nothing about the surface's colour space
//!   enters, so a texel is a function of the planes alone.
//! - **Converted** (`YCBCR8_420_2P` .. `YCBCR10_444_2P`, 500 to 507): the same
//!   filtered codes converted to RGB by the surface's **range** (its OSType,
//!   `'420v'` against `'420f'`) and **matrix** (`kIOSurfaceYCbCrMatrix` as it
//!   stood when the texture was made). Neither is a property of the pixel
//!   format, so a texel is *not* a function of the planes and the format alone;
//!   [`CodeLayout::texel_read`] refuses by name when either is unknown.
//! - Everything else this table names — the `_PACKED`, 12-bit, `_sRGB`, `_PQ`,
//!   `_HLG` and `_XR` formats — is a two-plane format this crate recognises and
//!   does not convert, each refused under its own slug.
//!
//! # Where the arithmetic comes from
//!
//! Measured on Apple GPUs by the metal2vulkan project (`bind::ycbcr_convert`,
//! whose module docs state each rule with the surfaces and texels it was
//! checked over) and restated here as integer arithmetic so the CPU produces
//! the same texels bit for bit:
//!
//! - **Chroma filter.** Along an axis the chroma plane halves, texel `p` reads
//!   two chroma texels: for an even `p`, `p / 2` weighted 3 and the one before
//!   it weighted 1; for an odd `p`, `p / 2` weighted 3 and the one after it
//!   weighted 1; each clamped to the plane. Along an axis it does not halve, the
//!   texel's own pair. The weighted sum `S` of `h` halved axes rounds to a code
//!   as `(S + 2^(2h - 1)) >> 2h` — `(S + 8) >> 4` for 4:2:0, `(S + 2) >> 2` for
//!   4:2:2 — and is the pair itself for 4:4:4.
//! - **Codes.** A code of 8 bits is a byte; one of 10 bits is the high 10 bits
//!   of a little-endian 16-bit word.
//! - **Conversion.** Counting codes as 10-bit (`u = 10 - bits`),
//!   `Cb' = (Cb << u) - 512`, `Cr' = (Cr << u) - 512`; video-range luma
//!   `L = 9576 ((Y << u) - 64)`, full-range `L = 256 ⌊(32768 Y + ⌊m / 2⌋) / m⌋`
//!   with `m = 2^bits - 1`; each of red, green and blue is `v / 2048` with
//!   `v = clamp((L + kb Cb' + kr Cr' + 2048) >> 12, 0, 2048)` (arithmetic shift)
//!   and `(kb, kr)` from [`coefficients`]; alpha is 1. Every value lies on the
//!   grid of `1 / 2048`, which a half float holds exactly.
//!
//! The output layouts are the ones that hold those values exactly:
//! `RGBA8Unorm` for 8-bit codes, `RGB10A2Unorm` for 10-bit codes, and
//! `RGBA16Float` for converted texels.

use alloc::string::{String, ToString};
use alloc::vec;
use alloc::vec::Vec;

use crate::pixel_format::{
    TexelLayout, MTL_FORMAT_R16_UNORM, MTL_FORMAT_R8_UNORM, MTL_FORMAT_RG16_UNORM,
    MTL_FORMAT_RG8_UNORM, MTL_FORMAT_RGB10A2_UNORM, MTL_FORMAT_RGBA16_FLOAT,
    MTL_FORMAT_RGBA8_UNORM,
};

/// How many luma texels share one chroma pair.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Subsampling {
    /// 4:2:0 — a pair for each 2 x 2 texels.
    Chroma420,
    /// 4:2:2 — a pair for each 2 x 1 texels.
    Chroma422,
    /// 4:4:4 — a pair for each texel.
    Chroma444,
}

impl Subsampling {
    /// Whether the chroma plane holds one pair for every two luma columns, and
    /// for every two luma rows.
    pub const fn halved(self) -> [bool; 2] {
        match self {
            Self::Chroma420 => [true, true],
            Self::Chroma422 => [true, false],
            Self::Chroma444 => [false, false],
        }
    }

    /// The chroma plane's extent for a texture of `width` x `height` luma
    /// texels: a halved axis is `⌈n / 2⌉` pairs.
    pub const fn chroma_extent(self, width: u32, height: u32) -> (u32, u32) {
        let [hx, hy] = self.halved();
        (
            if hx { width.div_ceil(2) } else { width },
            if hy { height.div_ceil(2) } else { height },
        )
    }
}

/// The bits of each luma and chroma code.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum CodeBits {
    Eight,
    Ten,
    Twelve,
}

impl CodeBits {
    pub const fn bits(self) -> u32 {
        match self {
            Self::Eight => 8,
            Self::Ten => 10,
            Self::Twelve => 12,
        }
    }
}

/// What a texel of a two-plane format reads of its codes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Reads {
    /// The codes, red `Cr`, green `Y`, blue `Cb` (`RGB*_2P`).
    Codes,
    /// RGB converted by the surface's range and matrix (`YCBCR*_2P`).
    Converted,
    /// The converted RGB through the sRGB curve (`_sRGB`).
    Srgb,
    /// The converted RGB through the PQ curve (`_PQ`).
    Pq,
    /// The converted RGB through the HLG curve (`_HLG`).
    Hlg,
    /// The converted RGB in the extended range (`_XR`).
    ExtendedRange,
}

impl Reads {
    const fn curve_name(self) -> &'static str {
        match self {
            Self::Codes => "codes",
            Self::Converted => "converted",
            Self::Srgb => "srgb",
            Self::Pq => "pq",
            Self::Hlg => "hlg",
            Self::ExtendedRange => "xr",
        }
    }
}

/// A two-plane Y′CbCr pixel format, parsed once from its ordinal.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TwoPlaneFormat {
    /// The `MTLPixelFormat` ordinal it was parsed from.
    pub pixel_format: u16,
    pub bits: CodeBits,
    pub subsampling: Subsampling,
    /// Three 10- or 12-bit codes to a 32-bit word (`_PACKED`), rather than one
    /// code to a byte or to the high bits of a 16-bit word.
    pub packed: bool,
    pub reads: Reads,
}

macro_rules! two_plane_table {
    ($($pf:literal => $bits:ident, $sub:ident, $packed:literal, $reads:ident;)*) => {
        /// Every ordinal [`two_plane_format`] answers for, in table order.
        pub const TWO_PLANE_ORDINALS: &[u16] = &[$($pf,)*];

        /// The two-plane Y′CbCr format `pixel_format` names, or `None` for
        /// every other ordinal.
        ///
        /// The table is the 69 two-plane formats Metal makes a texture of over
        /// an IOSurface's planes, by the names it prints for each ordinal. It is
        /// closed: an ordinal outside it is not a two-plane format, whatever
        /// its neighbours are.
        pub const fn two_plane_format(pixel_format: u16) -> Option<TwoPlaneFormat> {
            let (bits, subsampling, packed, reads) = match pixel_format {
                $($pf => (CodeBits::$bits, Subsampling::$sub, $packed, Reads::$reads),)*
                _ => return None,
            };
            Some(TwoPlaneFormat { pixel_format, bits, subsampling, packed, reads })
        }
    };
}

two_plane_table! {
    500 => Eight, Chroma420, false, Converted;   // YCBCR8_420_2P
    502 => Eight, Chroma422, false, Converted;   // YCBCR8_422_2P
    503 => Eight, Chroma444, false, Converted;   // YCBCR8_444_2P
    505 => Ten, Chroma420, false, Converted;     // YCBCR10_420_2P
    506 => Ten, Chroma422, false, Converted;     // YCBCR10_422_2P
    507 => Ten, Chroma444, false, Converted;     // YCBCR10_444_2P
    508 => Ten, Chroma420, true, Converted;      // YCBCR10_420_2P_PACKED
    509 => Ten, Chroma422, true, Converted;      // YCBCR10_422_2P_PACKED
    510 => Ten, Chroma444, true, Converted;      // YCBCR10_444_2P_PACKED
    520 => Eight, Chroma420, false, Srgb;        // YCBCR8_420_2P_sRGB
    522 => Eight, Chroma422, false, Srgb;        // YCBCR8_422_2P_sRGB
    523 => Eight, Chroma444, false, Srgb;        // YCBCR8_444_2P_sRGB
    525 => Ten, Chroma420, false, Srgb;          // YCBCR10_420_2P_sRGB
    526 => Ten, Chroma422, false, Srgb;          // YCBCR10_422_2P_sRGB
    527 => Ten, Chroma444, false, Srgb;          // YCBCR10_444_2P_sRGB
    528 => Ten, Chroma420, true, Srgb;           // YCBCR10_420_2P_PACKED_sRGB
    529 => Ten, Chroma422, true, Srgb;           // YCBCR10_422_2P_PACKED_sRGB
    530 => Ten, Chroma444, true, Srgb;           // YCBCR10_444_2P_PACKED_sRGB
    540 => Eight, Chroma420, false, Codes;       // RGB8_420_2P
    541 => Eight, Chroma422, false, Codes;       // RGB8_422_2P
    542 => Eight, Chroma444, false, Codes;       // RGB8_444_2P
    543 => Ten, Chroma420, false, Codes;         // RGB10_420_2P
    544 => Ten, Chroma422, false, Codes;         // RGB10_422_2P
    545 => Ten, Chroma444, false, Codes;         // RGB10_444_2P
    546 => Ten, Chroma420, true, Codes;          // RGB10_420_2P_PACKED
    547 => Ten, Chroma422, true, Codes;          // RGB10_422_2P_PACKED
    548 => Ten, Chroma444, true, Codes;          // RGB10_444_2P_PACKED
    564 => Ten, Chroma420, false, Pq;            // YCBCR10_420_2P_PQ
    565 => Ten, Chroma422, false, Pq;            // YCBCR10_422_2P_PQ
    566 => Ten, Chroma444, false, Pq;            // YCBCR10_444_2P_PQ
    567 => Ten, Chroma420, true, Pq;             // YCBCR10_420_2P_PACKED_PQ
    568 => Ten, Chroma422, true, Pq;             // YCBCR10_422_2P_PACKED_PQ
    569 => Ten, Chroma444, true, Pq;             // YCBCR10_444_2P_PACKED_PQ
    570 => Twelve, Chroma420, false, Converted;  // YCBCR12_420_2P
    571 => Twelve, Chroma422, false, Converted;  // YCBCR12_422_2P
    572 => Twelve, Chroma444, false, Converted;  // YCBCR12_444_2P
    573 => Twelve, Chroma420, false, Pq;         // YCBCR12_420_2P_PQ
    574 => Twelve, Chroma422, false, Pq;         // YCBCR12_422_2P_PQ
    575 => Twelve, Chroma444, false, Pq;         // YCBCR12_444_2P_PQ
    580 => Twelve, Chroma420, true, Converted;   // YCBCR12_420_2P_PACKED
    581 => Twelve, Chroma422, true, Converted;   // YCBCR12_422_2P_PACKED
    582 => Twelve, Chroma444, true, Converted;   // YCBCR12_444_2P_PACKED
    583 => Twelve, Chroma420, true, Pq;          // YCBCR12_420_2P_PACKED_PQ
    584 => Twelve, Chroma422, true, Pq;          // YCBCR12_422_2P_PACKED_PQ
    585 => Twelve, Chroma444, true, Pq;          // YCBCR12_444_2P_PACKED_PQ
    591 => Ten, Chroma420, false, ExtendedRange;    // YCBCR10_420_2P_XR
    592 => Ten, Chroma422, false, ExtendedRange;    // YCBCR10_422_2P_XR
    593 => Ten, Chroma444, false, ExtendedRange;    // YCBCR10_444_2P_XR
    594 => Ten, Chroma420, true, ExtendedRange;     // YCBCR10_420_2P_PACKED_XR
    595 => Ten, Chroma422, true, ExtendedRange;     // YCBCR10_422_2P_PACKED_XR
    596 => Ten, Chroma444, true, ExtendedRange;     // YCBCR10_444_2P_PACKED_XR
    597 => Twelve, Chroma420, false, ExtendedRange; // YCBCR12_420_2P_XR
    598 => Twelve, Chroma422, false, ExtendedRange; // YCBCR12_422_2P_XR
    599 => Twelve, Chroma444, false, ExtendedRange; // YCBCR12_444_2P_XR
    600 => Twelve, Chroma420, true, ExtendedRange;  // YCBCR12_420_2P_PACKED_XR
    601 => Twelve, Chroma422, true, ExtendedRange;  // YCBCR12_422_2P_PACKED_XR
    602 => Twelve, Chroma444, true, ExtendedRange;  // YCBCR12_444_2P_PACKED_XR
    608 => Ten, Chroma420, false, Hlg;           // YCBCR10_420_2P_HLG
    609 => Ten, Chroma422, false, Hlg;           // YCBCR10_422_2P_HLG
    610 => Ten, Chroma444, false, Hlg;           // YCBCR10_444_2P_HLG
    611 => Ten, Chroma420, true, Hlg;            // YCBCR10_420_2P_PACKED_HLG
    612 => Ten, Chroma422, true, Hlg;            // YCBCR10_422_2P_PACKED_HLG
    613 => Ten, Chroma444, true, Hlg;            // YCBCR10_444_2P_PACKED_HLG
    615 => Twelve, Chroma420, false, Hlg;        // YCBCR12_420_2P_HLG
    616 => Twelve, Chroma422, false, Hlg;        // YCBCR12_422_2P_HLG
    617 => Twelve, Chroma444, false, Hlg;        // YCBCR12_444_2P_HLG
    618 => Twelve, Chroma420, true, Hlg;         // YCBCR12_420_2P_PACKED_HLG
    619 => Twelve, Chroma422, true, Hlg;         // YCBCR12_422_2P_PACKED_HLG
    620 => Twelve, Chroma444, true, Hlg;         // YCBCR12_444_2P_PACKED_HLG
}

/// What a convertible layout's texels read: the codes, or the codes converted.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum ConvertibleReads {
    Codes,
    Converted,
}

/// A two-plane layout this crate converts: unpacked 8- or 10-bit codes, read
/// as codes or converted. Only [`TwoPlaneFormat::convertible`] makes one.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct CodeLayout {
    pixel_format: u16,
    ten_bit: bool,
    subsampling: Subsampling,
    reads: ConvertibleReads,
}

impl TwoPlaneFormat {
    /// This format as a layout the converter takes, or the refusal naming why
    /// it is not one.
    pub fn convertible(self) -> Result<CodeLayout, TwoPlaneRefusal> {
        let pixel_format = self.pixel_format;
        if self.packed {
            return Err(TwoPlaneRefusal::PackedCodes { pixel_format });
        }
        let ten_bit = match self.bits {
            CodeBits::Eight => false,
            CodeBits::Ten => true,
            CodeBits::Twelve => return Err(TwoPlaneRefusal::TwelveBitCodes { pixel_format }),
        };
        let reads = match self.reads {
            Reads::Codes => ConvertibleReads::Codes,
            Reads::Converted => ConvertibleReads::Converted,
            curve => {
                return Err(TwoPlaneRefusal::TransferCurve {
                    pixel_format,
                    curve: curve.curve_name(),
                })
            }
        };
        Ok(CodeLayout {
            pixel_format,
            ten_bit,
            subsampling: self.subsampling,
            reads,
        })
    }
}

impl CodeLayout {
    pub const fn pixel_format(self) -> u16 {
        self.pixel_format
    }

    pub const fn subsampling(self) -> Subsampling {
        self.subsampling
    }

    pub const fn reads(self) -> ConvertibleReads {
        self.reads
    }

    pub const fn bits(self) -> u32 {
        if self.ten_bit {
            10
        } else {
            8
        }
    }

    /// Bytes of one luma code: a byte, or a 16-bit word.
    pub const fn luma_bytes_per_texel(self) -> u32 {
        if self.ten_bit {
            2
        } else {
            1
        }
    }

    /// Bytes of one chroma pair: two codes.
    pub const fn chroma_bytes_per_texel(self) -> u32 {
        2 * self.luma_bytes_per_texel()
    }

    /// The single-plane format whose texels are one luma code, which is how a
    /// surface's plane records describe the luma plane.
    pub const fn luma_plane_format(self) -> u16 {
        if self.ten_bit {
            MTL_FORMAT_R16_UNORM
        } else {
            MTL_FORMAT_R8_UNORM
        }
    }

    /// The single-plane format whose texels are one chroma pair.
    pub const fn chroma_plane_format(self) -> u16 {
        if self.ten_bit {
            MTL_FORMAT_RG16_UNORM
        } else {
            MTL_FORMAT_RG8_UNORM
        }
    }

    /// The layout [`convert`] writes, which holds every texel this format
    /// reads exactly.
    pub const fn output_layout(self) -> TexelLayout {
        match (self.reads, self.ten_bit) {
            (ConvertibleReads::Codes, false) => TexelLayout::Rgba8,
            (ConvertibleReads::Codes, true) => TexelLayout::Rgb10a2Unorm,
            (ConvertibleReads::Converted, _) => TexelLayout::Rgba16Float,
        }
    }

    /// The single-plane `MTLPixelFormat` whose texels are [`Self::output_layout`].
    pub const fn output_pixel_format(self) -> u16 {
        match (self.reads, self.ten_bit) {
            (ConvertibleReads::Codes, false) => MTL_FORMAT_RGBA8_UNORM,
            (ConvertibleReads::Codes, true) => MTL_FORMAT_RGB10A2_UNORM,
            (ConvertibleReads::Converted, _) => MTL_FORMAT_RGBA16_FLOAT,
        }
    }

    /// Bytes of one output texel.
    pub const fn output_bytes_per_texel(self) -> u32 {
        match self.reads {
            ConvertibleReads::Codes => 4,
            ConvertibleReads::Converted => 8,
        }
    }

    /// How this layout's texels read, given what is known of the surface.
    ///
    /// A layout read as codes needs neither term and ignores both. A converted
    /// one needs both, and an unknown one is a refusal under its own slug —
    /// there is no default: BT.601 is what Metal reads for a surface *with no
    /// matrix attached*, which is a fact about a surface, not a stand-in for
    /// one nobody has read.
    pub fn texel_read(
        self,
        range: Option<Range>,
        matrix: Option<Matrix>,
    ) -> Result<TexelRead, TwoPlaneRefusal> {
        match self.reads {
            ConvertibleReads::Codes => Ok(TexelRead::Codes),
            ConvertibleReads::Converted => {
                let pixel_format = self.pixel_format;
                let range = range.ok_or(TwoPlaneRefusal::RangeUnknown { pixel_format })?;
                let matrix = matrix.ok_or(TwoPlaneRefusal::MatrixUnknown { pixel_format })?;
                Ok(TexelRead::Converted(ConversionTerms { range, matrix }))
            }
        }
    }
}

/// The range of a surface's codes, which its OSType names.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Range {
    /// Luma codes 16 to 235 (8-bit) span black to white: `'420v'`, `'x420'`.
    Video,
    /// Luma codes 0 to 255 span black to white: `'420f'`, `'xf20'`.
    Full,
}

/// The matrix a converted texel is read through: the surface's
/// `kIOSurfaceYCbCrMatrix` when the texture was made.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Matrix {
    Bt601,
    Bt709,
    Bt2020,
}

/// The two terms a converted texel depends on beyond its codes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ConversionTerms {
    pub range: Range,
    pub matrix: Matrix,
}

/// What [`convert`] produces from a layout's codes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum TexelRead {
    Codes,
    Converted(ConversionTerms),
}

const fn fourcc(code: &[u8; 4]) -> u32 {
    u32::from_be_bytes(*code)
}

/// The range a surface's OSType names, for the two-plane Y′CbCr OSTypes whose
/// last letter (or `x`/`xf` prefix) says it; `None` for every other OSType.
pub const fn range_of_surface_fourcc(code: u32) -> Option<Range> {
    const VIDEO: [u32; 6] = [
        fourcc(b"420v"),
        fourcc(b"422v"),
        fourcc(b"444v"),
        fourcc(b"x420"),
        fourcc(b"x422"),
        fourcc(b"x444"),
    ];
    const FULL: [u32; 6] = [
        fourcc(b"420f"),
        fourcc(b"422f"),
        fourcc(b"444f"),
        fourcc(b"xf20"),
        fourcc(b"xf22"),
        fourcc(b"xf44"),
    ];
    let mut i = 0;
    while i < VIDEO.len() {
        if VIDEO[i] == code {
            return Some(Range::Video);
        }
        if FULL[i] == code {
            return Some(Range::Full);
        }
        i += 1;
    }
    None
}

/// `(kb, kr)` of red, green and blue under `range` and `matrix`, in the units
/// of the module docs' conversion (measured, metal2vulkan
/// `bind::ycbcr_convert::coefficients`).
pub const fn coefficients(range: Range, matrix: Matrix) -> [(i32, i32); 3] {
    match (range, matrix) {
        (Range::Video, Matrix::Bt601) => [(-9, 13123), (-3218, -6686), (16591, 9)],
        (Range::Video, Matrix::Bt709) => [(-1, 14743), (-1753, -4383), (17373, 1)],
        (Range::Full, Matrix::Bt601) => [(-8, 11483), (-2816, -5850), (14518, 8)],
        (Range::Full, Matrix::Bt709) => [(-1, 12900), (-1534, -3835), (15201, 1)],
        (Range::Video, Matrix::Bt2020) => [(0, 13806), (-1540, -5349), (17614, -1)],
        (Range::Full, Matrix::Bt2020) => [(0, 12080), (-1348, -4680), (15412, -1)],
    }
}

/// Red, green and blue of a converted texel, each `v` of the value `v / 2048`.
pub fn converted_texel(terms: ConversionTerms, bits: u32, [y, cb, cr]: [u32; 3]) -> [u16; 3] {
    let up = 10 - bits;
    let luma = match terms.range {
        Range::Video => 9576 * (((y << up) as i32) - 64),
        Range::Full => {
            let max = (1u32 << bits) - 1;
            256 * ((y * 32768 + max / 2) / max) as i32
        }
    };
    let cb = ((cb << up) as i32) - 512;
    let cr = ((cr << up) as i32) - 512;
    coefficients(terms.range, terms.matrix)
        .map(|(kb, kr)| ((luma + kb * cb + kr * cr + 2048) >> 12).clamp(0, 2048) as u16)
}

/// The half-float bits of `v / 2048` for `v` in `0..=2048`, exactly.
pub const fn half_of_2048ths(v: u16) -> u16 {
    if v == 0 {
        return 0;
    }
    let v = v as u32;
    // `v * 2^-11` is `1.m * 2^(p - 11)` with `p` the top set bit of `v`.
    let p = 31 - v.leading_zeros();
    let exponent = p + 4; // (p - 11) + 15, never below 4: always normal.
    let mantissa = if p <= 10 {
        (v << (10 - p)) & 0x3ff
    } else {
        (v >> (p - 10)) & 0x3ff
    };
    ((exponent << 10) | mantissa) as u16
}

/// One plane's bytes as read: row `r` starts at `r * pitch`.
#[derive(Clone, Copy, Debug)]
pub struct PlaneRows<'a> {
    pub bytes: &'a [u8],
    pub pitch: usize,
}

/// The two chroma taps along one axis for texel `p` of a plane `n` pairs long,
/// each `(pair, weight)`, weights summing to 4 (module docs).
const fn halved_taps(p: u32, n: u32) -> [(u32, u32); 2] {
    let odd = p & 1;
    let a = (p >> 1) + odd;
    let first = if a == 0 { 0 } else { a - 1 };
    let second = if a > n - 1 { n - 1 } else { a };
    let w = if odd == 1 { 3 } else { 1 };
    [(first, w), (second, 4 - w)]
}

fn read_code(row: &[u8], index: usize, ten_bit: bool) -> u32 {
    if ten_bit {
        let at = 2 * index;
        u32::from(u16::from_le_bytes([row[at], row[at + 1]])) >> 6
    } else {
        u32::from(row[index])
    }
}

fn plane_reach(
    plane: &'static str,
    rows: PlaneRows<'_>,
    extent: (u32, u32),
    bytes_per_texel: u32,
) -> Result<(), TwoPlaneRefusal> {
    let tight = u64::from(extent.0) * u64::from(bytes_per_texel);
    if (rows.pitch as u64) < tight {
        return Err(TwoPlaneRefusal::PitchShort {
            plane,
            pitch: rows.pitch as u64,
            tight,
        });
    }
    let need = (rows.pitch as u64)
        .checked_mul(u64::from(extent.1 - 1))
        .and_then(|n| n.checked_add(tight))
        .ok_or(TwoPlaneRefusal::Extent {
            width: extent.0,
            height: extent.1,
        })?;
    if (rows.bytes.len() as u64) < need {
        return Err(TwoPlaneRefusal::PlaneShort {
            plane,
            need,
            have: rows.bytes.len() as u64,
        });
    }
    Ok(())
}

/// The texels a `width` x `height` texture of `layout` reads from its planes,
/// tightly packed in [`CodeLayout::output_layout`].
///
/// `read` must be the one [`CodeLayout::texel_read`] answered for `layout`; a
/// codes layout ignores conversion terms and a converted one with
/// [`TexelRead::Codes`] is refused rather than read as codes.
pub fn convert(
    layout: CodeLayout,
    read: TexelRead,
    width: u32,
    height: u32,
    luma: PlaneRows<'_>,
    chroma: PlaneRows<'_>,
) -> Result<Vec<u8>, TwoPlaneRefusal> {
    if width == 0 || height == 0 {
        return Err(TwoPlaneRefusal::Extent { width, height });
    }
    let terms = match (layout.reads, read) {
        (ConvertibleReads::Codes, _) => None,
        (ConvertibleReads::Converted, TexelRead::Converted(terms)) => Some(terms),
        (ConvertibleReads::Converted, TexelRead::Codes) => {
            return Err(TwoPlaneRefusal::MatrixUnknown {
                pixel_format: layout.pixel_format,
            })
        }
    };
    let (cw, ch) = layout.subsampling.chroma_extent(width, height);
    plane_reach("luma", luma, (width, height), layout.luma_bytes_per_texel())?;
    plane_reach("chroma", chroma, (cw, ch), layout.chroma_bytes_per_texel())?;
    let out_bpp = layout.output_bytes_per_texel() as usize;
    let len = (width as usize)
        .checked_mul(height as usize)
        .and_then(|n| n.checked_mul(out_bpp))
        .ok_or(TwoPlaneRefusal::Extent { width, height })?;
    let mut out = vec![0u8; len];
    let ten_bit = layout.ten_bit;
    let [hx, hy] = layout.subsampling.halved();
    let shift = 2 * (u32::from(hx) + u32::from(hy));
    let bias = if shift == 0 { 0 } else { 1 << (shift - 1) };
    let max = (1u32 << layout.bits()) - 1;
    for y in 0..height {
        let luma_row = &luma.bytes[y as usize * luma.pitch..];
        let ys: [(u32, u32); 2] = if hy {
            halved_taps(y, ch)
        } else {
            [(y, 1), (y, 0)]
        };
        for x in 0..width {
            let code_y = read_code(luma_row, x as usize, ten_bit);
            let xs: [(u32, u32); 2] = if hx {
                halved_taps(x, cw)
            } else {
                [(x, 1), (x, 0)]
            };
            let (mut cb, mut cr) = (0u32, 0u32);
            for (row, wy) in ys {
                if wy == 0 {
                    continue;
                }
                let chroma_row = &chroma.bytes[row as usize * chroma.pitch..];
                for (col, wx) in xs {
                    if wx == 0 {
                        continue;
                    }
                    let w = wx * wy;
                    cb += w * read_code(chroma_row, 2 * col as usize, ten_bit);
                    cr += w * read_code(chroma_row, 2 * col as usize + 1, ten_bit);
                }
            }
            let (cb, cr) = ((cb + bias) >> shift, (cr + bias) >> shift);
            let at = (y as usize * width as usize + x as usize) * out_bpp;
            let texel = &mut out[at..at + out_bpp];
            match terms {
                None if !ten_bit => {
                    texel.copy_from_slice(&[cr as u8, code_y as u8, cb as u8, 0xff])
                }
                None => {
                    let word = (cr & max) | ((code_y & max) << 10) | ((cb & max) << 20) | (3 << 30);
                    texel.copy_from_slice(&word.to_le_bytes());
                }
                Some(terms) => {
                    let [r, g, b] = converted_texel(terms, layout.bits(), [code_y, cb, cr]);
                    for (i, v) in [r, g, b, 2048].into_iter().enumerate() {
                        texel[2 * i..2 * i + 2].copy_from_slice(&half_of_2048ths(v).to_le_bytes());
                    }
                }
            }
        }
    }
    Ok(out)
}

/// Why a two-plane texture's texels were not produced.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TwoPlaneRefusal {
    /// Three codes to a 32-bit word (`_PACKED`): recognised, not converted.
    PackedCodes { pixel_format: u16 },
    /// 12-bit codes: recognised, not converted.
    TwelveBitCodes { pixel_format: u16 },
    /// A format read through a transfer curve (`_sRGB`, `_PQ`, `_HLG`, `_XR`):
    /// recognised, not converted.
    TransferCurve {
        pixel_format: u16,
        curve: &'static str,
    },
    /// A converted format over a surface whose range nothing decoded names.
    RangeUnknown { pixel_format: u16 },
    /// A converted format over a surface whose YCbCr matrix nothing decoded
    /// names.
    MatrixUnknown { pixel_format: u16 },
    /// A plane's row pitch is shorter than one row of its texels.
    PitchShort {
        plane: &'static str,
        pitch: u64,
        tight: u64,
    },
    /// A plane's bytes end before its last texel.
    PlaneShort {
        plane: &'static str,
        need: u64,
        have: u64,
    },
    /// A zero or unrepresentable extent.
    Extent { width: u32, height: u32 },
    /// `RGB10A8_2P_XR10_sRGB` (551): its color channels decode through
    /// Metal's own sRGB tables for the extended range, which are measured only
    /// to within half a `2^-13` step, not recovered exactly.
    ExtendedSrgbTable { pixel_format: u16 },
}

impl reims_vgpu_observe::Decline for TwoPlaneRefusal {
    fn slug(&self) -> &'static str {
        match self {
            Self::PackedCodes { .. } => "two_plane_packed_codes",
            Self::TwelveBitCodes { .. } => "two_plane_twelve_bit_codes",
            Self::TransferCurve { .. } => "two_plane_transfer_curve",
            Self::RangeUnknown { .. } => "two_plane_range_unknown",
            Self::MatrixUnknown { .. } => "two_plane_matrix_unknown",
            Self::PitchShort { .. } => "two_plane_pitch_short",
            Self::PlaneShort { .. } => "two_plane_plane_short",
            Self::Extent { .. } => "two_plane_extent",
            Self::ExtendedSrgbTable { .. } => "two_plane_extended_srgb_table",
        }
    }

    fn fields(&self) -> Vec<(&'static str, String)> {
        let pf = |pf: &u16| alloc::format!("{pf}");
        match self {
            Self::PackedCodes { pixel_format }
            | Self::TwelveBitCodes { pixel_format }
            | Self::RangeUnknown { pixel_format }
            | Self::MatrixUnknown { pixel_format }
            | Self::ExtendedSrgbTable { pixel_format } => vec![("pf", pf(pixel_format))],
            Self::TransferCurve {
                pixel_format,
                curve,
            } => vec![("pf", pf(pixel_format)), ("curve", (*curve).to_string())],
            Self::PitchShort {
                plane,
                pitch,
                tight,
            } => vec![
                ("plane", (*plane).to_string()),
                ("pitch", pitch.to_string()),
                ("tight", tight.to_string()),
            ],
            Self::PlaneShort { plane, need, have } => vec![
                ("plane", (*plane).to_string()),
                ("need", need.to_string()),
                ("have", have.to_string()),
            ],
            Self::Extent { width, height } => {
                vec![("width", width.to_string()), ("height", height.to_string())]
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The chroma filter as the measurement states it, spelled independently
    /// of [`halved_taps`]: the nearer pair weighted 3, the farther 1, the
    /// farther one before for an even texel and after for an odd one, each
    /// clamped to the plane.
    fn reference_taps(p: u32, n: u32) -> [(u32, u32); 2] {
        let nearer = p / 2;
        let farther = if p % 2 == 0 {
            nearer.saturating_sub(1)
        } else {
            (nearer + 1).min(n - 1)
        };
        [(nearer.min(n - 1), 3), (farther, 1)]
    }

    fn reference_codes(
        layout: CodeLayout,
        width: u32,
        height: u32,
        luma: &[u32],
        chroma: &[[u32; 2]],
    ) -> Vec<[u32; 3]> {
        let (cw, ch) = layout.subsampling().chroma_extent(width, height);
        let [hx, hy] = layout.subsampling().halved();
        let mut out = Vec::new();
        for y in 0..height {
            for x in 0..width {
                let xs = if hx {
                    reference_taps(x, cw).to_vec()
                } else {
                    vec![(x, 4)]
                };
                let ys = if hy {
                    reference_taps(y, ch).to_vec()
                } else {
                    vec![(y, 4)]
                };
                let mut s = [0u32; 2];
                for &(cy, wy) in &ys {
                    for &(cx, wx) in &xs {
                        let pair = chroma[(cy * cw + cx) as usize];
                        s[0] += wx * wy * pair[0];
                        s[1] += wx * wy * pair[1];
                    }
                }
                // Weights along each axis sum to 4 (a direct axis's one tap
                // weighs 4), so every texel's weights sum to 16 and rounding
                // the sum to the nearest sixteenth is the measured rounding
                // for all three subsamplings.
                let [cb, cr] = s.map(|v| (v + 8) / 16);
                out.push([luma[(y * width + x) as usize], cb, cr]);
            }
        }
        out
    }

    fn layout(pf: u16) -> CodeLayout {
        two_plane_format(pf)
            .expect("a two-plane ordinal")
            .convertible()
            .expect("a convertible layout")
    }

    fn planes_8bit(
        width: u32,
        height: u32,
        cw: u32,
        ch: u32,
    ) -> (Vec<u8>, Vec<u8>, Vec<u32>, Vec<[u32; 2]>) {
        let luma: Vec<u32> = (0..width * height).map(|i| (i * 37 + 11) % 256).collect();
        let chroma: Vec<[u32; 2]> = (0..cw * ch)
            .map(|i| [(i * 53 + 7) % 256, (i * 91 + 200) % 256])
            .collect();
        let luma_bytes = luma.iter().map(|&v| v as u8).collect();
        let chroma_bytes = chroma
            .iter()
            .flat_map(|p| [p[0] as u8, p[1] as u8])
            .collect();
        (luma_bytes, chroma_bytes, luma, chroma)
    }

    #[test]
    fn the_table_is_closed_and_names_the_measured_ordinals() {
        assert_eq!(TWO_PLANE_ORDINALS.len(), 69);
        assert_eq!(two_plane_format(499), None);
        assert_eq!(
            two_plane_format(501),
            None,
            "a gap in the table is not a format"
        );
        let rgb8_444 = two_plane_format(542).unwrap();
        assert_eq!(
            (
                rgb8_444.bits,
                rgb8_444.subsampling,
                rgb8_444.packed,
                rgb8_444.reads
            ),
            (CodeBits::Eight, Subsampling::Chroma444, false, Reads::Codes)
        );
        // Every ordinal answers for itself.
        for &pf in TWO_PLANE_ORDINALS {
            assert_eq!(two_plane_format(pf).map(|f| f.pixel_format), Some(pf));
        }
    }

    #[test]
    fn only_unpacked_eight_and_ten_bit_codes_and_conversions_are_convertible() {
        for pf in 540..=545 {
            assert_eq!(layout(pf).reads(), ConvertibleReads::Codes, "{pf}");
        }
        for pf in [500, 502, 503, 505, 506, 507] {
            assert_eq!(layout(pf).reads(), ConvertibleReads::Converted, "{pf}");
        }
        let refused = |pf| two_plane_format(pf).unwrap().convertible().unwrap_err();
        assert_eq!(
            refused(546),
            TwoPlaneRefusal::PackedCodes { pixel_format: 546 }
        );
        assert_eq!(
            refused(570),
            TwoPlaneRefusal::TwelveBitCodes { pixel_format: 570 }
        );
        assert_eq!(
            refused(520),
            TwoPlaneRefusal::TransferCurve {
                pixel_format: 520,
                curve: "srgb"
            }
        );
        assert_eq!(
            refused(564),
            TwoPlaneRefusal::TransferCurve {
                pixel_format: 564,
                curve: "pq"
            }
        );
    }

    #[test]
    fn a_converted_layout_names_the_term_it_is_missing_and_a_codes_one_needs_none() {
        let ycbcr = layout(500);
        assert_eq!(
            ycbcr.texel_read(None, Some(Matrix::Bt709)),
            Err(TwoPlaneRefusal::RangeUnknown { pixel_format: 500 })
        );
        assert_eq!(
            ycbcr.texel_read(Some(Range::Video), None),
            Err(TwoPlaneRefusal::MatrixUnknown { pixel_format: 500 })
        );
        assert_eq!(layout(542).texel_read(None, None), Ok(TexelRead::Codes));
    }

    #[test]
    fn the_range_comes_from_the_surface_ostype() {
        assert_eq!(range_of_surface_fourcc(fourcc(b"420v")), Some(Range::Video));
        assert_eq!(range_of_surface_fourcc(fourcc(b"420f")), Some(Range::Full));
        assert_eq!(range_of_surface_fourcc(fourcc(b"xf44")), Some(Range::Full));
        assert_eq!(range_of_surface_fourcc(fourcc(b"BGRA")), None);
    }

    /// `RGB8_444_2P` (542, the login window's compute input): every texel its
    /// own luma code and its own pair, red Cr, green Y, blue Cb, alpha 1.
    #[test]
    fn rgb8_444_reads_each_texels_own_codes() {
        let (w, h) = (5, 3);
        let (luma, chroma, ly, cc) = planes_8bit(w, h, w, h);
        let out = convert(
            layout(542),
            TexelRead::Codes,
            w,
            h,
            PlaneRows {
                bytes: &luma,
                pitch: w as usize,
            },
            PlaneRows {
                bytes: &chroma,
                pitch: 2 * w as usize,
            },
        )
        .unwrap();
        assert_eq!(out.len(), (w * h * 4) as usize);
        for i in 0..(w * h) as usize {
            assert_eq!(
                out[4 * i..4 * i + 4],
                [cc[i][1] as u8, ly[i] as u8, cc[i][0] as u8, 0xff],
                "texel {i}"
            );
        }
    }

    /// `RGB8_420_2P` (540) through the 9/3/3/1 filter, against the reference
    /// spelling, at odd extents so the clamp at the last pair is exercised.
    #[test]
    fn rgb8_420_filters_chroma_nine_three_three_one() {
        let (w, h) = (7, 5);
        let l = layout(540);
        let (cw, ch) = l.subsampling().chroma_extent(w, h);
        assert_eq!((cw, ch), (4, 3));
        let (luma, chroma, ly, cc) = planes_8bit(w, h, cw, ch);
        let out = convert(
            l,
            TexelRead::Codes,
            w,
            h,
            PlaneRows {
                bytes: &luma,
                pitch: w as usize,
            },
            PlaneRows {
                bytes: &chroma,
                pitch: 2 * cw as usize,
            },
        )
        .unwrap();
        let want = reference_codes(l, w, h, &ly, &cc);
        for (i, [y, cb, cr]) in want.into_iter().enumerate() {
            assert_eq!(
                out[4 * i..4 * i + 4],
                [cr as u8, y as u8, cb as u8, 0xff],
                "texel {i}"
            );
        }
        // One texel by hand: (1, 1) reads pairs (0,0)x9, (1,0)x3, (0,1)x3, (1,1)x1.
        let s = |c: usize| {
            9 * cc[0][c] + 3 * cc[1][c] + 3 * cc[cw as usize][c] + cc[cw as usize + 1][c]
        };
        let texel = (w + 1) as usize;
        assert_eq!(out[4 * texel + 2] as u32, (s(0) + 8) >> 4, "Cb");
        assert_eq!(out[4 * texel] as u32, (s(1) + 8) >> 4, "Cr");
    }

    /// `RGB8_422_2P` (541): the row's two nearest pairs, 3 and 1,
    /// `(S + 2) >> 2`.
    #[test]
    fn rgb8_422_filters_chroma_three_one_across() {
        let (w, h) = (6, 2);
        let l = layout(541);
        let (cw, ch) = l.subsampling().chroma_extent(w, h);
        let (luma, chroma, _, cc) = planes_8bit(w, h, cw, ch);
        let out = convert(
            l,
            TexelRead::Codes,
            w,
            h,
            PlaneRows {
                bytes: &luma,
                pitch: w as usize,
            },
            PlaneRows {
                bytes: &chroma,
                pitch: 2 * cw as usize,
            },
        )
        .unwrap();
        // Texel (2, 1): even, so pair 1 weighted 3 and pair 0 weighted 1.
        let row = cw as usize;
        let cb = (3 * cc[row + 1][0] + cc[row][0] + 2) >> 2;
        let cr = (3 * cc[row + 1][1] + cc[row][1] + 2) >> 2;
        let at = 4 * (w as usize + 2);
        assert_eq!((out[at + 2] as u32, out[at] as u32), (cb, cr));
    }

    /// `RGB10_420_2P` (543): codes in the high ten bits of 16-bit words, out as
    /// `RGB10A2Unorm` with red Cr in the low bits and alpha 3.
    #[test]
    fn rgb10_codes_come_from_the_high_bits_and_pack_as_rgb10a2() {
        let (w, h) = (2, 2);
        let l = layout(543);
        assert_eq!(l.output_layout(), TexelLayout::Rgb10a2Unorm);
        let word = |code: u16| (code << 6 | 0x15).to_le_bytes(); // junk in the low six bits
        let luma: Vec<u8> = [100u16, 200, 300, 1023]
            .iter()
            .flat_map(|&c| word(c))
            .collect();
        let chroma: Vec<u8> = [512u16, 64].iter().flat_map(|&c| word(c)).collect();
        let out = convert(
            l,
            TexelRead::Codes,
            w,
            h,
            PlaneRows {
                bytes: &luma,
                pitch: 4,
            },
            PlaneRows {
                bytes: &chroma,
                pitch: 4,
            },
        )
        .unwrap();
        // One pair for the whole 2x2: every tap reads it, so it comes back exact.
        for (i, y) in [100u32, 200, 300, 1023].into_iter().enumerate() {
            let got = u32::from_le_bytes(out[4 * i..4 * i + 4].try_into().unwrap());
            assert_eq!(got, 64 | (y << 10) | (512 << 20) | (3 << 30), "texel {i}");
        }
    }

    #[test]
    fn a_plane_shorter_than_its_rows_is_refused_by_name() {
        let l = layout(542);
        let luma = [0u8; 8];
        let chroma = [0u8; 15];
        assert_eq!(
            convert(
                l,
                TexelRead::Codes,
                4,
                2,
                PlaneRows {
                    bytes: &luma,
                    pitch: 4
                },
                PlaneRows {
                    bytes: &chroma,
                    pitch: 8
                },
            ),
            Err(TwoPlaneRefusal::PlaneShort {
                plane: "chroma",
                need: 16,
                have: 15
            })
        );
        assert_eq!(
            convert(
                l,
                TexelRead::Codes,
                4,
                2,
                PlaneRows {
                    bytes: &luma,
                    pitch: 3
                },
                PlaneRows {
                    bytes: &chroma,
                    pitch: 8
                },
            ),
            Err(TwoPlaneRefusal::PitchShort {
                plane: "luma",
                pitch: 3,
                tight: 4
            })
        );
    }

    /// Values of the measured conversion, worked by hand from the formula.
    #[test]
    fn the_conversion_matches_the_measured_formula() {
        let video709 = ConversionTerms {
            range: Range::Video,
            matrix: Matrix::Bt709,
        };
        // White and black of video range, and full range's white.
        assert_eq!(converted_texel(video709, 8, [235, 128, 128]), [2048; 3]);
        assert_eq!(converted_texel(video709, 8, [16, 128, 128]), [0; 3]);
        let full601 = ConversionTerms {
            range: Range::Full,
            matrix: Matrix::Bt601,
        };
        assert_eq!(converted_texel(full601, 8, [255, 128, 128]), [2048; 3]);
        // Y 128, Cb 128, Cr 255 under video BT.709:
        //   L = 9576 * (512 - 64) = 4 290 048, Cb' = 0, Cr' = 1020 - 512 = 508
        //   red   (L + 14743 * 508 + 2048) >> 12 = 2876, clamped to 2048
        //   green (L -  4383 * 508 + 2048) >> 12 = 504
        //   blue  (L +     1 * 508 + 2048) >> 12 = 1047
        assert_eq!(
            converted_texel(video709, 8, [128, 128, 255]),
            [2048, 504, 1047]
        );
        // The same codes as 10-bit ones read the same.
        assert_eq!(
            converted_texel(video709, 10, [512, 512, 1020]),
            [2048, 504, 1047]
        );
    }

    #[test]
    fn halves_of_the_2048_grid_are_exact() {
        assert_eq!(half_of_2048ths(0), 0);
        assert_eq!(half_of_2048ths(2048), 0x3c00, "1.0");
        assert_eq!(half_of_2048ths(1024), 0x3800, "0.5");
        assert_eq!(half_of_2048ths(1), 0x1000, "2^-11");
        assert_eq!(half_of_2048ths(504), 0x33e0, "0.24609375");
        assert_eq!(half_of_2048ths(2047), 0x3bff, "1 - 2^-11");
    }

    #[test]
    fn a_converted_layout_writes_half_floats_with_alpha_one() {
        let l = layout(503); // YCBCR8_444_2P
        let terms = ConversionTerms {
            range: Range::Video,
            matrix: Matrix::Bt709,
        };
        let out = convert(
            l,
            TexelRead::Converted(terms),
            1,
            1,
            PlaneRows {
                bytes: &[128],
                pitch: 1,
            },
            PlaneRows {
                bytes: &[128, 255],
                pitch: 2,
            },
        )
        .unwrap();
        let halves: Vec<u16> = out
            .chunks_exact(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]))
            .collect();
        assert_eq!(
            halves,
            [0x3c00, half_of_2048ths(504), half_of_2048ths(1047), 0x3c00]
        );
        assert_eq!(
            convert(
                l,
                TexelRead::Codes,
                1,
                1,
                PlaneRows {
                    bytes: &[128],
                    pitch: 1
                },
                PlaneRows {
                    bytes: &[128, 255],
                    pitch: 2
                },
            ),
            Err(TwoPlaneRefusal::MatrixUnknown { pixel_format: 503 }),
            "a converted layout is never read as codes"
        );
    }
}
