//! Two-plane colour textures: where their planes are, and the texels they read.
//!
//! A two-plane texture reaches this device two ways, and both end here:
//!
//! - **In guest memory**, as an object of type 12
//!   ([`OBJECT_TYPE_DUAL_PLANE_TEXTURE`]): a normal texture whose descriptor
//!   carries two `APVTextureDimension` blocks, one per plane, cut from one
//!   allocation. [`guest_planes`] places them.
//! - **Over an IOSurface mapping**, as a texture whose format is a two-plane one
//!   over a surface whose device descriptor lists two planes. [`surface_planes`]
//!   places them from the surface's own plane records.
//!
//! Either way the bind needs one image whose texels are what Metal reads, and
//! neither plane alone is that: plane 0 sampled as a texture is luma read as
//! colour. So both planes are read and converted on the CPU by
//! [`TwoPlaneConversion::convert`], which owns the arithmetic — the two-plane
//! Y′CbCr formats and Apple's `RGB10A8_2P_XR10` (550) alike. The result is
//! memoized on the planes' bytes: every bind re-reads them, and only a change
//! re-converts.
//!
//! # What is executed and what is refused
//!
//! The formats that read their codes (`RGB8_420_2P` .. `RGB10_444_2P`) are a
//! function of the planes alone and are executed. The formats that convert
//! (`YCBCR*_2P`) also read the surface's range and YCbCr matrix. A surface's
//! OSType names its range, so the surface rail recovers it; nothing decoded
//! names the matrix on either rail, nor the range of a guest-memory texture. A
//! converted format therefore refuses with `two_plane_matrix_unknown` (or
//! `two_plane_range_unknown`) rather than guessing BT.601 — see
//! [`recovered_matrix`] for the one place that answer changes.
//!
//! `RGB10A8_2P_XR10` holds a colour word and an alpha **byte** per texel. A
//! type-12 descriptor that states its alpha plane in 4-byte elements (as the
//! macOS 26.4 guest's do) does not say which byte of each element, or which
//! bytes of each row, hold alpha, so it refuses
//! (`two_plane_alpha_layout_unrecovered`) and writes the plane's byte
//! statistics out once (`two_plane_alpha_census`) for recovering the layout.
//! Depth/stencil textures (`Depth32Float_Stencil8`, 260) also arrive as type 12
//! and are no colour format: they refuse as `two_plane_not_color`.

use std::sync::Arc;

use crate::model::{DeviceState, MappingEntry};
use crate::protocol::iosurface_pages::{
    decode_device_surface, device_desc_plane, sample_window_from_device_plane,
};
use crate::protocol::two_plane::{two_plane_conversion, TwoPlaneConversion};
use crate::protocol::ycbcr::{self, Matrix, PlaneRows, Range, TwoPlaneRefusal};
use crate::runtime::decode::resource::{DualPlaneTextureDescriptor, TextureLevelLayout};
use crate::runtime::draw::host_alloc_len;
use crate::runtime::gva_mem;
use crate::runtime::host::{HostMemory, HostOps};
use crate::runtime::mapping_write;

pub use crate::runtime::decode::resource::OBJECT_TYPE_DUAL_PLANE_TEXTURE;

/// Why a two-plane texture was not bound.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum TwoPlaneBindRefusal {
    /// The format is not a two-plane colour format this device converts —
    /// `Depth32Float_Stencil8` (260) among them.
    NotTwoPlane { pixel_format: u16 },
    /// `RGB10A8_2P_XR10` over guest memory whose alpha plane the descriptor
    /// states in elements other than the alpha byte: which byte holds a
    /// texel's alpha is not recovered.
    AlphaLayout { bytes_per_element: u32, pitch: u64 },
    /// The format, its terms or its planes' bytes, as the protocol layer
    /// refused them.
    Layout(TwoPlaneRefusal),
    /// A type-12 object whose descriptor did not decode as two planes.
    Undecoded,
    /// More than one mip level. Metal makes no mipmapped two-plane texture.
    Mipmapped { levels: usize },
    /// A plane's level 0 names no window inside the allocation.
    PlaneWindow {
        plane: &'static str,
        offset: u64,
        span: u64,
        allocation: u64,
    },
    /// A plane's stated geometry is not the one the format gives it.
    PlaneGeometry {
        plane: &'static str,
        width: u32,
        height: u32,
        bytes_per_element: u32,
        want_width: u32,
        want_height: u32,
        want_bytes_per_element: u32,
    },
    /// A plane's span is past what this host can allocate.
    HostLen { plane: &'static str, span: u64 },
    /// The guest's pages for a plane did not read.
    GuestRead {
        plane: &'static str,
        gva: u64,
        len: u64,
    },
    /// A surface whose plane records are not the two planes of this format's
    /// geometry.
    SurfacePlanes {
        plane_count: u32,
        plane: &'static str,
    },
    /// The surface's pages for a plane did not read.
    SurfaceRead {
        plane: &'static str,
        mapping_id: u32,
    },
    /// A storage binding: the converted image is a copy, so a write to it would
    /// reach no plane.
    StorageBind { pixel_format: u16 },
    /// A view that reinterprets the two-plane texture as another format.
    ViewFormat { pixel_format: u16, view: u16 },
}

impl crate::observe::Decline for TwoPlaneBindRefusal {
    fn slug(&self) -> &'static str {
        match self {
            Self::NotTwoPlane { .. } => "two_plane_not_color",
            Self::AlphaLayout { .. } => "two_plane_alpha_layout_unrecovered",
            Self::Layout(inner) => inner.slug(),
            Self::Undecoded => "two_plane_descriptor_undecoded",
            Self::Mipmapped { .. } => "two_plane_mipmapped",
            Self::PlaneWindow { .. } => "two_plane_plane_window",
            Self::PlaneGeometry { .. } => "two_plane_plane_geometry",
            Self::HostLen { .. } => "two_plane_host_len",
            Self::GuestRead { .. } => "two_plane_guest_read",
            Self::SurfacePlanes { .. } => "two_plane_surface_planes",
            Self::SurfaceRead { .. } => "two_plane_surface_read",
            Self::StorageBind { .. } => "two_plane_storage_bind",
            Self::ViewFormat { .. } => "two_plane_view_format",
        }
    }

    fn fields(&self) -> Vec<(&'static str, String)> {
        match self {
            Self::NotTwoPlane { pixel_format } => vec![("pf", pixel_format.to_string())],
            Self::AlphaLayout {
                bytes_per_element,
                pitch,
            } => vec![
                ("bpe", bytes_per_element.to_string()),
                ("pitch", pitch.to_string()),
            ],
            Self::Layout(inner) => inner.fields(),
            Self::Undecoded => Vec::new(),
            Self::Mipmapped { levels } => vec![("levels", levels.to_string())],
            Self::PlaneWindow {
                plane,
                offset,
                span,
                allocation,
            } => vec![
                ("plane", plane.to_string()),
                ("offset", format!("{offset:#x}")),
                ("span", span.to_string()),
                ("alloc", allocation.to_string()),
            ],
            Self::PlaneGeometry {
                plane,
                width,
                height,
                bytes_per_element,
                want_width,
                want_height,
                want_bytes_per_element,
            } => vec![
                ("plane", plane.to_string()),
                ("got", format!("{width}x{height}:bpe={bytes_per_element}")),
                (
                    "want",
                    format!("{want_width}x{want_height}:bpe={want_bytes_per_element}"),
                ),
            ],
            Self::HostLen { plane, span } => {
                vec![("plane", plane.to_string()), ("span", span.to_string())]
            }
            Self::GuestRead { plane, gva, len } => vec![
                ("plane", plane.to_string()),
                ("gva", format!("{gva:#x}")),
                ("len", len.to_string()),
            ],
            Self::SurfacePlanes { plane_count, plane } => vec![
                ("plane_count", plane_count.to_string()),
                ("plane", plane.to_string()),
            ],
            Self::SurfaceRead { plane, mapping_id } => vec![
                ("plane", plane.to_string()),
                ("mapping", mapping_id.to_string()),
            ],
            Self::StorageBind { pixel_format } => vec![("pf", pixel_format.to_string())],
            Self::ViewFormat { pixel_format, view } => {
                vec![("pf", pixel_format.to_string()), ("view", view.to_string())]
            }
        }
    }

    fn owner(&self) -> &'static str {
        match self {
            Self::Layout(inner) => inner.owner(),
            _ => core::any::type_name::<Self>(),
        }
    }
}

crate::observe::decline_display!(TwoPlaneBindRefusal);

impl From<TwoPlaneRefusal> for TwoPlaneBindRefusal {
    fn from(inner: TwoPlaneRefusal) -> Self {
        Self::Layout(inner)
    }
}

/// The YCbCr matrix a two-plane texture was made under, as far as anything
/// this device decodes says.
///
/// **The hook for converted formats.** A converted texel (`YCBCR8_420_2P`, 500)
/// is the surface's `kIOSurfaceYCbCrMatrix` applied to its codes, and no field
/// this device decodes carries that matrix: not the type-12 descriptor's
/// decoded fields, not the IOSurface device descriptor. So the answer is `None`
/// on every rail and every converted bind refuses by name. Once
/// `dual_plane_texture_descriptor` lines from a live guest show where the
/// matrix travels (the descriptor's undecoded header and trailer bytes are
/// written out whole for exactly this), decode it and answer it here; every
/// converted format then executes with no other change.
fn recovered_matrix() -> Option<Matrix> {
    None
}

/// One plane in guest memory: its first texel and its row pitch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct GuestPlane {
    pub gva: u64,
    pub pitch: u64,
    /// Bytes from `gva` to one past the last texel: `pitch * (rows - 1) + tight`.
    pub span: u64,
}

/// Where a two-plane texture's planes are, and what its texels read.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct TwoPlanes<P> {
    pub conversion: TwoPlaneConversion,
    pub width: u32,
    pub height: u32,
    pub planes: [P; 2],
}

/// The conversion a two-plane object's format names.
fn conversion_of(
    pixel_format: u16,
    range: Option<Range>,
) -> Result<TwoPlaneConversion, TwoPlaneBindRefusal> {
    match two_plane_conversion(pixel_format, range, recovered_matrix()) {
        None => Err(TwoPlaneBindRefusal::NotTwoPlane { pixel_format }),
        Some(conversion) => Ok(conversion?),
    }
}

/// A stated bytes-per-element corroborates the plane's role when it is
/// non-zero; zero states nothing, and is not read as agreement or as conflict.
fn bytes_per_element_agrees(stated: u32, want: u32) -> bool {
    stated == 0 || stated == want
}

const PLANE_NAMES: [&str; 2] = ["plane0", "plane1"];

/// Place one plane of a type-12 texture at the geometry it is checked against.
fn place_guest_plane(
    dual: &DualPlaneTextureDescriptor,
    index: usize,
    want: (u32, u32),
    bpe: u32,
    page_shift: u32,
) -> Result<GuestPlane, TwoPlaneBindRefusal> {
    let plane = PLANE_NAMES[index];
    let tex = &dual.planes[index];
    let l: TextureLevelLayout = tex.level(0).copied().unwrap_or_default();
    let stated_bpe = u32::from(tex.bytes_per_element);
    if (l.width, l.height) != want || !bytes_per_element_agrees(stated_bpe, bpe) {
        return Err(TwoPlaneBindRefusal::PlaneGeometry {
            plane,
            width: l.width,
            height: l.height,
            bytes_per_element: stated_bpe,
            want_width: want.0,
            want_height: want.1,
            want_bytes_per_element: bpe,
        });
    }
    let tight = u64::from(want.0) * u64::from(bpe);
    let window = |span| TwoPlaneBindRefusal::PlaneWindow {
        plane,
        offset: l.offset,
        span,
        allocation: tex.allocation_size,
    };
    if l.row_stride < tight {
        return Err(window(0));
    }
    let span = l.read_span(tight as u32).ok_or_else(|| window(0))?;
    let (gva, _) = tex.level_gva(0, page_shift).ok_or_else(|| window(span))?;
    if tex.allocation_size != 0 && l.offset.saturating_add(span) > tex.allocation_size {
        return Err(window(span));
    }
    Ok(GuestPlane {
        gva,
        pitch: l.row_stride,
        span,
    })
}

/// Place a type-12 texture's planes in guest memory.
///
/// Each plane is checked against what the format makes of it — extent, and
/// bytes per element where the block states one — so a descriptor whose planes
/// are not the format's refuses instead of being read as something else.
pub(crate) fn guest_planes(
    dual: &DualPlaneTextureDescriptor,
    page_shift: u32,
) -> Result<TwoPlanes<GuestPlane>, TwoPlaneBindRefusal> {
    // A guest-memory texture has no surface, so nothing names its range.
    let conversion = conversion_of(dual.pixel_format(), None)?;
    let levels = dual.planes[0].levels.len();
    if levels != 1 {
        return Err(TwoPlaneBindRefusal::Mipmapped { levels });
    }
    let l0 = dual.planes[0].level(0).copied().unwrap_or_default();
    let (width, height) = (l0.width, l0.height);
    let extents = conversion.plane_extents(width, height);
    let bpe = conversion.plane_bytes_per_texel();
    // The alpha plane of `RGB10A8_2P_XR10` is a byte a texel. A block stating
    // wider elements does not say where in each the byte is.
    if conversion == TwoPlaneConversion::ExtendedRange {
        let stated = u32::from(dual.planes[1].bytes_per_element);
        if !bytes_per_element_agrees(stated, bpe[1]) {
            let pitch = dual.planes[1].level(0).map_or(0, |l| l.row_stride);
            return Err(TwoPlaneBindRefusal::AlphaLayout {
                bytes_per_element: stated,
                pitch,
            });
        }
    }
    let plane0 = place_guest_plane(dual, 0, extents[0], bpe[0], page_shift)?;
    let plane1 = place_guest_plane(dual, 1, extents[1], bpe[1], page_shift)?;
    Ok(TwoPlanes {
        conversion,
        width,
        height,
        planes: [plane0, plane1],
    })
}

/// How many distinct alpha planes [`note_alpha_census`] writes out in one
/// process.
const ALPHA_CENSUS_CAP: u32 = 16;

/// Write out what an `RGB10A8_2P_XR10` texture's unrecovered alpha plane holds,
/// once per extent, so a live guest's bytes can settle where the alpha byte is.
///
/// Both planes are read as the descriptor states them. For the alpha plane it
/// counts the non-zero bytes in each lane of the stated 4-byte elements
/// (`lanes=`), in the first `w` bytes of each row against the rest of the row
/// (`row_head=`/`row_tail=`), and in the plane's first `align(w, 16) * h`
/// bytes — Metal's own alpha rows — against the bytes past them
/// (`metal_rows=`/`past_metal_rows=`), with the first 32 bytes of rows 0 and
/// `h / 2`. `color_nonzero=` says whether the colour plane holds anything at
/// all; an all-zero texture settles nothing.
fn note_alpha_census<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    task_id: u32,
    texture_ref: u32,
    dual: &DualPlaneTextureDescriptor,
    site: crate::runtime::render_writeback::SettleSite,
) {
    let l0 = dual.planes[0].level(0).copied().unwrap_or_default();
    let (w, h) = (l0.width, l0.height);
    let key = (u64::from(w) << 32) | u64::from(h);
    if !crate::observe::first_sight("two_plane_alpha_census", key) {
        return;
    }
    if !(0..ALPHA_CENSUS_CAP)
        .any(|slot| crate::observe::first_sight("two_plane_alpha_census_slot", slot.into()))
    {
        return;
    }
    let stated = u32::from(dual.planes[1].bytes_per_element).max(1);
    let read = |state: &mut DeviceState, host: &mut M, index: usize, bpe: u32| {
        let plane = place_guest_plane(dual, index, (w, h), bpe, state.page_shift).ok()?;
        let len = host_alloc_len(plane.span)?;
        crate::runtime::writeback_debt::settle_for_texture(
            state,
            host,
            task_id,
            texture_ref,
            plane.gva,
            plane.span,
            site,
        );
        let mut bytes = vec![0u8; len];
        gva_mem::read_task_gva_by_id(
            host,
            &state.tasks,
            task_id,
            plane.gva,
            &mut bytes,
            state.page_shift,
        )
        .ok()?;
        Some((plane.pitch as usize, bytes))
    };
    let (Some((cpitch, color)), Some((apitch, alpha))) =
        (read(state, host, 0, 4), read(state, host, 1, stated))
    else {
        crate::observe::off(format!(
            "two_plane_alpha_census task={task_id} ref={texture_ref} {w}x{h} reason=unread"
        ));
        return;
    };
    let (w, h) = (w as usize, h as usize);
    let color_nonzero = (0..h)
        .map(|y| {
            color[y * cpitch..y * cpitch + 4 * w]
                .chunks_exact(4)
                .filter(|t| t.iter().any(|&b| b != 0))
                .count()
        })
        .sum::<usize>();
    let mut lanes = [0usize; 4];
    let (mut row_head, mut row_tail) = (0usize, 0usize);
    let row_len = (stated as usize * w).min(apitch);
    for y in 0..h {
        let row = &alpha[y * apitch..(y * apitch + row_len).min(alpha.len())];
        for (i, &b) in row.iter().enumerate() {
            if b != 0 {
                lanes[i % 4] += 1;
                if i < w {
                    row_head += 1;
                } else {
                    row_tail += 1;
                }
            }
        }
    }
    let metal_rows = (w.div_ceil(16) * 16 * h).min(alpha.len());
    let nonzero = |b: &[u8]| b.iter().filter(|&&v| v != 0).count();
    let hex = |at: usize| -> String {
        alpha[at.min(alpha.len())..(at + 32).min(alpha.len())]
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect()
    };
    crate::observe::off(format!(
        "two_plane_alpha_census task={task_id} ref={texture_ref} pf={} {w}x{h} bpe={stated} \
         pitch={apitch} color_nonzero={color_nonzero} lanes={lanes:?} row_head={row_head} \
         row_tail={row_tail} metal_rows={} past_metal_rows={} row0={} row_mid={}",
        dual.pixel_format(),
        nonzero(&alpha[..metal_rows]),
        nonzero(&alpha[metal_rows..]),
        hex(0),
        hex((h / 2) * apitch),
    ));
}

/// A type-12 texture's texels, converted: placement and read in one, with the
/// census an unrecovered alpha plane writes out.
pub(crate) fn guest_texels<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    task_id: u32,
    texture_ref: u32,
    dual: &DualPlaneTextureDescriptor,
    site: crate::runtime::render_writeback::SettleSite,
) -> Result<TwoPlaneTexels, TwoPlaneBindRefusal> {
    let planes = match guest_planes(dual, state.page_shift) {
        Err(refusal @ TwoPlaneBindRefusal::AlphaLayout { .. }) => {
            note_alpha_census(state, host, task_id, texture_ref, dual, site);
            return Err(refusal);
        }
        placed => placed?,
    };
    load_guest(state, host, task_id, texture_ref, &planes, site)
}

/// One plane of an IOSurface mapping, as the existing surface readers take it.
pub(crate) type SurfacePlane = mapping_write::SurfaceWindow;

/// Place a texture of a two-plane `pixel_format` over an IOSurface mapping.
///
/// The texture is over the surface's plane 0, and Metal reads plane 1 beside
/// it; the surface's own plane records say where each is. A surface that does
/// not list exactly two planes matching the format's geometry refuses — over
/// plane 1, or over a three-plane surface, Metal reads planes this device would
/// have to guess at.
///
/// A Y′CbCr format's range comes from the surface's OSType, which its device
/// descriptor carries.
pub(crate) fn surface_planes(
    mapping: &MappingEntry,
    pixel_format: u16,
    width: u32,
    height: u32,
) -> Result<TwoPlanes<SurfacePlane>, TwoPlaneBindRefusal> {
    let surface = decode_device_surface(&mapping.device_desc);
    let range = surface
        .as_ref()
        .and_then(|s| ycbcr::range_of_surface_fourcc(s.pixel_format));
    let conversion = conversion_of(pixel_format, range)?;
    let plane_count = surface.as_ref().map_or(0, |s| u32::from(s.plane_count));
    let formats = conversion.plane_formats();
    let extents = conversion.plane_extents(width, height);
    let bpes = conversion.plane_bytes_per_texel();
    let place = |index: usize| -> Result<SurfacePlane, TwoPlaneBindRefusal> {
        let refusal = TwoPlaneBindRefusal::SurfacePlanes {
            plane_count,
            plane: PLANE_NAMES[index],
        };
        if plane_count != 2 {
            return Err(refusal);
        }
        let (record, _) = device_desc_plane(&mapping.device_desc, index as u32).ok_or(refusal)?;
        if !bytes_per_element_agrees(u32::from(record.bytes_per_element), bpes[index]) {
            return Err(refusal);
        }
        let (w, h) = extents[index];
        let (base_off, bpr, span_end) =
            sample_window_from_device_plane(&record, formats[index], w, h).ok_or(refusal)?;
        Ok(SurfacePlane {
            base_off,
            bpr,
            span_end,
            bpp: bpes[index],
        })
    };
    Ok(TwoPlanes {
        conversion,
        width,
        height,
        planes: [place(0)?, place(1)?],
    })
}

/// A two-plane texture's texels, converted.
#[derive(Clone, Debug)]
pub(crate) struct TwoPlaneTexels {
    pub width: u32,
    pub height: u32,
    pub conversion: TwoPlaneConversion,
    /// Tightly packed in [`TwoPlaneConversion::output_layout`].
    pub bytes: Arc<Vec<u8>>,
    /// From [`DeviceState::next_sampled_content_generation`]: changes exactly
    /// when the planes' bytes did.
    pub generation: u64,
    /// The guest address of plane 0's first texel, for a guest-memory texture.
    pub plane0_gva: Option<u64>,
}

/// Where a memoized conversion's planes were read from.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum TwoPlaneSource {
    Guest { gvas: [u64; 2] },
    Surface { mapping_id: u32, offsets: [u64; 2] },
}

/// The key of [`DeviceState::two_plane_memo`]: everything besides the planes'
/// bytes that the converted texels depend on.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct TwoPlaneMemoKey {
    task_id: u32,
    source: TwoPlaneSource,
    width: u32,
    height: u32,
    pitches: [u64; 2],
    conversion: TwoPlaneConversion,
}

/// See [`DeviceState::two_plane_memo`].
#[derive(Clone, Debug)]
pub struct TwoPlaneMemo {
    /// The planes' bytes as last read, plane 0 then plane 1.
    native: Vec<u8>,
    texels: Arc<Vec<u8>>,
    generation: u64,
}

/// Convert `native` (plane 0's bytes, then plane 1's) unless the memo already
/// holds a conversion of exactly these bytes under `key`.
fn convert_memoized(
    state: &mut DeviceState,
    key: TwoPlaneMemoKey,
    native: Vec<u8>,
    plane0_len: usize,
    plane0_gva: Option<u64>,
) -> Result<TwoPlaneTexels, TwoPlaneBindRefusal> {
    let done = |bytes, generation| TwoPlaneTexels {
        width: key.width,
        height: key.height,
        conversion: key.conversion,
        bytes,
        generation,
        plane0_gva,
    };
    match state.two_plane_memo.get_touch(&key) {
        Some(memo) if memo.native == native => {
            crate::runtime::drain::note_store_route("two_plane_memo_hit");
            return Ok(done(memo.texels.clone(), memo.generation));
        }
        Some(_) => crate::runtime::drain::note_store_route("two_plane_memo_changed"),
        None => crate::runtime::drain::note_store_route("two_plane_memo_absent"),
    }
    let (plane0, plane1) = native.split_at(plane0_len);
    let texels = key.conversion.convert(
        key.width,
        key.height,
        PlaneRows {
            bytes: plane0,
            pitch: key.pitches[0] as usize,
        },
        PlaneRows {
            bytes: plane1,
            pitch: key.pitches[1] as usize,
        },
    )?;
    let generation = state.next_sampled_content_generation();
    let texels = Arc::new(texels);
    let entry_bytes = native.len() + texels.len();
    state.two_plane_memo.insert(
        key,
        TwoPlaneMemo {
            native,
            texels: texels.clone(),
            generation,
        },
        entry_bytes,
    );
    Ok(done(texels, generation))
}

/// Read a type-12 texture's planes out of guest memory and convert them.
///
/// Each plane's span is settled against outstanding write-back first, the
/// same obligation every CPU read of a named resource's guest bytes has.
pub(crate) fn load_guest<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    task_id: u32,
    texture_ref: u32,
    planes: &TwoPlanes<GuestPlane>,
    site: crate::runtime::render_writeback::SettleSite,
) -> Result<TwoPlaneTexels, TwoPlaneBindRefusal> {
    let mut native = Vec::new();
    let mut plane0_len = 0;
    for (index, p) in planes.planes.into_iter().enumerate() {
        let plane = PLANE_NAMES[index];
        let len = host_alloc_len(p.span).ok_or(TwoPlaneBindRefusal::HostLen {
            plane,
            span: p.span,
        })?;
        crate::runtime::writeback_debt::settle_for_texture(
            state,
            host,
            task_id,
            texture_ref,
            p.gva,
            p.span,
            site,
        );
        let at = native.len();
        native.resize(at + len, 0);
        gva_mem::read_task_gva_by_id(
            host,
            &state.tasks,
            task_id,
            p.gva,
            &mut native[at..],
            state.page_shift,
        )
        .map_err(|_| TwoPlaneBindRefusal::GuestRead {
            plane,
            gva: p.gva,
            len: p.span,
        })?;
        if index == 0 {
            plane0_len = len;
        }
    }
    let key = TwoPlaneMemoKey {
        task_id,
        source: TwoPlaneSource::Guest {
            gvas: planes.planes.map(|p| p.gva),
        },
        width: planes.width,
        height: planes.height,
        pitches: planes.planes.map(|p| p.pitch),
        conversion: planes.conversion,
    };
    convert_memoized(state, key, native, plane0_len, Some(planes.planes[0].gva))
}

/// Read a two-plane texture's planes out of an IOSurface mapping and convert
/// them.
pub(crate) fn load_surface<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    task_id: u32,
    mapping_id: u32,
    planes: &TwoPlanes<SurfacePlane>,
) -> Result<TwoPlaneTexels, TwoPlaneBindRefusal> {
    let extents = planes.conversion.plane_extents(planes.width, planes.height);
    let mut native = Vec::new();
    let mut plane0_len = 0;
    let mut pitches = [0u64; 2];
    for (index, window) in planes.planes.into_iter().enumerate() {
        let plane = PLANE_NAMES[index];
        let (w, h) = extents[index];
        let tight = w * window.bpp;
        let span = u64::from(tight) * u64::from(h);
        let len = host_alloc_len(span).ok_or(TwoPlaneBindRefusal::HostLen { plane, span })?;
        let at = native.len();
        native.resize(at + len, 0);
        if !mapping_write::read_rect_raw_at(
            state,
            host,
            mapping_id,
            window,
            mapping_write::Rect {
                origin_x: 0,
                origin_y: 0,
                width: w,
                height: h,
            },
            &mut native[at..],
            tight,
        ) {
            return Err(TwoPlaneBindRefusal::SurfaceRead { plane, mapping_id });
        }
        pitches[index] = u64::from(tight);
        if index == 0 {
            plane0_len = len;
        }
    }
    let key = TwoPlaneMemoKey {
        task_id,
        source: TwoPlaneSource::Surface {
            mapping_id,
            offsets: planes.planes.map(|p| p.base_off),
        },
        width: planes.width,
        height: planes.height,
        pitches,
        conversion: planes.conversion,
    };
    convert_memoized(state, key, native, plane0_len, None)
}

#[cfg(test)]
pub(crate) mod tests;
