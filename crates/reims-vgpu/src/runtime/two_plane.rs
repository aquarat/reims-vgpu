//! Two-plane Y′CbCr textures: where their planes are, and the texels they read.
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
//! [`crate::protocol::ycbcr::convert`], which owns the arithmetic. The result is
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

use std::sync::Arc;

use crate::model::{DeviceState, MappingEntry};
use crate::protocol::iosurface_pages::{
    decode_device_surface, device_desc_plane, sample_window_from_device_plane,
};
use crate::protocol::ycbcr::{
    self, CodeLayout, Matrix, PlaneRows, Range, TexelRead, TwoPlaneRefusal,
};
use crate::runtime::decode::resource::{DualPlaneTextureDescriptor, TextureLevelLayout};
use crate::runtime::draw::host_alloc_len;
use crate::runtime::gva_mem;
use crate::runtime::host::{HostMemory, HostOps};
use crate::runtime::mapping_write;

pub use crate::runtime::decode::resource::OBJECT_TYPE_DUAL_PLANE_TEXTURE;

/// Why a two-plane texture was not bound.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum TwoPlaneBindRefusal {
    /// The format is not one of the two-plane Y′CbCr ordinals.
    NotTwoPlane { pixel_format: u16 },
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
    /// A surface whose plane records are not a luma plane and a chroma plane
    /// of this format's geometry.
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
            Self::NotTwoPlane { .. } => "two_plane_not_ycbcr",
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
    pub layout: CodeLayout,
    pub read: TexelRead,
    pub width: u32,
    pub height: u32,
    pub luma: P,
    pub chroma: P,
}

/// The format a two-plane object names, as a layout the converter takes, and
/// how its texels read.
fn layout_and_read(
    pixel_format: u16,
    range: Option<Range>,
) -> Result<(CodeLayout, TexelRead), TwoPlaneBindRefusal> {
    let format = ycbcr::two_plane_format(pixel_format)
        .ok_or(TwoPlaneBindRefusal::NotTwoPlane { pixel_format })?;
    let layout = format.convertible()?;
    let read = layout.texel_read(range, recovered_matrix())?;
    Ok((layout, read))
}

/// A stated bytes-per-element corroborates the plane's role when it is
/// non-zero; zero states nothing, and is not read as agreement or as conflict.
fn bytes_per_element_agrees(stated: u32, want: u32) -> bool {
    stated == 0 || stated == want
}

/// Place a type-12 texture's planes in guest memory.
///
/// Plane 0 is the luma plane at the texture's extent; plane 1 is the chroma
/// plane at the format's chroma extent. Each is checked against what the
/// format makes of it — extent, and bytes per element where the block states
/// one — so a descriptor whose planes are not the format's refuses instead of
/// being read as something else.
pub(crate) fn guest_planes(
    dual: &DualPlaneTextureDescriptor,
    page_shift: u32,
) -> Result<TwoPlanes<GuestPlane>, TwoPlaneBindRefusal> {
    // A guest-memory texture has no surface, so nothing names its range.
    let (layout, read) = layout_and_read(dual.pixel_format(), None)?;
    let levels = dual.planes[0].levels.len();
    if levels != 1 {
        return Err(TwoPlaneBindRefusal::Mipmapped { levels });
    }
    let place = |plane: &'static str,
                 index: usize,
                 want: (u32, u32),
                 bpe: u32|
     -> Result<GuestPlane, TwoPlaneBindRefusal> {
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
    };
    let luma0 = dual.planes[0].level(0).copied().unwrap_or_default();
    let (width, height) = (luma0.width, luma0.height);
    let luma = place("luma", 0, (width, height), layout.luma_bytes_per_texel())?;
    let chroma = place(
        "chroma",
        1,
        layout.subsampling().chroma_extent(width, height),
        layout.chroma_bytes_per_texel(),
    )?;
    Ok(TwoPlanes {
        layout,
        read,
        width,
        height,
        luma,
        chroma,
    })
}

/// One plane of an IOSurface mapping, as the existing surface readers take it.
pub(crate) type SurfacePlane = mapping_write::SurfaceWindow;

/// Place a texture of a two-plane `pixel_format` over an IOSurface mapping.
///
/// The texture is over the surface's plane 0, and Metal reads plane 1 as its
/// chroma; the surface's own plane records say where each is. A surface that
/// does not list exactly two planes matching the format's luma and chroma
/// geometry refuses — over plane 1, or over a three-plane surface, Metal reads
/// planes this device would have to guess at.
///
/// The range comes from the surface's OSType, which its device descriptor
/// carries.
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
    let (layout, read) = layout_and_read(pixel_format, range)?;
    let plane_count = surface.as_ref().map_or(0, |s| u32::from(s.plane_count));
    let place = |plane: &'static str,
                 index: u32,
                 format: u16,
                 extent: (u32, u32),
                 bpe: u32|
     -> Result<SurfacePlane, TwoPlaneBindRefusal> {
        let refusal = TwoPlaneBindRefusal::SurfacePlanes { plane_count, plane };
        if plane_count != 2 {
            return Err(refusal);
        }
        let (record, _) = device_desc_plane(&mapping.device_desc, index).ok_or(refusal)?;
        if !bytes_per_element_agrees(u32::from(record.bytes_per_element), bpe) {
            return Err(refusal);
        }
        let (base_off, bpr, span_end) =
            sample_window_from_device_plane(&record, format, extent.0, extent.1).ok_or(refusal)?;
        Ok(SurfacePlane {
            base_off,
            bpr,
            span_end,
            bpp: bpe,
        })
    };
    let luma = place(
        "luma",
        0,
        layout.luma_plane_format(),
        (width, height),
        layout.luma_bytes_per_texel(),
    )?;
    let chroma = place(
        "chroma",
        1,
        layout.chroma_plane_format(),
        layout.subsampling().chroma_extent(width, height),
        layout.chroma_bytes_per_texel(),
    )?;
    Ok(TwoPlanes {
        layout,
        read,
        width,
        height,
        luma,
        chroma,
    })
}

/// A two-plane texture's texels, converted.
#[derive(Clone, Debug)]
pub(crate) struct TwoPlaneTexels {
    pub width: u32,
    pub height: u32,
    pub layout: CodeLayout,
    /// Tightly packed in [`CodeLayout::output_layout`].
    pub bytes: Arc<Vec<u8>>,
    /// From [`DeviceState::next_sampled_content_generation`]: changes exactly
    /// when the planes' bytes did.
    pub generation: u64,
}

/// Where a memoized conversion's planes were read from.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum TwoPlaneSource {
    Guest {
        luma_gva: u64,
        chroma_gva: u64,
    },
    Surface {
        mapping_id: u32,
        luma_off: u64,
        chroma_off: u64,
    },
}

/// The key of [`DeviceState::two_plane_memo`]: everything besides the planes'
/// bytes that the converted texels depend on.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct TwoPlaneMemoKey {
    task_id: u32,
    source: TwoPlaneSource,
    width: u32,
    height: u32,
    luma_pitch: u64,
    chroma_pitch: u64,
    layout: CodeLayout,
    read: TexelRead,
}

/// See [`DeviceState::two_plane_memo`].
#[derive(Clone, Debug)]
pub struct TwoPlaneMemo {
    /// The planes' bytes as last read, luma then chroma.
    native: Vec<u8>,
    texels: Arc<Vec<u8>>,
    generation: u64,
}

/// Convert `native` (luma bytes, then chroma bytes) unless the memo already
/// holds a conversion of exactly these bytes under `key`.
fn convert_memoized<P>(
    state: &mut DeviceState,
    key: TwoPlaneMemoKey,
    planes: &TwoPlanes<P>,
    native: Vec<u8>,
    luma_len: usize,
) -> Result<TwoPlaneTexels, TwoPlaneBindRefusal> {
    let done = |bytes, generation| TwoPlaneTexels {
        width: planes.width,
        height: planes.height,
        layout: planes.layout,
        bytes,
        generation,
    };
    match state.two_plane_memo.get_touch(&key) {
        Some(memo) if memo.native == native => {
            crate::runtime::drain::note_store_route("two_plane_memo_hit");
            return Ok(done(memo.texels.clone(), memo.generation));
        }
        Some(_) => crate::runtime::drain::note_store_route("two_plane_memo_changed"),
        None => crate::runtime::drain::note_store_route("two_plane_memo_absent"),
    }
    let (luma, chroma) = native.split_at(luma_len);
    let texels = ycbcr::convert(
        planes.layout,
        planes.read,
        planes.width,
        planes.height,
        PlaneRows {
            bytes: luma,
            pitch: key.luma_pitch as usize,
        },
        PlaneRows {
            bytes: chroma,
            pitch: key.chroma_pitch as usize,
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
    let mut luma_len = 0;
    for (plane, p) in [("luma", planes.luma), ("chroma", planes.chroma)] {
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
        if plane == "luma" {
            luma_len = len;
        }
    }
    let key = TwoPlaneMemoKey {
        task_id,
        source: TwoPlaneSource::Guest {
            luma_gva: planes.luma.gva,
            chroma_gva: planes.chroma.gva,
        },
        width: planes.width,
        height: planes.height,
        luma_pitch: planes.luma.pitch,
        chroma_pitch: planes.chroma.pitch,
        layout: planes.layout,
        read: planes.read,
    };
    convert_memoized(state, key, planes, native, luma_len)
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
    let (cw, ch) = planes
        .layout
        .subsampling()
        .chroma_extent(planes.width, planes.height);
    let mut native = Vec::new();
    let mut luma_len = 0;
    let mut pitches = [0u64; 2];
    for (i, (plane, window, (w, h))) in [
        ("luma", planes.luma, (planes.width, planes.height)),
        ("chroma", planes.chroma, (cw, ch)),
    ]
    .into_iter()
    .enumerate()
    {
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
        pitches[i] = u64::from(tight);
        if i == 0 {
            luma_len = len;
        }
    }
    let key = TwoPlaneMemoKey {
        task_id,
        source: TwoPlaneSource::Surface {
            mapping_id,
            luma_off: planes.luma.base_off,
            chroma_off: planes.chroma.base_off,
        },
        width: planes.width,
        height: planes.height,
        luma_pitch: pitches[0],
        chroma_pitch: pitches[1],
        layout: planes.layout,
        read: planes.read,
    };
    convert_memoized(state, key, planes, native, luma_len)
}

#[cfg(test)]
pub(crate) mod tests;
