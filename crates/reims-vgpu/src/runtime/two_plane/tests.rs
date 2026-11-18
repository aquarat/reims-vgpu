use super::*;
use crate::model::{DeviceId, PAGE_SHIFT_ARM64E};
use crate::protocol::endian::{st16, st32, st64};
use crate::runtime::decode::resource::tests::dual_plane_body;
use crate::runtime::decode::resource::{
    decode_dual_plane_texture_descriptor, Descriptor, LINEAR_DESC_HANDLE, LINEAR_DESC_SIZE,
    TEXTURE_DESC_BYTES_PER_ELEMENT, TEXTURE_DESC_DATA_OFFSET, TEXTURE_DESC_MIP_LEVEL_RECORD_LEN,
    TEXTURE_DIM_BASE, TEXTURE_DIM_HEADER_LEN,
};
use crate::runtime::host::FakeHost;

/// Where the fixture puts the chroma plane inside the allocation.
pub(crate) const CHROMA_OFFSET: u64 = 0x800;
/// The fixture's allocation: one page at guest page 1.
const HANDLE: u32 = 1;
const ALLOCATION: u64 = 0x1000;

/// A single-level dual-plane body over one small allocation, with each
/// plane's bytes per element stated and the chroma plane at
/// [`CHROMA_OFFSET`], built from the decoder's own fixture rather than a
/// second spelling of the layout.
pub(crate) fn dual_plane_fixture_body(pixel_format: u16, extent: (u32, u32)) -> Vec<u8> {
    let mut b = dual_plane_body([1, 1], extent, pixel_format);
    st64(&mut b[LINEAR_DESC_SIZE..], ALLOCATION);
    st32(&mut b[LINEAR_DESC_HANDLE..], HANDLE);
    let block = TEXTURE_DIM_HEADER_LEN + TEXTURE_DESC_MIP_LEVEL_RECORD_LEN;
    let bpe_at = TEXTURE_DESC_BYTES_PER_ELEMENT - TEXTURE_DIM_BASE;
    let ten_bit = ycbcr::two_plane_format(pixel_format).is_some_and(|f| f.bits.bits() > 8);
    let unit = if ten_bit { 2 } else { 1 };
    b[TEXTURE_DIM_BASE + bpe_at] = unit;
    b[TEXTURE_DIM_BASE + block + bpe_at] = 2 * unit;
    // Level 0's offset is the block header's data offset, not its record's.
    let data_offset_at = TEXTURE_DESC_DATA_OFFSET - TEXTURE_DIM_BASE;
    st64(
        &mut b[TEXTURE_DIM_BASE + block + data_offset_at..],
        CHROMA_OFFSET,
    );
    b
}

fn dual(pixel_format: u16, extent: (u32, u32)) -> DualPlaneTextureDescriptor {
    decode_dual_plane_texture_descriptor(&dual_plane_fixture_body(pixel_format, extent))
        .expect("the fixture decodes")
}

#[test]
fn a_four_two_zero_codes_texture_places_both_planes_in_its_allocation() {
    let planes = guest_planes(&dual(540, (8, 4)), PAGE_SHIFT_ARM64E).expect("placed");
    let base = u64::from(HANDLE) << PAGE_SHIFT_ARM64E;
    assert_eq!((planes.width, planes.height), (8, 4));
    assert!(matches!(
        planes.conversion,
        TwoPlaneConversion::YCbCr {
            read: crate::protocol::ycbcr::TexelRead::Codes,
            ..
        }
    ));
    assert_eq!(
        planes.planes[0],
        GuestPlane {
            gva: base,
            pitch: 8,
            span: 8 * 4
        }
    );
    assert_eq!(
        planes.planes[1],
        GuestPlane {
            gva: base + CHROMA_OFFSET,
            pitch: 8,
            span: 8 * 2
        },
        "4x2 pairs of two bytes"
    );
}

/// The fixture's chroma plane is half size, which is 4:2:0's; a 4:4:4 format
/// over the same descriptor is a descriptor whose planes are not the format's.
#[test]
fn a_chroma_plane_of_the_wrong_extent_refuses_rather_than_being_resampled() {
    assert_eq!(
        guest_planes(&dual(542, (8, 4)), PAGE_SHIFT_ARM64E),
        Err(TwoPlaneBindRefusal::PlaneGeometry {
            plane: "plane1",
            width: 4,
            height: 2,
            bytes_per_element: 2,
            want_width: 8,
            want_height: 4,
            want_bytes_per_element: 2,
        })
    );
}

/// A converted format over guest memory has no surface to name its range, and
/// nothing names its matrix: a typed refusal, not BT.601.
#[test]
fn a_converted_format_without_its_terms_refuses_by_name() {
    let refused = guest_planes(&dual(500, (8, 4)), PAGE_SHIFT_ARM64E).unwrap_err();
    assert_eq!(
        refused,
        TwoPlaneBindRefusal::Layout(TwoPlaneRefusal::RangeUnknown { pixel_format: 500 })
    );
    assert_eq!(
        crate::observe::Decline::slug(&refused),
        "two_plane_range_unknown"
    );
    assert_eq!(
        guest_planes(&dual(0x50, (8, 4)), PAGE_SHIFT_ARM64E),
        Err(TwoPlaneBindRefusal::NotTwoPlane { pixel_format: 0x50 })
    );
}

/// A task whose object list holds `texture_ref` as a type-12 object with
/// descriptor `body` over an allocation of `allocation` bytes at guest page
/// [`HANDLE`], and `writes` (offset into the allocation, bytes) in it, in guest
/// memory the fake host backs.
pub(crate) fn guest_type12_texture(
    body: &[u8],
    allocation: u64,
    writes: &[(u64, &[u8])],
) -> (DeviceState, FakeHost, u32, u32) {
    use crate::protocol::gva::{DIRECTORY_DEPTH, DIRECTORY_ROOT_PFN};
    use crate::runtime::decode::resource::{list_object_entry_offset, OBJECT_LIST_ENTRY_LEN};
    use crate::runtime::gva_mem::write_task_gva_arm64e;

    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    let (dir_pfn, root_pfn) = (2u32, 3u32);
    let dir_gpa = u64::from(dir_pfn) << PAGE_SHIFT_ARM64E;
    let root_gpa = u64::from(root_pfn) << PAGE_SHIFT_ARM64E;
    host.map_range(dir_gpa, 0x20, 0);
    host.map_range(root_gpa, 0x4000, 0);
    let mut d = [0u8; 8];
    st32(&mut d[DIRECTORY_ROOT_PFN as usize..], root_pfn);
    st32(&mut d[DIRECTORY_DEPTH as usize..], 1);
    assert!(host.write_gpa(dir_gpa, &d).is_ok());
    // Page 0 holds the object list and the descriptor; the allocation follows.
    let pages = 1 + u32::from(HANDLE) + (allocation >> PAGE_SHIFT_ARM64E) as u32 + 1;
    for i in 0..pages {
        let pfn = 4 + i;
        host.map_range(u64::from(pfn) << PAGE_SHIFT_ARM64E, 0x4000, 0);
        let mut pte = [0u8; 4];
        st32(&mut pte, pfn);
        assert!(host.write_gpa(root_gpa + u64::from(i) * 4, &pte).is_ok());
    }
    let task_id = 1;
    state.define_task(task_id, 0x1000, dir_pfn);
    assert!(state.set_object_list(task_id, 0, 32));

    let texture_ref = 6u32;
    let desc_gva = 0x200u64;
    write_task_gva_arm64e(&mut host, &state.tasks[task_id], desc_gva, body);
    let off = list_object_entry_offset(texture_ref, 32).unwrap();
    let mut entry = [0u8; OBJECT_LIST_ENTRY_LEN];
    st32(
        &mut entry[0..],
        u32::from(OBJECT_TYPE_DUAL_PLANE_TEXTURE) | ((body.len() as u32) << 8),
    );
    entry[4..12].copy_from_slice(&desc_gva.to_le_bytes());
    write_task_gva_arm64e(&mut host, &state.tasks[task_id], off, &entry);
    let base = u64::from(HANDLE) << PAGE_SHIFT_ARM64E;
    for &(at, bytes) in writes {
        write_task_gva_arm64e(&mut host, &state.tasks[task_id], base + at, bytes);
    }
    (state, host, task_id, texture_ref)
}

/// [`guest_type12_texture`] over the [`dual_plane_fixture_body`] of
/// `pixel_format`, with plane 0's and plane 1's bytes.
pub(crate) fn guest_two_plane_texture(
    pixel_format: u16,
    extent: (u32, u32),
    plane0: &[u8],
    plane1: &[u8],
) -> (DeviceState, FakeHost, u32, u32) {
    guest_type12_texture(
        &dual_plane_fixture_body(pixel_format, extent),
        ALLOCATION,
        &[(0, plane0), (CHROMA_OFFSET, plane1)],
    )
}

/// Decode to bind: the object list's type-12 entry resolves, decodes as two
/// planes, places them, reads them out of guest memory and converts them to
/// what the protocol converter makes of the same bytes. An unchanged pair is
/// the memo's; a guest write to either plane is seen.
#[test]
fn a_guest_two_plane_texture_decodes_places_reads_and_converts() {
    let (w, h) = (4u32, 2u32);
    let luma: Vec<u8> = (0..w * h).map(|i| (16 + 29 * i) as u8).collect();
    let chroma: Vec<u8> = vec![40, 220, 90, 30];
    let (mut state, mut host, task_id, texture_ref) =
        guest_two_plane_texture(540, (w, h), &luma, &chroma);

    let resource = crate::runtime::objects::resolve_resource(&state, &host, task_id, texture_ref)
        .expect("a type-12 object is a resource");
    let Ok(Descriptor::DualPlaneTexture(dual)) = resource.decoded() else {
        panic!("tag 12 decodes as two planes");
    };
    let planes = guest_planes(dual, state.page_shift).expect("placed");
    let site = crate::runtime::render_writeback::SettleSite::LinearMemoRead;
    let first =
        load_guest(&mut state, &mut host, task_id, texture_ref, &planes, site).expect("converted");
    let want = planes
        .conversion
        .convert(
            w,
            h,
            PlaneRows {
                bytes: &luma,
                pitch: w as usize,
            },
            PlaneRows {
                bytes: &chroma,
                pitch: 4,
            },
        )
        .unwrap();
    assert_eq!(*first.bytes, want);
    assert_eq!(
        first.conversion.output_pixel_format(),
        crate::protocol::pixel_format::MTL_FORMAT_RGBA8_UNORM
    );
    // Texel (0, 0) reads pair 0 alone: its nearer taps are pair 0 and its
    // farther ones clamp onto it. Red Cr, green Y, blue Cb.
    assert_eq!(first.bytes[0..4], [220, luma[0], 40, 0xff]);

    let again = load_guest(&mut state, &mut host, task_id, texture_ref, &planes, site).unwrap();
    assert!(
        Arc::ptr_eq(&first.bytes, &again.bytes),
        "unchanged planes reuse the conversion"
    );
    assert_eq!(first.generation, again.generation);

    let base = u64::from(HANDLE) << PAGE_SHIFT_ARM64E;
    crate::runtime::gva_mem::write_task_gva_arm64e(
        &mut host,
        &state.tasks[task_id],
        base + CHROMA_OFFSET,
        &[41],
    );
    let changed = load_guest(&mut state, &mut host, task_id, texture_ref, &planes, site).unwrap();
    assert_ne!(
        changed.generation, first.generation,
        "a chroma write is seen"
    );
    assert_eq!(changed.bytes[0..4], [220, luma[0], 41, 0xff]);
}

#[test]
fn a_surface_texture_takes_its_range_from_the_ostype_and_still_names_the_missing_matrix() {
    use crate::protocol::iosurface_pages::{
        DEVICE_DESC_LEN, DEVICE_DESC_PIXEL_FORMAT, DEVICE_DESC_PLANES, DEVICE_DESC_PLANE_COUNT,
        DEVICE_PLANE_BPE, DEVICE_PLANE_BPR, DEVICE_PLANE_DESC_LEN, DEVICE_PLANE_DIMS,
        DEVICE_PLANE_OFFSET, DEVICE_PLANE_SIZE,
    };
    let (w, h) = (8u32, 4u32);
    let pack_dims = |w: u64, h: u64| ((w & 0xff_ffff) << 8) | ((h & 0xff_ffff) << 40);
    let mut desc = vec![0u8; DEVICE_DESC_LEN];
    assert!(DEVICE_DESC_PLANES + 2 * DEVICE_PLANE_DESC_LEN <= DEVICE_DESC_LEN);
    st32(
        &mut desc[DEVICE_DESC_PIXEL_FORMAT..],
        u32::from_be_bytes(*b"420f"),
    );
    desc[DEVICE_DESC_PLANE_COUNT] = 2;
    for (i, (pw, ph, bpe, off)) in [(w, h, 1u16, 0u32), (w / 2, h / 2, 2, 0x800)]
        .into_iter()
        .enumerate()
    {
        let p = DEVICE_DESC_PLANES + i * DEVICE_PLANE_DESC_LEN;
        st32(&mut desc[p + DEVICE_PLANE_OFFSET..], off);
        st32(&mut desc[p + DEVICE_PLANE_SIZE..], 64 * ph);
        st64(
            &mut desc[p + DEVICE_PLANE_DIMS..],
            pack_dims(u64::from(pw), u64::from(ph)),
        );
        st32(&mut desc[p + DEVICE_PLANE_BPR..], 64);
        st16(&mut desc[p + DEVICE_PLANE_BPE..], bpe);
    }
    let mapping = MappingEntry {
        device_desc: desc,
        ..Default::default()
    };
    let codes = surface_planes(&mapping, 540, w, h).expect("a codes format needs no terms");
    assert_eq!(
        (
            codes.planes[0].base_off,
            codes.planes[0].bpr,
            codes.planes[0].bpp
        ),
        (0, 64, 1)
    );
    assert_eq!(
        (
            codes.planes[1].base_off,
            codes.planes[1].bpr,
            codes.planes[1].bpp
        ),
        (0x800, 64, 2)
    );
    // '420f' names full range, so the refusal is the matrix's, not the range's.
    assert_eq!(
        surface_planes(&mapping, 500, w, h),
        Err(TwoPlaneBindRefusal::Layout(
            TwoPlaneRefusal::MatrixUnknown { pixel_format: 500 }
        ))
    );
    // 4:4:4 over this 4:2:0 surface: its chroma record is not the format's.
    assert_eq!(
        surface_planes(&mapping, 542, w, h),
        Err(TwoPlaneBindRefusal::SurfacePlanes {
            plane_count: 2,
            plane: "plane1"
        })
    );
}

/// One type-12 descriptor exactly as a macOS 26.4 guest sent it, from the
/// `dual_plane_texture_descriptor` census of a run on an M1 Max test host: both dimension
/// headers and the trailer verbatim, and each plane's level-0 record rebuilt
/// from the census's decoded fields (`plane0=320x128:bpr=1280:...`). The
/// object header's handle is the fixture's, so the allocation lands in the
/// fake host's pages.
pub(crate) struct GuestDescriptor {
    pub allocation: u64,
    /// `(width, height, bpr, size)` of each plane's level 0.
    pub records: [(u32, u32, u64, u64); 2],
    pub headers: [&'static str; 2],
    pub trailer: &'static str,
}

/// The login window's first `RGB10A8_2P_XR10` texture (M1 Max test host, 2026-11-17).
pub(crate) const CENSUS_XR10_320: GuestDescriptor = GuestDescriptor {
    allocation: 0x50000,
    records: [(320, 128, 1280, 163840), (320, 128, 1280, 163840)],
    headers: [
        "010001000000000000000000008002000000000001010104",
        "010001000080020000000000008002000000000001010104",
    ],
    trailer:
        "010000002c000000090000004205260240010000800000000100000001000100010020000000000000000000",
};

/// The login window's second `RGB10A8_2P_XR10` texture.
pub(crate) const CENSUS_XR10_352: GuestDescriptor = GuestDescriptor {
    allocation: 0x58000,
    records: [(352, 128, 1408, 180224), (352, 128, 1408, 180224)],
    headers: [
        "01000100000000000000000000c002000000000001010104",
        "0100010000c002000000000000c002000000000001010104",
    ],
    trailer:
        "010000002c000000090000004205260260010000800000000100000001000100010020000000000000000000",
};

/// A `Depth32Float_Stencil8` texture the same guest sent as type 12: a 4-byte
/// depth plane and a 1-byte stencil plane.
pub(crate) const CENSUS_D32S8: GuestDescriptor = GuestDescriptor {
    allocation: 0x764000,
    records: [(492, 492, 7872, 3873024), (492, 492, 7872, 3873024)],
    headers: [
        "01000100000000000000000000193b000000000001010104",
        "0100010000193b000000000000193b000000000001010101",
    ],
    trailer:
        "010000002c000000bf02000044140401ec010000ec0100000100000001000400010020000000000000000000",
};

fn unhex(s: &str) -> Vec<u8> {
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap())
        .collect()
}

impl GuestDescriptor {
    pub(crate) fn body(&self) -> Vec<u8> {
        use crate::runtime::decode::resource::{
            TEXTURE_DESC_LEVEL_RECORDS, TEXTURE_LEVEL_DEPTH, TEXTURE_LEVEL_HEIGHT,
            TEXTURE_LEVEL_ROW_STRIDE, TEXTURE_LEVEL_SIZE, TEXTURE_LEVEL_WIDTH,
        };
        let mut b = vec![0u8; TEXTURE_DIM_BASE];
        st64(&mut b[LINEAR_DESC_SIZE..], self.allocation);
        st32(&mut b[LINEAR_DESC_HANDLE..], HANDLE);
        for (header, (w, h, bpr, size)) in self.headers.iter().zip(self.records) {
            b.extend(unhex(header));
            let rec = b.len();
            b.resize(rec + TEXTURE_DESC_MIP_LEVEL_RECORD_LEN, 0);
            st64(&mut b[rec + TEXTURE_LEVEL_SIZE..], size);
            st64(&mut b[rec + TEXTURE_LEVEL_ROW_STRIDE..], bpr);
            st32(&mut b[rec + TEXTURE_LEVEL_WIDTH..], w);
            st32(&mut b[rec + TEXTURE_LEVEL_HEIGHT..], h);
            st32(&mut b[rec + TEXTURE_LEVEL_DEPTH..], 1);
        }
        b.extend(unhex(self.trailer));
        assert_eq!(b.len(), 176, "the census's len=176");
        assert_eq!(
            TEXTURE_DIM_BASE + TEXTURE_DIM_HEADER_LEN,
            TEXTURE_DESC_LEVEL_RECORDS - TEXTURE_DESC_MIP_LEVEL_RECORD_LEN
        );
        b
    }

    /// The same descriptor with its plane 1 stated as one byte a texel, at the
    /// tight pitch: the alpha plane `RGB10A8_2P_XR10` holds.
    pub(crate) fn with_byte_alpha(&self) -> Vec<u8> {
        let mut b = self.body();
        let block = TEXTURE_DIM_HEADER_LEN + TEXTURE_DESC_MIP_LEVEL_RECORD_LEN;
        let bpe_at = TEXTURE_DESC_BYTES_PER_ELEMENT - TEXTURE_DIM_BASE;
        b[TEXTURE_DIM_BASE + block + bpe_at] = 1;
        let rec1 = TEXTURE_DIM_BASE + block + TEXTURE_DIM_HEADER_LEN;
        let (w, h, _, _) = self.records[1];
        st64(
            &mut b[rec1 + crate::runtime::decode::resource::TEXTURE_LEVEL_ROW_STRIDE..],
            u64::from(w),
        );
        st64(
            &mut b[rec1 + crate::runtime::decode::resource::TEXTURE_LEVEL_SIZE..],
            u64::from(w) * u64::from(h),
        );
        b
    }

    fn plane1_offset(&self) -> u64 {
        self.records[0].3
    }
}

#[test]
fn the_census_descriptors_decode_as_the_census_printed_them() {
    for (g, pf) in [
        (&CENSUS_XR10_320, 550u16),
        (&CENSUS_XR10_352, 550),
        (&CENSUS_D32S8, 260),
    ] {
        let d = decode_dual_plane_texture_descriptor(&g.body()).expect("decodes");
        assert_eq!(d.pixel_format(), pf);
        for (plane, (w, h, bpr, _)) in d.planes.iter().zip(g.records) {
            let l = plane.level(0).unwrap();
            assert_eq!((l.width, l.height, l.row_stride), (w, h, bpr));
        }
        assert_eq!(d.planes[1].level(0).unwrap().offset, g.plane1_offset());
    }
}

/// The guest states `RGB10A8_2P_XR10`'s alpha plane in 4-byte elements, where
/// the format holds a byte a texel: which byte is not recovered, so the bind
/// refuses by name, and the census says where the plane's bytes are.
#[test]
fn the_census_xr10_texture_refuses_its_unrecovered_alpha_plane_and_writes_a_census() {
    let g = &CENSUS_XR10_320;
    let (w, h) = (320usize, 128usize);
    let color = vec![0x11u8; 4 * w * h];
    // Lane 0 of each stated 4-byte element non-zero, the rest zero.
    let mut alpha = vec![0u8; 4 * w * h];
    for e in alpha.chunks_exact_mut(4) {
        e[0] = 0xff;
    }
    let (mut state, mut host, task_id, texture_ref) = guest_type12_texture(
        &g.body(),
        g.allocation,
        &[(0, &color), (g.plane1_offset(), &alpha)],
    );
    let resource = crate::runtime::objects::resolve_resource(&state, &host, task_id, texture_ref)
        .expect("a resource");
    let Ok(Descriptor::DualPlaneTexture(dual)) = resource.decoded() else {
        panic!("decodes");
    };
    let cap = crate::observe::FailCapture::start();
    let site = crate::runtime::render_writeback::SettleSite::LinearMemoRead;
    assert_eq!(
        guest_texels(&mut state, &mut host, task_id, texture_ref, dual, site).unwrap_err(),
        TwoPlaneBindRefusal::AlphaLayout {
            bytes_per_element: 4,
            pitch: 1280
        }
    );
    let lines = cap.lines();
    let census = lines
        .iter()
        .find(|l| l.contains("two_plane_alpha_census"))
        .expect("the census reaches the always-on channel");
    assert!(
        census.contains("pf=550 320x128 bpe=4 pitch=1280")
            && census.contains(&format!("color_nonzero={}", w * h))
            && census.contains(&format!("lanes=[{}, 0, 0, 0]", w * h)),
        "{census}"
    );
}

/// The same texture with its alpha plane stated as a byte a texel binds: the
/// colour word is `BGR10_XR`'s, the alpha byte `b / 255`, written as
/// `RGBA16Float`.
#[test]
fn an_xr10_texture_with_a_byte_alpha_plane_converts_to_half_floats() {
    use crate::protocol::two_plane::{alpha_byte_half, xr10_code_half};
    let g = &CENSUS_XR10_320;
    let (w, h) = (320usize, 128usize);
    let word = |r: u32, gr: u32, b: u32| ((r << 20) | (gr << 10) | b).to_le_bytes();
    let mut color = vec![0u8; 4 * w * h];
    color[0..4].copy_from_slice(&word(894, 384, 0));
    color[4..8].copy_from_slice(&word(1023, 639, 129));
    let mut alpha = vec![0u8; w * h];
    alpha[0] = 255;
    alpha[1] = 128;
    let (mut state, mut host, task_id, texture_ref) = guest_type12_texture(
        &g.with_byte_alpha(),
        g.allocation,
        &[(0, &color), (g.plane1_offset(), &alpha)],
    );
    let resource = crate::runtime::objects::resolve_resource(&state, &host, task_id, texture_ref)
        .expect("a resource");
    let Ok(Descriptor::DualPlaneTexture(dual)) = resource.decoded() else {
        panic!("decodes");
    };
    let site = crate::runtime::render_writeback::SettleSite::LinearMemoRead;
    let texels = guest_texels(&mut state, &mut host, task_id, texture_ref, dual, site)
        .expect("a byte alpha plane binds");
    assert_eq!((texels.width, texels.height), (320, 128));
    assert_eq!(texels.conversion, TwoPlaneConversion::ExtendedRange);
    assert_eq!(
        texels.conversion.output_pixel_format(),
        crate::protocol::pixel_format::MTL_FORMAT_RGBA16_FLOAT
    );
    let half = |i: usize| u16::from_le_bytes([texels.bytes[2 * i], texels.bytes[2 * i + 1]]);
    assert_eq!(
        (0..8).map(half).collect::<Vec<_>>(),
        [
            0x3c00,
            0,
            xr10_code_half(0),
            0x3c00,
            xr10_code_half(1023),
            0x3800,
            0xb800,
            alpha_byte_half(128)
        ]
    );
}

/// `Depth32Float_Stencil8` arrives as type 12 too and is no colour format: it
/// refuses by name, before any plane is placed or read, and writes no census.
#[test]
fn the_census_depth_stencil_texture_is_not_converted() {
    let d = decode_dual_plane_texture_descriptor(&CENSUS_D32S8.body()).unwrap();
    assert_eq!(
        guest_planes(&d, PAGE_SHIFT_ARM64E),
        Err(TwoPlaneBindRefusal::NotTwoPlane { pixel_format: 260 })
    );
    assert_eq!(
        crate::observe::Decline::slug(&TwoPlaneBindRefusal::NotTwoPlane { pixel_format: 260 }),
        "two_plane_not_color"
    );
}
