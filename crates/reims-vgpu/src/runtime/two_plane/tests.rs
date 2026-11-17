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
    assert_eq!(planes.read, TexelRead::Codes);
    assert_eq!(
        planes.luma,
        GuestPlane {
            gva: base,
            pitch: 8,
            span: 8 * 4
        }
    );
    assert_eq!(
        planes.chroma,
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
            plane: "chroma",
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

/// A task whose object list holds `texture_ref` as a type-12 object of
/// `pixel_format` with these plane bytes, in guest memory the fake host backs.
pub(crate) fn guest_two_plane_texture(
    pixel_format: u16,
    extent: (u32, u32),
    luma: &[u8],
    chroma: &[u8],
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
    for i in 0..4u32 {
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
    let body = dual_plane_fixture_body(pixel_format, extent);
    let desc_gva = 0x200u64;
    write_task_gva_arm64e(&mut host, &state.tasks[task_id], desc_gva, &body);
    let off = list_object_entry_offset(texture_ref, 32).unwrap();
    let mut entry = [0u8; OBJECT_LIST_ENTRY_LEN];
    st32(
        &mut entry[0..],
        u32::from(OBJECT_TYPE_DUAL_PLANE_TEXTURE) | ((body.len() as u32) << 8),
    );
    entry[4..12].copy_from_slice(&desc_gva.to_le_bytes());
    write_task_gva_arm64e(&mut host, &state.tasks[task_id], off, &entry);
    let base = u64::from(HANDLE) << PAGE_SHIFT_ARM64E;
    write_task_gva_arm64e(&mut host, &state.tasks[task_id], base, luma);
    write_task_gva_arm64e(
        &mut host,
        &state.tasks[task_id],
        base + CHROMA_OFFSET,
        chroma,
    );
    (state, host, task_id, texture_ref)
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
    let want = ycbcr::convert(
        planes.layout,
        TexelRead::Codes,
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
        first.layout.output_pixel_format(),
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
        (codes.luma.base_off, codes.luma.bpr, codes.luma.bpp),
        (0, 64, 1)
    );
    assert_eq!(
        (codes.chroma.base_off, codes.chroma.bpr, codes.chroma.bpp),
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
            plane: "chroma"
        })
    );
}
