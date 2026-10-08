//! The host driver's answer to the guest's `heapTextureSizeAndAlign` contract.
//!
//! # Why this is neither a `Backend` method nor part of `backend::metal`
//!
//! The guest is asking what Apple's driver requires to place one texture in a
//! heap. That is a term of the guest ABI, and the answer has to be the same on
//! every rail this host can run: the arm64 macOS pathway executes one guest
//! stream through Metal and through MoltenVK without a rebuild, and a heap
//! layout that differed between them would be this device answering a guest
//! contract with a host implementation detail.
//!
//! So the running rail does not enter into it, and neither does the *compiled*
//! one. The gate below is `target_os = "macos"` alone — the gate this code has
//! always carried — because the question is whether a driver exists to be
//! asked, not which rail this build ships. Moving it under `backend::metal`,
//! which is gated on `feature = "backend-metal"`, would silently make the
//! macOS Vulkan pathway start refusing a query it answers today.
//!
//! Everything above this — the request decode, the descriptor, the reply
//! encoding, the refusal vocabulary — is wire and stays in
//! [`crate::runtime::heap_query`]. Only the driver call is here.

use crate::runtime::heap_query::{QueryError, SizeAndAlign, TextureDescriptor};

#[cfg(target_os = "macos")]
pub fn heap_texture_size_and_align(desc: &TextureDescriptor) -> Result<SizeAndAlign, QueryError> {
    use crate::protocol::texture_shape::TextureKind;
    use metal::{
        MTLResourceOptions, MTLTextureType, MTLTextureUsage, TextureDescriptor as MtlDescriptor,
    };
    use objc::runtime::{NO, YES};
    use objc::{msg_send, sel, sel_impl};

    // Which types exist and which ordinal names each one is the guest ABI's,
    // and `protocol::texture_shape` is where this device says so. Read as a
    // bare ordinal match here, it was the same nine-row table written twice —
    // and the copy a guest actually reaches for a heap placement is this one,
    // so a tenth type added to the owner would have silently kept refusing
    // here while every other reader accepted it.
    let kind =
        crate::protocol::texture_shape::TextureKind::from_ordinal(u32::from(desc.texture_type))
            .ok_or(QueryError::UnknownTextureType)?;
    // The ordinal is parsed once above; this is the driver's spelling of the
    // type it named, and nothing but a rename.
    let texture_type = match kind {
        TextureKind::D1 => MTLTextureType::D1,
        TextureKind::D1Array => MTLTextureType::D1Array,
        TextureKind::D2 => MTLTextureType::D2,
        TextureKind::D2Array => MTLTextureType::D2Array,
        TextureKind::D2Multisample => MTLTextureType::D2Multisample,
        TextureKind::Cube => MTLTextureType::Cube,
        TextureKind::CubeArray => MTLTextureType::CubeArray,
        TextureKind::D3 => MTLTextureType::D3,
        TextureKind::D2MultisampleArray => MTLTextureType::D2MultisampleArray,
    };
    let pixel_format =
        pixel_format_from_wire(desc.pixel_format).ok_or(QueryError::UnknownPixelFormat)?;
    let usage = MTLTextureUsage::from_bits(desc.usage as u64).ok_or(QueryError::UnknownUsage)?;
    let resource_options = MTLResourceOptions::from_bits(desc.resource_options as u64)
        .ok_or(QueryError::UnknownResourceOptions)?;
    if desc.protection_options != 0 {
        return Err(QueryError::UnsupportedProtectionOptions);
    }
    let device = metal::Device::system_default().ok_or(QueryError::NoMetalDevice)?;
    let mtl = MtlDescriptor::new();
    mtl.set_texture_type(texture_type);
    mtl.set_pixel_format(pixel_format);
    mtl.set_width(desc.width as u64);
    mtl.set_height(desc.height as u64);
    mtl.set_depth(desc.depth as u64);
    mtl.set_mipmap_level_count(desc.mipmap_level_count as u64);
    mtl.set_sample_count(desc.sample_count as u64);
    mtl.set_array_length(desc.array_length as u64);
    mtl.set_resource_options(resource_options);
    mtl.set_usage(usage);
    unsafe {
        let framebuffer_only = if desc.framebuffer_only { YES } else { NO };
        let is_drawable = if desc.is_drawable { YES } else { NO };
        let allow_gpu_optimized = if desc.allow_gpu_optimized_contents {
            YES
        } else {
            NO
        };
        let _: () = msg_send![&*mtl, setFramebufferOnly: framebuffer_only];
        let _: () = msg_send![&*mtl, setIsDrawable: is_drawable];
        let _: () = msg_send![
            &*mtl,
            setAllowGPUOptimizedContents: allow_gpu_optimized
        ];
    }
    let requirement = device.heap_texture_size_and_align(&mtl);
    if requirement.size == 0 || requirement.align == 0 {
        return Err(QueryError::ZeroRequirement);
    }
    Ok(SizeAndAlign {
        size: requirement.size,
        align: requirement.align,
    })
}

/// Heap placement where no Metal driver exists to ask.
///
/// A heap texture is executed as a host residency identity of its own (see
/// `runtime::compute_exec`): the device never reads or writes its bytes through
/// the heap's guest pages. For a **private** heap the guest cannot address
/// those bytes either, so the only thing the answer decides is how much of the
/// heap the guest reserves for the texture, and any answer that covers the
/// texture's own extent keeps two placements apart. This answers with the
/// texture's linear extent — every mip level at a 256-byte row pitch, every
/// slice, face and sample — rounded up to, and aligned at, one guest page.
///
/// Shared and managed heaps expose the bytes to the guest CPU, where Apple's
/// real layout is the contract; those are still refused, by name.
#[cfg(not(target_os = "macos"))]
pub fn heap_texture_size_and_align(desc: &TextureDescriptor) -> Result<SizeAndAlign, QueryError> {
    use crate::protocol::storage_mode::{from_resource_options, StorageMode};
    use crate::protocol::texture_shape::TextureKind;
    /// Row pitch the extent is computed at; at least what any GPU rounds to.
    const ROW_ALIGN: u64 = 256;
    /// One arm64e guest page: the placement granule the extent is rounded to.
    const PLACEMENT_ALIGN: u64 = 1 << 14;
    match from_resource_options(u32::from(desc.resource_options)) {
        Ok(StorageMode::Private) => {}
        _ => return Err(QueryError::UnsupportedStorageMode),
    }
    if desc.protection_options != 0 {
        return Err(QueryError::UnsupportedProtectionOptions);
    }
    let kind = TextureKind::from_ordinal(u32::from(desc.texture_type))
        .ok_or(QueryError::UnknownTextureType)?;
    let faces: u64 = match kind {
        TextureKind::Cube | TextureKind::CubeArray => 6,
        _ => 1,
    };
    let layers = u64::from(desc.array_length.max(1)) * faces;
    let samples = u64::from(desc.sample_count.max(1));
    let levels = u32::from(desc.mipmap_level_count.max(1));
    let mut total: u64 = 0;
    for level in 0..levels {
        let w = (desc.width >> level).max(1);
        let h = (desc.height >> level).max(1);
        let d = u64::from((desc.depth >> level).max(1));
        let row = crate::protocol::pixel_format::tight_row_bytes(w, desc.pixel_format)
            .ok_or(QueryError::UnknownPixelFormat)?;
        let rows = crate::protocol::pixel_format::tight_row_count(h, desc.pixel_format)
            .ok_or(QueryError::UnknownPixelFormat)?;
        let pitch = u64::from(row).div_ceil(ROW_ALIGN) * ROW_ALIGN;
        total = pitch
            .checked_mul(u64::from(rows))
            .and_then(|v| v.checked_mul(d))
            .and_then(|v| total.checked_add(v))
            .ok_or(QueryError::ZeroRequirement)?;
    }
    let size = total
        .checked_mul(layers)
        .and_then(|v| v.checked_mul(samples))
        .map(|v| v.div_ceil(PLACEMENT_ALIGN) * PLACEMENT_ALIGN)
        .filter(|&v| v != 0)
        .ok_or(QueryError::ZeroRequirement)?;
    Ok(SizeAndAlign {
        size,
        align: PLACEMENT_ALIGN,
    })
}

#[cfg(all(test, not(target_os = "macos")))]
mod tests {
    use super::*;

    fn private(fmt: u16, w: u32, h: u32) -> TextureDescriptor {
        TextureDescriptor {
            texture_type: 2,
            framebuffer_only: false,
            is_drawable: false,
            allow_gpu_optimized_contents: true,
            usage: 5,
            pixel_format: fmt,
            width: w,
            height: h,
            depth: 1,
            mipmap_level_count: 1,
            sample_count: 1,
            array_length: 1,
            resource_options: 0x20,
            protection_options: 0,
            swizzle: None,
        }
    }

    /// The live query from a macOS 26 guest: 278x117 RGBA16Float in a private
    /// heap. 278 * 8 = 2224 bytes pitched to 2304, times 117 rows, is 269 568
    /// bytes, which is seventeen 16 KiB pages.
    #[test]
    fn a_private_heap_texture_reserves_its_linear_extent_in_whole_pages() {
        let got = heap_texture_size_and_align(&private(0x73, 278, 117)).unwrap();
        assert_eq!(got.align, 16384);
        assert_eq!(got.size, 17 * 16384);
    }

    #[test]
    fn a_shared_heap_texture_is_still_refused() {
        let mut desc = private(0x73, 64, 64);
        desc.resource_options = 0x00;
        assert_eq!(
            heap_texture_size_and_align(&desc),
            Err(QueryError::UnsupportedStorageMode)
        );
    }
}

#[cfg(target_os = "macos")]
fn pixel_format_from_wire(raw: u16) -> Option<metal::MTLPixelFormat> {
    use metal::MTLPixelFormat as F;
    Some(match raw as u64 {
        x if x == F::Invalid as u64 => F::Invalid,
        x if x == F::A8Unorm as u64 => F::A8Unorm,
        x if x == F::R8Unorm as u64 => F::R8Unorm,
        x if x == F::R8Unorm_sRGB as u64 => F::R8Unorm_sRGB,
        x if x == F::R8Snorm as u64 => F::R8Snorm,
        x if x == F::R8Uint as u64 => F::R8Uint,
        x if x == F::R8Sint as u64 => F::R8Sint,
        x if x == F::R16Unorm as u64 => F::R16Unorm,
        x if x == F::R16Snorm as u64 => F::R16Snorm,
        x if x == F::R16Uint as u64 => F::R16Uint,
        x if x == F::R16Sint as u64 => F::R16Sint,
        x if x == F::R16Float as u64 => F::R16Float,
        x if x == F::RG8Unorm as u64 => F::RG8Unorm,
        x if x == F::RG8Unorm_sRGB as u64 => F::RG8Unorm_sRGB,
        x if x == F::RG8Snorm as u64 => F::RG8Snorm,
        x if x == F::RG8Uint as u64 => F::RG8Uint,
        x if x == F::RG8Sint as u64 => F::RG8Sint,
        x if x == F::B5G6R5Unorm as u64 => F::B5G6R5Unorm,
        x if x == F::A1BGR5Unorm as u64 => F::A1BGR5Unorm,
        x if x == F::ABGR4Unorm as u64 => F::ABGR4Unorm,
        x if x == F::BGR5A1Unorm as u64 => F::BGR5A1Unorm,
        x if x == F::R32Uint as u64 => F::R32Uint,
        x if x == F::R32Sint as u64 => F::R32Sint,
        x if x == F::R32Float as u64 => F::R32Float,
        x if x == F::RG16Unorm as u64 => F::RG16Unorm,
        x if x == F::RG16Snorm as u64 => F::RG16Snorm,
        x if x == F::RG16Uint as u64 => F::RG16Uint,
        x if x == F::RG16Sint as u64 => F::RG16Sint,
        x if x == F::RG16Float as u64 => F::RG16Float,
        x if x == F::RGBA8Unorm as u64 => F::RGBA8Unorm,
        x if x == F::RGBA8Unorm_sRGB as u64 => F::RGBA8Unorm_sRGB,
        x if x == F::RGBA8Snorm as u64 => F::RGBA8Snorm,
        x if x == F::RGBA8Uint as u64 => F::RGBA8Uint,
        x if x == F::RGBA8Sint as u64 => F::RGBA8Sint,
        x if x == F::BGRA8Unorm as u64 => F::BGRA8Unorm,
        x if x == F::BGRA8Unorm_sRGB as u64 => F::BGRA8Unorm_sRGB,
        x if x == F::RGB10A2Unorm as u64 => F::RGB10A2Unorm,
        x if x == F::RGB10A2Uint as u64 => F::RGB10A2Uint,
        x if x == F::RG11B10Float as u64 => F::RG11B10Float,
        x if x == F::RGB9E5Float as u64 => F::RGB9E5Float,
        x if x == F::BGR10A2Unorm as u64 => F::BGR10A2Unorm,
        x if x == F::RG32Uint as u64 => F::RG32Uint,
        x if x == F::RG32Sint as u64 => F::RG32Sint,
        x if x == F::RG32Float as u64 => F::RG32Float,
        x if x == F::RGBA16Unorm as u64 => F::RGBA16Unorm,
        x if x == F::RGBA16Snorm as u64 => F::RGBA16Snorm,
        x if x == F::RGBA16Uint as u64 => F::RGBA16Uint,
        x if x == F::RGBA16Sint as u64 => F::RGBA16Sint,
        x if x == F::RGBA16Float as u64 => F::RGBA16Float,
        x if x == F::RGBA32Uint as u64 => F::RGBA32Uint,
        x if x == F::RGBA32Sint as u64 => F::RGBA32Sint,
        x if x == F::RGBA32Float as u64 => F::RGBA32Float,
        x if x == F::Depth16Unorm as u64 => F::Depth16Unorm,
        x if x == F::Depth32Float as u64 => F::Depth32Float,
        x if x == F::Stencil8 as u64 => F::Stencil8,
        x if x == F::Depth24Unorm_Stencil8 as u64 => F::Depth24Unorm_Stencil8,
        x if x == F::Depth32Float_Stencil8 as u64 => F::Depth32Float_Stencil8,
        x if x == F::X32_Stencil8 as u64 => F::X32_Stencil8,
        x if x == F::X24_Stencil8 as u64 => F::X24_Stencil8,
        _ => return None,
    })
}
