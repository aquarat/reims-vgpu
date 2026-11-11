//! Whether guest RAM can reach this device as dma-buf imports, the rail for a
//! device without [`super::host_pointer`] (Mesa's Honeykrisp).
//!
//! Same shape as the host-pointer rail and the same consumers: guest RAM is
//! imported in chunks, each one `VkDeviceMemory` with a `VkBuffer` over all of
//! it, and every reference into guest RAM is a bounded slice of one chunk. What
//! differs is where the memory comes from: a udmabuf over the guest-RAM memfd
//! (`crate::runtime::guest_ram::dmabuf`), imported with
//! `VK_EXT_external_memory_dma_buf`. The out-of-process backend can make one
//! only when the VMM handed it `/dev/udmabuf`; the in-process build never has a
//! source and never takes this rail.
//!
//! Two limits shape the chunks. A udmabuf can be no larger than the module's
//! `size_limit_mb` (64 MiB unless the host raised it), so the span cap is at
//! most that; and it pins every page it covers while it lives, so chunks are
//! imported when a draw first binds them and never warmed ahead of time.

use ash::vk;

use super::host_pointer::{GUEST_IMPORT_USAGE, IMPORT_SPAN_CEILING};

/// Whether this device can import guest RAM as dma-bufs, and when it cannot,
/// which check said so.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum DmaBufImport {
    /// Both extensions are advertised, the device reports `DMA_BUF_EXT`
    /// importable for [`GUEST_IMPORT_USAGE`], and the VMM gave this process a
    /// udmabuf source.
    Supported,
    /// No [`query`] has been run.
    #[default]
    Unqueried,
    /// The host-pointer rail is available, and it is preferred: it pins
    /// nothing this rail would and has no size limit per import.
    HostPointerPreferred,
    /// `VK_EXT_external_memory_dma_buf` or `VK_KHR_external_memory_fd` is not
    /// advertised.
    NoDmaBufExtension,
    /// The extensions are advertised and the device still declines `DMA_BUF_EXT`
    /// as an importable handle type for [`GUEST_IMPORT_USAGE`].
    NotImportable,
    /// Nothing in this process can make a dma-buf of guest RAM: the VMM did not
    /// offer `/dev/udmabuf` (an older VMM, a balloon, the in-process build).
    NoSource,
    /// The host page size is not a power of two, or the source's size limit is
    /// below it.
    AlignmentUnsatisfiable,
    /// `REIMS_VGPU_UDMABUF` or `REIMS_VGPU_GUEST_IMPORT` was switched off.
    DisabledByEnv,
}

impl DmaBufImport {
    pub fn is_available(self) -> bool {
        matches!(self, Self::Supported)
    }

    pub fn slug(self) -> &'static str {
        match self {
            Self::Supported => "supported",
            Self::Unqueried => "unqueried",
            Self::HostPointerPreferred => "host_pointer_preferred",
            Self::NoDmaBufExtension => "no_dma_buf_extension",
            Self::NotImportable => "not_importable",
            Self::NoSource => "no_udmabuf_source",
            Self::AlignmentUnsatisfiable => "alignment_unsatisfiable",
            Self::DisabledByEnv => "disabled_by_env",
        }
    }

    /// Device extension names this rung requires.
    pub fn required_extensions(self) -> Vec<*const std::os::raw::c_char> {
        if self.is_available() {
            vec![
                vk::KHR_EXTERNAL_MEMORY_FD_NAME.as_ptr(),
                vk::EXT_EXTERNAL_MEMORY_DMA_BUF_NAME.as_ptr(),
            ]
        } else {
            Vec::new()
        }
    }
}

/// The dma-buf rail's answer and the limits it was answered with.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DmaBufImportCaps {
    pub rung: DmaBufImport,
    /// The host page size: what the kernel pins in and what every chunk's base
    /// and length are multiples of.
    pub granularity: u64,
    /// The roomiest heap an import can be charged to, as for the host-pointer
    /// rail.
    pub heap_budget: u64,
    /// The largest single chunk: the host-pointer rail's ceiling, the device's
    /// own allocation and buffer limits, and the udmabuf size limit.
    pub span_max: u64,
}

impl DmaBufImportCaps {
    fn refused(rung: DmaBufImport) -> Self {
        Self {
            rung,
            ..Self::default()
        }
    }

    pub fn is_available(self) -> bool {
        self.rung.is_available()
    }
}

fn disabled_by_env() -> bool {
    use crate::config::{read, Switch, GUEST_IMPORT, UDMABUF};
    read(UDMABUF).0 == Switch::Off || read(GUEST_IMPORT).0 == Switch::Off
}

/// Ask the device, the environment and the process's dma-buf source.
///
/// # Safety
///
/// `instance` and `pd` must be live.
pub unsafe fn query(
    instance: &ash::Instance,
    pd: vk::PhysicalDevice,
    has_extension: &dyn Fn(&std::ffi::CStr) -> bool,
    max_allocation: u64,
    host_pointer_available: bool,
) -> DmaBufImportCaps {
    if host_pointer_available {
        return DmaBufImportCaps::refused(DmaBufImport::HostPointerPreferred);
    }
    if disabled_by_env() {
        return DmaBufImportCaps::refused(DmaBufImport::DisabledByEnv);
    }
    if !has_extension(vk::EXT_EXTERNAL_MEMORY_DMA_BUF_NAME)
        || !has_extension(vk::KHR_EXTERNAL_MEMORY_FD_NAME)
    {
        return DmaBufImportCaps::refused(DmaBufImport::NoDmaBufExtension);
    }
    let info = vk::PhysicalDeviceExternalBufferInfo::default()
        .usage(GUEST_IMPORT_USAGE)
        .handle_type(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT);
    let mut props = vk::ExternalBufferProperties::default();
    unsafe { instance.get_physical_device_external_buffer_properties(pd, &info, &mut props) };
    if !props
        .external_memory_properties
        .external_memory_features
        .contains(vk::ExternalMemoryFeatureFlags::IMPORTABLE)
    {
        return DmaBufImportCaps::refused(DmaBufImport::NotImportable);
    }
    let Some(source) = crate::runtime::guest_ram::dmabuf::source() else {
        return DmaBufImportCaps::refused(DmaBufImport::NoSource);
    };
    let granularity = source.granularity();
    if granularity == 0 || !granularity.is_power_of_two() || source.size_limit() < granularity {
        return DmaBufImportCaps::refused(DmaBufImport::AlignmentUnsatisfiable);
    }
    let memory = unsafe { instance.get_physical_device_memory_properties(pd) };
    let heap_budget = super::memory_topology::roomiest_heap_for(
        &memory,
        &super::memory_topology::classify_memory(&memory)
            .topology
            .request(super::MemoryClass::Upload),
    );
    let mut v13 = vk::PhysicalDeviceVulkan13Properties::default();
    let device_api = {
        let mut base = vk::PhysicalDeviceProperties2::default();
        unsafe { instance.get_physical_device_properties2(pd, &mut base) };
        base.properties.api_version
    };
    if vk::api_version_major(device_api) > 1 || vk::api_version_minor(device_api) >= 3 {
        let mut limits = vk::PhysicalDeviceProperties2::default().push_next(&mut v13);
        unsafe { instance.get_physical_device_properties2(pd, &mut limits) };
    }
    let span_max = span_max(
        max_allocation,
        v13.max_buffer_size,
        source.size_limit(),
        granularity,
    );
    DmaBufImportCaps {
        rung: DmaBufImport::Supported,
        granularity,
        heap_budget,
        span_max,
    }
}

/// The largest chunk: the smallest of the four limits, cut to the granularity.
/// A `max_buffer_size` of zero is a device that did not report one.
fn span_max(max_allocation: u64, max_buffer_size: u64, size_limit: u64, granularity: u64) -> u64 {
    [
        IMPORT_SPAN_CEILING,
        max_allocation,
        if max_buffer_size == 0 {
            u64::MAX
        } else {
            max_buffer_size
        },
        size_limit,
    ]
    .into_iter()
    .min()
    .unwrap_or(IMPORT_SPAN_CEILING)
        & !(granularity - 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// At the udmabuf module's stock limit the chunk is 64 MiB on a 16 KiB-page
    /// host, and a raised limit is still held under the host-pointer ceiling.
    #[test]
    fn the_chunk_is_the_smallest_limit_cut_to_the_page() {
        const MIB: u64 = 1 << 20;
        assert_eq!(span_max(u64::MAX, 0, 64 * MIB, 16384), 64 * MIB);
        assert_eq!(
            span_max(u64::MAX, 0, 4096 * MIB, 16384),
            IMPORT_SPAN_CEILING
        );
        assert_eq!(span_max(u64::MAX, 0, 64 * MIB + 100, 16384), 64 * MIB);
        assert_eq!(span_max(32 * MIB, 0, 64 * MIB, 16384), 32 * MIB);
    }

    #[test]
    fn every_rung_has_its_own_slug_and_only_supported_needs_extensions() {
        let all = [
            DmaBufImport::Supported,
            DmaBufImport::Unqueried,
            DmaBufImport::HostPointerPreferred,
            DmaBufImport::NoDmaBufExtension,
            DmaBufImport::NotImportable,
            DmaBufImport::NoSource,
            DmaBufImport::AlignmentUnsatisfiable,
            DmaBufImport::DisabledByEnv,
        ];
        let mut slugs: Vec<_> = all.iter().map(|r| r.slug()).collect();
        slugs.sort();
        slugs.dedup();
        assert_eq!(slugs.len(), all.len());
        for r in all {
            assert_eq!(r.required_extensions().is_empty(), !r.is_available());
        }
    }
}
