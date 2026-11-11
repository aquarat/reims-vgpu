//! Guest RAM as dma-bufs: the second way a range of guest memory can reach the
//! GPU without a copy.
//!
//! The first is a host-pointer import (`VK_EXT_external_memory_host`) of this
//! process's own mapping. A driver without it can still import a dma-buf, and a
//! udmabuf is a dma-buf over the pages of a memfd — which is what guest RAM is
//! when the VMM backs it with `memory-backend-memfd`. Making one needs
//! `/dev/udmabuf` and the memfd; the out-of-process backend holds the memfds
//! already (the VMM's `MEM_TABLE`) and is handed an open `/dev/udmabuf` by the
//! VMM, so it needs no device node and gains access to nothing it could not
//! already read and write.
//!
//! This module is only the meeting point. The process that knows which file
//! backs a host address registers a [`DmaBufSource`]; the GPU backend asks it
//! for a dma-buf of exactly the range an import covers. Nothing here opens a
//! device or issues an ioctl, so the crate stays free of the platform.
//!
//! # What a udmabuf costs
//!
//! It **pins** every page it covers for as long as it lives: the kernel faults
//! them in when it is made and holds them until the last reference (the fd and
//! the GPU's import) goes. So a source is asked for a range only when the
//! device is about to bind it, never ahead of time, and the pinned total is
//! what the GPU has touched. A page the guest gives back through a balloon would
//! stay pinned under the dma-buf and go stale for the GPU, which is why the VMM
//! does not offer `/dev/udmabuf` when a balloon is present.

use std::os::fd::OwnedFd;
use std::sync::OnceLock;

/// Why a source could not make a dma-buf of a range.
///
/// Each is a routing answer, not a failure of the guest's work: the import is
/// declined and the copying rails carry the bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DmaBufRefusal {
    /// The range is not inside guest RAM this process maps, or not inside one
    /// packed view of it.
    NotGuestRam,
    /// The range is not aligned to the host page size the kernel pins in.
    Unaligned,
    /// Longer than the largest dma-buf the udmabuf module will make
    /// (`size_limit_mb`).
    OverSizeLimit { len: u64, limit: u64 },
    /// A packed view whose pages fall in more separate stretches of guest RAM
    /// than one `UDMABUF_CREATE_LIST` may name (`list_limit`).
    OverListLimit { runs: u32, limit: u32 },
    /// The kernel refused: `errno` from `UDMABUF_CREATE` or `_CREATE_LIST`.
    Kernel { errno: i32 },
}

impl DmaBufRefusal {
    /// Stable slug for a decline line.
    pub fn slug(self) -> &'static str {
        match self {
            Self::NotGuestRam => "dmabuf_not_guest_ram",
            Self::Unaligned => "dmabuf_unaligned",
            Self::OverSizeLimit { .. } => "dmabuf_over_size_limit",
            Self::OverListLimit { .. } => "dmabuf_over_list_limit",
            Self::Kernel { .. } => "dmabuf_kernel_refused",
        }
    }
}

/// Something that can turn a range of this process's guest-RAM mappings into a
/// dma-buf of the same pages.
pub trait DmaBufSource: Send + Sync {
    /// A dma-buf of exactly `[host_va, host_va + len)`, whose page `i` is the
    /// page this process sees at `host_va + i * page`. The caller owns the fd.
    fn dmabuf(&self, host_va: usize, len: u64) -> Result<OwnedFd, DmaBufRefusal>;

    /// The largest dma-buf this source can make, in bytes. Imports are cut to
    /// fit it rather than refused at it.
    fn size_limit(&self) -> u64;

    /// The page size the kernel pins in: every range's base and length must be
    /// a multiple of it.
    fn granularity(&self) -> u64;

    /// A one-line description for the log (the limits it was given).
    fn describe(&self) -> String;
}

static SOURCE: OnceLock<Box<dyn DmaBufSource>> = OnceLock::new();

/// Register the process's source. Once per process: the backend serves one VMM
/// and a second offer is a protocol error the caller reports.
///
/// Returns `false` when a source is already registered.
pub fn register_source(source: Box<dyn DmaBufSource>) -> bool {
    SOURCE.set(source).is_ok()
}

/// The registered source, if the VMM offered one.
pub fn source() -> Option<&'static dyn DmaBufSource> {
    SOURCE.get().map(|b| b.as_ref())
}

static PINNED: std::sync::atomic::AtomicI64 = std::sync::atomic::AtomicI64::new(0);

/// Count bytes a live dma-buf import pins (positive when one is made, negative
/// when its memory is freed).
pub fn note_pinned(delta: i64) {
    PINNED.fetch_add(delta, std::sync::atomic::Ordering::Relaxed);
}

/// Bytes of guest RAM the device's live dma-buf imports pin right now.
pub fn pinned_bytes() -> u64 {
    PINNED.load(std::sync::atomic::Ordering::Relaxed).max(0) as u64
}

/// Every refusal has its own slug, so a log names the check that refused.
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_refusal_has_its_own_slug() {
        let all = [
            DmaBufRefusal::NotGuestRam,
            DmaBufRefusal::Unaligned,
            DmaBufRefusal::OverSizeLimit { len: 0, limit: 0 },
            DmaBufRefusal::OverListLimit { runs: 0, limit: 0 },
            DmaBufRefusal::Kernel { errno: 0 },
        ];
        let mut slugs: Vec<_> = all.iter().map(|r| r.slug()).collect();
        slugs.sort();
        slugs.dedup();
        assert_eq!(slugs.len(), all.len());
    }
}
