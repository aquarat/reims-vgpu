//! Guest RAM as udmabufs, made with the `/dev/udmabuf` the VMM hands over.
//!
//! The VMM sends an open `/dev/udmabuf` in an `APV_PROXY_UDMABUF` request (see
//! the protocol header) together with the udmabuf module's limits. With it this
//! process makes dma-bufs of the guest-RAM memfds it already maps, when the GPU
//! backend asks for one (`reims_vgpu::runtime::guest_ram::dmabuf`): never ahead
//! of time, because a udmabuf pins its pages. The fd lets this process do
//! nothing to memory it could not already read and write.

use std::io;
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};

use reims_vgpu::runtime::guest_ram::dmabuf::{DmaBufRefusal, DmaBufSource};

use crate::backend::Backend;
use crate::guest_ram::FileRun;

/// `struct udmabuf_create` (linux/udmabuf.h).
#[repr(C)]
struct Create {
    memfd: u32,
    flags: u32,
    offset: u64,
    size: u64,
}

/// `struct udmabuf_create_item`.
#[repr(C)]
#[derive(Clone, Copy)]
struct Item {
    memfd: u32,
    pad: u32,
    offset: u64,
    size: u64,
}

/// `struct udmabuf_create_list` without its flexible array.
#[repr(C)]
struct ListHead {
    flags: u32,
    count: u32,
}

const UDMABUF_FLAGS_CLOEXEC: u32 = 0x01;

/// `_IOW('u', nr, size)` on Linux: write direction, size, type, number.
const fn iow(nr: u64, size: usize) -> u64 {
    (1 << 30) | ((size as u64) << 16) | ((b'u' as u64) << 8) | nr
}
const UDMABUF_CREATE: u64 = iow(0x42, std::mem::size_of::<Create>());
const UDMABUF_CREATE_LIST: u64 = iow(0x43, std::mem::size_of::<ListHead>());

const _: () = assert!(std::mem::size_of::<Create>() == 24);
const _: () = assert!(std::mem::size_of::<Item>() == 24);
const _: () = assert!(std::mem::size_of::<ListHead>() == 8);

/// The VMM's `/dev/udmabuf` and the limits it read, over this backend's guest
/// RAM.
pub struct Udmabuf {
    dev: OwnedFd,
    size_limit: u64,
    list_limit: u32,
    page: u64,
    backend: &'static Backend,
}

impl Udmabuf {
    pub fn new(dev: OwnedFd, size_limit: u64, list_limit: u32, backend: &'static Backend) -> Self {
        // SAFETY: sysconf has no preconditions.
        let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) }.max(4096) as u64;
        Self {
            dev,
            size_limit,
            list_limit,
            page,
            backend,
        }
    }

    fn create(&self, runs: &[FileRun]) -> Result<OwnedFd, DmaBufRefusal> {
        let rc = if let [one] = runs {
            let req = Create {
                memfd: one.fd as u32,
                flags: UDMABUF_FLAGS_CLOEXEC,
                offset: one.offset,
                size: one.len,
            };
            // SAFETY: a valid request for this ioctl; the kernel reads it only.
            unsafe { libc::ioctl(self.dev.as_raw_fd(), UDMABUF_CREATE as _, &req) }
        } else {
            let mut buf: Vec<u64> = vec![
                0;
                (std::mem::size_of::<ListHead>()
                    + runs.len() * std::mem::size_of::<Item>())
                .div_ceil(8)
            ];
            // SAFETY: `buf` is large and aligned enough for the head and items.
            unsafe {
                let head = buf.as_mut_ptr() as *mut ListHead;
                (*head).flags = UDMABUF_FLAGS_CLOEXEC;
                (*head).count = runs.len() as u32;
                let items = (head as *mut u8).add(std::mem::size_of::<ListHead>()) as *mut Item;
                for (i, r) in runs.iter().enumerate() {
                    *items.add(i) = Item {
                        memfd: r.fd as u32,
                        pad: 0,
                        offset: r.offset,
                        size: r.len,
                    };
                }
            }
            // SAFETY: as above; the kernel reads `count` items after the head.
            unsafe { libc::ioctl(self.dev.as_raw_fd(), UDMABUF_CREATE_LIST as _, buf.as_ptr()) }
        };
        if rc < 0 {
            let errno = io::Error::last_os_error().raw_os_error().unwrap_or(0);
            return Err(DmaBufRefusal::Kernel { errno });
        }
        // SAFETY: the ioctl returned a new fd this process owns.
        Ok(unsafe { OwnedFd::from_raw_fd(rc) })
    }
}

impl DmaBufSource for Udmabuf {
    fn dmabuf(&self, host_va: usize, len: u64) -> Result<OwnedFd, DmaBufRefusal> {
        if len == 0 || host_va as u64 % self.page != 0 || len % self.page != 0 {
            return Err(DmaBufRefusal::Unaligned);
        }
        if len > self.size_limit {
            return Err(DmaBufRefusal::OverSizeLimit {
                len,
                limit: self.size_limit,
            });
        }
        // Held across the ioctl: the runs name fds of the current table.
        let ram = self.backend.ram.read().unwrap();
        let runs = ram
            .file_runs(host_va, len)
            .ok_or(DmaBufRefusal::NotGuestRam)?;
        if runs.len() > 1 && runs.len() as u64 > u64::from(self.list_limit) {
            return Err(DmaBufRefusal::OverListLimit {
                runs: runs.len() as u32,
                limit: self.list_limit,
            });
        }
        self.create(&runs)
    }

    fn size_limit(&self) -> u64 {
        self.size_limit
    }

    fn granularity(&self) -> u64 {
        self.page
    }

    fn describe(&self) -> String {
        format!(
            "udmabuf size_limit={} MiB list_limit={} page={}",
            self.size_limit >> 20,
            self.list_limit,
            self.page
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The request numbers the kernel's linux/udmabuf.h spells.
    #[test]
    fn the_ioctl_numbers_are_the_kernels() {
        assert_eq!(UDMABUF_CREATE, 0x4018_7542);
        assert_eq!(UDMABUF_CREATE_LIST, 0x4008_7543);
    }
}
