//! Guest RAM, mapped from the VMM's own memory-backend fds.
//!
//! The VMM describes each writable RAM region as (gpa, size, fd, fd offset).
//! Mapping the same fd MAP_SHARED makes this process's view *the* guest's
//! memory: stores land in the guest's pages, a balloon hole punched in the fd
//! shows up here at once, and nothing has to be copied or kept in sync.

use std::io;
use std::os::fd::{AsRawFd, OwnedFd, RawFd};
use std::sync::Mutex;

use crate::sys;

/// The alignment the VMM gives its RAM so a THP-backed memfd maps with PMDs
/// (32 MiB on a 16 KiB-page kernel, a multiple of 2 MiB elsewhere).
const MAP_ALIGN: u64 = 32 << 20;

pub struct Region {
    pub gpa: u64,
    pub size: u64,
    pub hva: *mut u8,
    fd: OwnedFd,
    fd_offset: u64,
}

// SAFETY: hva is a process-lifetime shared mapping; access is via copies.
unsafe impl Send for Region {}
unsafe impl Sync for Region {}

#[derive(Default)]
pub struct GuestRam {
    regions: Vec<Region>,
    /// Packed views handed out by map_pages.
    views: Mutex<Vec<View>>,
}

/// One packed view: its address and length, and the stretches of guest-RAM
/// files it is made of, in view order.
struct View {
    addr: usize,
    len: usize,
    runs: Vec<FileRun>,
}

/// `len` bytes of a guest-RAM file at `offset`. The fd is a region's, and a
/// region's fd lives for the process (a replaced table is leaked, not closed).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FileRun {
    pub fd: RawFd,
    pub offset: u64,
    pub len: u64,
}

impl GuestRam {
    /// Map a new table. Old mappings are kept (leaked) rather than unmapped:
    /// a drain may still be copying through them, and the table only changes
    /// on hotplug, which this machine does not have.
    pub fn set_table(&mut self, regions: Vec<(u64, u64, u64, OwnedFd)>) -> io::Result<()> {
        let mut out = Vec::with_capacity(regions.len());
        for (gpa, size, fd_offset, fd) in regions {
            let hva = sys::map_shared_aligned(fd.as_raw_fd(), size, fd_offset, MAP_ALIGN)?;
            out.push(Region {
                gpa,
                size,
                hva,
                fd,
                fd_offset,
            });
        }
        out.sort_by_key(|r| r.gpa);
        std::mem::forget(std::mem::replace(&mut self.regions, out));
        Ok(())
    }

    pub fn regions(&self) -> &[Region] {
        &self.regions
    }

    fn find(&self, gpa: u64) -> Option<&Region> {
        self.regions
            .iter()
            .find(|r| gpa >= r.gpa && gpa - r.gpa < r.size)
    }

    pub fn is_ram(&self, gpa: u64) -> bool {
        self.find(gpa).is_some()
    }

    /// Copy guest bytes out; a range spanning adjacent regions is split.
    pub fn read(&self, mut gpa: u64, buf: &mut [u8]) -> bool {
        let mut done = 0usize;
        while done < buf.len() {
            let Some(r) = self.find(gpa) else {
                return false;
            };
            let off = gpa - r.gpa;
            let n = ((r.size - off) as usize).min(buf.len() - done);
            // SAFETY: source within the region mapping, destination within buf.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    r.hva.add(off as usize),
                    buf.as_mut_ptr().add(done),
                    n,
                )
            };
            done += n;
            gpa += n as u64;
        }
        true
    }

    pub fn write(&self, mut gpa: u64, buf: &[u8]) -> bool {
        let mut done = 0usize;
        while done < buf.len() {
            let Some(r) = self.find(gpa) else {
                return false;
            };
            let off = gpa - r.gpa;
            let n = ((r.size - off) as usize).min(buf.len() - done);
            // SAFETY: destination within the region mapping, source within buf.
            unsafe {
                std::ptr::copy_nonoverlapping(buf.as_ptr().add(done), r.hva.add(off as usize), n)
            };
            done += n;
            gpa += n as u64;
        }
        true
    }

    /// One contiguous host view of `gpas` (each `page` bytes). A run that is
    /// contiguous in this process's mapping is returned directly; otherwise a
    /// packed view is built by mapping each page's fd offset into one
    /// reservation, which the caller releases with `unmap_pages`.
    ///
    /// Err carries (stage, page_index, errno) for the failure report.
    pub fn map_pages(&self, gpas: &[u64], page: u64) -> Result<*mut u8, (u32, u64, i32)> {
        const RESERVATION: u32 = 1;
        const ALIAS: u32 = 2;
        const INVALID: u32 = 3;
        if gpas.is_empty() {
            return Err((INVALID, u64::MAX, 0));
        }
        let mut hvas = Vec::with_capacity(gpas.len());
        let mut owners = Vec::with_capacity(gpas.len());
        for (i, &gpa) in gpas.iter().enumerate() {
            let Some(r) = self.find(gpa) else {
                return Err((INVALID, i as u64, 0));
            };
            let off = gpa - r.gpa;
            if off % page != 0 || page > r.size - off {
                return Err((INVALID, i as u64, 0));
            }
            // SAFETY: within the region mapping.
            hvas.push(unsafe { r.hva.add(off as usize) } as usize);
            owners.push((r.fd.as_raw_fd(), r.fd_offset + off));
        }
        let page_us = page as usize;
        if hvas.windows(2).all(|w| w[1] == w[0] + page_us) {
            return Ok(hvas[0] as *mut u8);
        }
        let len = gpas.len() * page_us;
        // SAFETY: a fresh reservation.
        let view = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                len,
                libc::PROT_NONE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_NORESERVE,
                -1,
                0,
            )
        };
        if view == libc::MAP_FAILED {
            return Err((
                RESERVATION,
                0,
                io::Error::last_os_error().raw_os_error().unwrap_or(0),
            ));
        }
        let mut i = 0;
        let mut runs = Vec::new();
        while i < gpas.len() {
            let (fd, foff) = owners[i];
            let mut run = 1;
            while i + run < gpas.len()
                && owners[i + run].0 == fd
                && owners[i + run].1 == foff + (run as u64) * page
            {
                run += 1;
            }
            // SAFETY: replaces part of our own reservation.
            let p = unsafe {
                libc::mmap(
                    (view as usize + i * page_us) as *mut libc::c_void,
                    run * page_us,
                    libc::PROT_READ | libc::PROT_WRITE,
                    libc::MAP_SHARED | libc::MAP_FIXED,
                    fd,
                    foff as libc::off_t,
                )
            };
            if p == libc::MAP_FAILED {
                let errno = io::Error::last_os_error().raw_os_error().unwrap_or(0);
                // SAFETY: our reservation.
                unsafe { libc::munmap(view, len) };
                return Err((ALIAS, i as u64, errno));
            }
            runs.push(FileRun {
                fd,
                offset: foff,
                len: (run * page_us) as u64,
            });
            i += run;
        }
        self.views.lock().unwrap().push(View {
            addr: view as usize,
            len,
            runs,
        });
        Ok(view as *mut u8)
    }

    /// Release a packed view. A pointer that is not one of ours is a direct
    /// mapping of guest RAM and is left alone.
    pub fn unmap_pages(&self, ptr: *mut u8) {
        let mut views = self.views.lock().unwrap();
        if let Some(i) = views.iter().position(|v| v.addr == ptr as usize) {
            let v = views.swap_remove(i);
            // SAFETY: a view this type created.
            unsafe { libc::munmap(v.addr as *mut libc::c_void, v.len) };
        }
    }

    /// Drop every packed view (device reset).
    pub fn unmap_all(&self) -> usize {
        let mut views = self.views.lock().unwrap();
        let n = views.len();
        for v in views.drain(..) {
            // SAFETY: views this type created.
            unsafe { libc::munmap(v.addr as *mut libc::c_void, v.len) };
        }
        n
    }

    /// The guest-RAM file stretches behind `[host_va, host_va + len)` of this
    /// process's view of guest RAM: one stretch inside a region's own mapping,
    /// or the slice of a packed view's stretches. `None` when the range is in
    /// neither, or straddles a region's end or a view's end.
    pub fn file_runs(&self, host_va: usize, len: u64) -> Option<Vec<FileRun>> {
        if len == 0 {
            return None;
        }
        let end = (host_va as u64).checked_add(len)?;
        for r in &self.regions {
            let base = r.hva as u64;
            if host_va as u64 >= base && end <= base + r.size {
                return Some(vec![FileRun {
                    fd: r.fd.as_raw_fd(),
                    offset: r.fd_offset + (host_va as u64 - base),
                    len,
                }]);
            }
        }
        let views = self.views.lock().unwrap();
        let v = views
            .iter()
            .find(|v| host_va >= v.addr && end <= (v.addr + v.len) as u64)?;
        Some(slice_runs(&v.runs, (host_va - v.addr) as u64, len))
    }
}

/// The part of `runs` (laid end to end) covering `[start, start + len)`.
fn slice_runs(runs: &[FileRun], start: u64, len: u64) -> Vec<FileRun> {
    let end = start + len;
    let mut out = Vec::new();
    let mut at = 0u64;
    for r in runs {
        let (lo, hi) = (at, at + r.len);
        at = hi;
        if hi <= start || lo >= end {
            continue;
        }
        let from = start.max(lo);
        let to = end.min(hi);
        out.push(FileRun {
            fd: r.fd,
            offset: r.offset + (from - lo),
            len: to - from,
        });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(fd: RawFd, offset: u64, len: u64) -> FileRun {
        FileRun { fd, offset, len }
    }

    /// A slice of a packed view keeps each stretch's file and moves its offset
    /// by how far into the stretch the slice begins.
    #[test]
    fn a_slice_of_a_view_keeps_files_and_offsets() {
        let runs = [
            run(3, 0x10000, 0x8000),
            run(3, 0x40000, 0x4000),
            run(4, 0, 0x8000),
        ];
        assert_eq!(slice_runs(&runs, 0, 0xc000), runs[..2].to_vec());
        assert_eq!(slice_runs(&runs, 0, 0x14000), runs.to_vec());
        assert_eq!(
            slice_runs(&runs, 0x4000, 0x8000),
            vec![run(3, 0x14000, 0x4000), run(3, 0x40000, 0x4000)]
        );
        assert_eq!(slice_runs(&runs, 0xc000, 0x4000), vec![run(4, 0, 0x4000)]);
        assert_eq!(slice_runs(&runs, 0x14000, 0x1000), vec![]);
    }
}
