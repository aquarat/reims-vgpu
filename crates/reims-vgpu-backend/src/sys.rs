//! Thin wrappers over the Linux primitives the protocol is built from:
//! memfds, eventfds, shared mappings and SOCK_SEQPACKET messages with fds.

use std::io;
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd, RawFd};
use std::ptr;

use reims_vgpu_proxy_wire::{self as wire, Hdr, Pod};

pub fn last_err(what: &str) -> io::Error {
    let e = io::Error::last_os_error();
    io::Error::new(e.kind(), format!("{what}: {e}"))
}

pub fn memfd(name: &str, size: u64) -> io::Result<OwnedFd> {
    let cname = std::ffi::CString::new(name).unwrap();
    // SAFETY: valid NUL-terminated name.
    let fd = unsafe { libc::memfd_create(cname.as_ptr(), libc::MFD_CLOEXEC) };
    if fd < 0 {
        return Err(last_err("memfd_create"));
    }
    // SAFETY: fd is ours.
    let fd = unsafe { OwnedFd::from_raw_fd(fd) };
    // SAFETY: plain syscall on our fd.
    if unsafe { libc::ftruncate(fd.as_raw_fd(), size as libc::off_t) } != 0 {
        return Err(last_err("ftruncate"));
    }
    Ok(fd)
}

pub fn eventfd() -> io::Result<OwnedFd> {
    // SAFETY: plain syscall.
    let fd = unsafe { libc::eventfd(0, libc::EFD_CLOEXEC) };
    if fd < 0 {
        return Err(last_err("eventfd"));
    }
    // SAFETY: fd is ours.
    Ok(unsafe { OwnedFd::from_raw_fd(fd) })
}

/// Add 1 to an eventfd. Never blocks in practice (the counter would have to
/// reach 2^64-2).
pub fn eventfd_signal(fd: RawFd) {
    let one: u64 = 1;
    // SAFETY: 8-byte write from a valid u64.
    unsafe { libc::write(fd, &one as *const u64 as *const libc::c_void, 8) };
}

/// Block until the eventfd is non-zero, then reset it. Returns false on error.
pub fn eventfd_wait(fd: RawFd) -> bool {
    let mut v: u64 = 0;
    loop {
        // SAFETY: 8-byte read into a valid u64.
        let n = unsafe { libc::read(fd, &mut v as *mut u64 as *mut libc::c_void, 8) };
        if n == 8 {
            return true;
        }
        if n < 0 && io::Error::last_os_error().kind() == io::ErrorKind::Interrupted {
            continue;
        }
        return false;
    }
}

/// A shared read-write mapping of a whole fd, kept for the process lifetime.
pub fn map_shared(fd: RawFd, size: u64, offset: u64) -> io::Result<*mut u8> {
    // SAFETY: fresh mapping chosen by the kernel.
    let p = unsafe {
        libc::mmap(
            ptr::null_mut(),
            size as usize,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_SHARED,
            fd,
            offset as libc::off_t,
        )
    };
    if p == libc::MAP_FAILED {
        return Err(last_err("mmap"));
    }
    Ok(p as *mut u8)
}

/// Map `size` bytes of `fd` at `offset`, at an address aligned to `align`, so
/// the kernel can use huge-page (PMD) mappings for a THP-backed memfd.
pub fn map_shared_aligned(fd: RawFd, size: u64, offset: u64, align: u64) -> io::Result<*mut u8> {
    let span = size as usize + align as usize;
    // SAFETY: a reservation the kernel places; nothing else uses it.
    let res = unsafe {
        libc::mmap(
            ptr::null_mut(),
            span,
            libc::PROT_NONE,
            libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_NORESERVE,
            -1,
            0,
        )
    };
    if res == libc::MAP_FAILED {
        return Err(last_err("mmap reserve"));
    }
    let base = res as usize;
    let aligned = (base + align as usize - 1) & !(align as usize - 1);
    // SAFETY: replaces part of our own reservation.
    let p = unsafe {
        libc::mmap(
            aligned as *mut libc::c_void,
            size as usize,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_SHARED | libc::MAP_FIXED,
            fd,
            offset as libc::off_t,
        )
    };
    if p == libc::MAP_FAILED {
        // SAFETY: our reservation.
        unsafe { libc::munmap(res, span) };
        return Err(last_err("mmap fixed"));
    }
    // Trim the reservation around the mapping.
    // SAFETY: both pieces are parts of our reservation outside the mapping.
    unsafe {
        if aligned > base {
            libc::munmap(res, aligned - base);
        }
        let end = aligned + size as usize;
        if base + span > end {
            libc::munmap(end as *mut libc::c_void, base + span - end);
        }
        libc::madvise(p, size as usize, libc::MADV_HUGEPAGE);
    }
    Ok(p as *mut u8)
}

/// One received message.
pub struct Msg {
    pub hdr: Hdr,
    pub payload: Vec<u8>,
    pub fds: Vec<OwnedFd>,
}

const HDR: usize = std::mem::size_of::<Hdr>();

/// Send one message (header + payload) with `fds` attached.
pub fn send(fd: RawFd, hdr: &Hdr, payload: &[u8], fds: &[RawFd], dontwait: bool) -> io::Result<()> {
    let mut buf = Vec::with_capacity(HDR + payload.len());
    buf.extend_from_slice(hdr.as_bytes());
    buf.extend_from_slice(payload);
    let mut iov = libc::iovec {
        iov_base: buf.as_ptr() as *mut libc::c_void,
        iov_len: buf.len(),
    };
    let fd_bytes = std::mem::size_of_val(fds);
    // SAFETY: CMSG_SPACE is a pure computation.
    let space = unsafe { libc::CMSG_SPACE(fd_bytes as u32) } as usize;
    let mut cbuf = vec![0u8; if fds.is_empty() { 0 } else { space }];
    // SAFETY: zeroed msghdr is a valid starting point.
    let mut mh: libc::msghdr = unsafe { std::mem::zeroed() };
    mh.msg_iov = &mut iov;
    mh.msg_iovlen = 1;
    if !fds.is_empty() {
        mh.msg_control = cbuf.as_mut_ptr() as *mut libc::c_void;
        mh.msg_controllen = space as _;
        // SAFETY: control buffer sized with CMSG_SPACE for these fds.
        unsafe {
            let cm = libc::CMSG_FIRSTHDR(&mh);
            (*cm).cmsg_level = libc::SOL_SOCKET;
            (*cm).cmsg_type = libc::SCM_RIGHTS;
            (*cm).cmsg_len = libc::CMSG_LEN(fd_bytes as u32) as _;
            ptr::copy_nonoverlapping(fds.as_ptr() as *const u8, libc::CMSG_DATA(cm), fd_bytes);
        }
    }
    let flags = libc::MSG_NOSIGNAL | if dontwait { libc::MSG_DONTWAIT } else { 0 };
    loop {
        // SAFETY: msghdr points at live buffers.
        let n = unsafe { libc::sendmsg(fd, &mh, flags) };
        if n >= 0 {
            return Ok(());
        }
        let e = io::Error::last_os_error();
        if e.kind() != io::ErrorKind::Interrupted {
            return Err(e);
        }
    }
}

pub fn send_pod<T: Pod>(fd: RawFd, type_: u32, flags: u32, seq: u32, body: &T) -> io::Result<()> {
    let hdr = Hdr {
        type_,
        flags,
        size: std::mem::size_of::<T>() as u32,
        seq,
    };
    send(fd, &hdr, body.as_bytes(), &[], false)
}

/// Receive one message. `Ok(None)` is an orderly EOF.
pub fn recv(fd: RawFd) -> io::Result<Option<Msg>> {
    let mut buf = vec![0u8; HDR + wire::MAX_PAYLOAD as usize];
    let mut iov = libc::iovec {
        iov_base: buf.as_mut_ptr() as *mut libc::c_void,
        iov_len: buf.len(),
    };
    let fd_bytes = wire::MAX_FDS as usize * std::mem::size_of::<RawFd>();
    // SAFETY: pure computation.
    let space = unsafe { libc::CMSG_SPACE(fd_bytes as u32) } as usize;
    let mut cbuf = vec![0u8; space];
    // SAFETY: zeroed msghdr is a valid starting point.
    let mut mh: libc::msghdr = unsafe { std::mem::zeroed() };
    mh.msg_iov = &mut iov;
    mh.msg_iovlen = 1;
    mh.msg_control = cbuf.as_mut_ptr() as *mut libc::c_void;
    mh.msg_controllen = space as _;
    let n = loop {
        // SAFETY: msghdr points at live buffers.
        let n = unsafe { libc::recvmsg(fd, &mut mh, libc::MSG_CMSG_CLOEXEC) };
        if n >= 0 {
            break n as usize;
        }
        let e = io::Error::last_os_error();
        if e.kind() != io::ErrorKind::Interrupted {
            return Err(e);
        }
    };
    if n == 0 {
        return Ok(None);
    }
    let mut fds = Vec::new();
    // SAFETY: walking the control buffer recvmsg filled.
    unsafe {
        let mut cm = libc::CMSG_FIRSTHDR(&mh);
        while !cm.is_null() {
            if (*cm).cmsg_level == libc::SOL_SOCKET && (*cm).cmsg_type == libc::SCM_RIGHTS {
                let data = libc::CMSG_DATA(cm) as *const RawFd;
                let count = ((*cm).cmsg_len as usize - libc::CMSG_LEN(0) as usize)
                    / std::mem::size_of::<RawFd>();
                for i in 0..count {
                    fds.push(OwnedFd::from_raw_fd(ptr::read_unaligned(data.add(i))));
                }
            }
            cm = libc::CMSG_NXTHDR(&mh, cm);
        }
    }
    if n < HDR || mh.msg_flags & libc::MSG_TRUNC != 0 {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "short or truncated message"));
    }
    let hdr = Hdr::read_from(&buf).unwrap();
    if hdr.size as usize != n - HDR {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "size field disagrees"));
    }
    buf.truncate(n);
    buf.drain(..HDR);
    Ok(Some(Msg {
        hdr,
        payload: buf,
        fds,
    }))
}

pub fn now_ns(clock: libc::clockid_t) -> u64 {
    // SAFETY: zeroed timespec, valid clock id.
    let mut ts: libc::timespec = unsafe { std::mem::zeroed() };
    unsafe { libc::clock_gettime(clock, &mut ts) };
    ts.tv_sec as u64 * 1_000_000_000 + ts.tv_nsec as u64
}
