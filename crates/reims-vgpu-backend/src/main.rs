//! reims-vgpu-backend: the Reims paravirtual GPU device as its own process.
//!
//! The VMM (QEMU's `apv-gpu-proxy` device) connects to the socket this process
//! listens on and speaks the apv-gpu-proxy protocol
//! (`reims-vgpu-proxy-wire/include/apv-gpu-proxy-protocol.h`). One connection,
//! one device; when the VMM goes away, so does this process.
//!
//! usage: reims-vgpu-backend --socket PATH
//!        reims-vgpu-backend --listen-fd N      (an already listening socket)
//!        reims-vgpu-backend --fold-breadcrumbs (after an instance died: move a
//!            driver-call breadcrumb left in REIMS_VGPU_BREADCRUMB_DIR into the
//!            host's driver quarantine; prints the call it was, if any)

mod backend;
mod dirty;
mod guest_ram;
mod host_ops;
mod regs;
mod sys;

use std::io;
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd, RawFd};
use std::sync::atomic::Ordering;

use reims_vgpu::qemu::abi as rabi;
use reims_vgpu_proxy_wire::{self as wire, Pod};

use backend::Backend;
use host_ops::{CallCtx, CALL};

fn listen(path: &str) -> io::Result<OwnedFd> {
    let _ = std::fs::remove_file(path);
    // SAFETY: plain syscall.
    let fd = unsafe { libc::socket(libc::AF_UNIX, libc::SOCK_SEQPACKET | libc::SOCK_CLOEXEC, 0) };
    if fd < 0 {
        return Err(sys::last_err("socket"));
    }
    // SAFETY: fresh fd.
    let fd = unsafe { OwnedFd::from_raw_fd(fd) };
    // SAFETY: zeroed sockaddr_un is valid; the path is bounds-checked below.
    let mut addr: libc::sockaddr_un = unsafe { std::mem::zeroed() };
    addr.sun_family = libc::AF_UNIX as libc::sa_family_t;
    let bytes = path.as_bytes();
    if bytes.len() >= addr.sun_path.len() {
        return Err(io::Error::new(io::ErrorKind::InvalidInput, "socket path too long"));
    }
    for (d, s) in addr.sun_path.iter_mut().zip(bytes) {
        *d = *s as libc::c_char;
    }
    // SAFETY: addr is a valid sockaddr_un.
    if unsafe {
        libc::bind(
            fd.as_raw_fd(),
            &addr as *const libc::sockaddr_un as *const libc::sockaddr,
            std::mem::size_of::<libc::sockaddr_un>() as libc::socklen_t,
        )
    } != 0
    {
        return Err(sys::last_err("bind"));
    }
    // SAFETY: plain syscall.
    if unsafe { libc::listen(fd.as_raw_fd(), 1) } != 0 {
        return Err(sys::last_err("listen"));
    }
    Ok(fd)
}

fn accept(listener: RawFd) -> io::Result<OwnedFd> {
    loop {
        // SAFETY: plain syscall.
        let fd = unsafe { libc::accept4(listener, std::ptr::null_mut(), std::ptr::null_mut(), libc::SOCK_CLOEXEC) };
        if fd >= 0 {
            // SAFETY: fresh fd.
            return Ok(unsafe { OwnedFd::from_raw_fd(fd) });
        }
        let e = io::Error::last_os_error();
        if e.kind() != io::ErrorKind::Interrupted {
            return Err(e);
        }
    }
}

fn reply<T: Pod>(ctl: RawFd, req: &wire::Hdr, body: &T) -> io::Result<()> {
    sys::send_pod(ctl, req.type_, wire::F_REPLY, req.seq, body)
}

fn reply_err(ctl: RawFd, req: &wire::Hdr, code: i32) -> io::Result<()> {
    sys::send_pod(
        ctl,
        req.type_,
        wire::F_REPLY | wire::F_ERROR,
        req.seq,
        &wire::Error { code, reserved: 0 },
    )
}

fn hello(ctl: RawFd, msg: &sys::Msg) -> io::Result<&'static Backend> {
    let h = wire::Hello::read_from(&msg.payload)
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "short HELLO"))?;
    if h.version != wire::VERSION {
        reply_err(ctl, &msg.hdr, -libc::EPROTO)?;
        return Err(io::Error::other(format!("protocol version {} (want {})", h.version, wire::VERSION)));
    }
    if h.window_size[wire::WINDOW_GFX as usize] < reims_vgpu::remote::GFX_MMIO_SIZE {
        reply_err(ctl, &msg.hdr, -libc::EINVAL)?;
        return Err(io::Error::other("gfx window smaller than the device's register file"));
    }
    let nr_irqs = h.nr_irqs.clamp(1, 8);
    let (b, shared) = Backend::create(h.page_shift, nr_irqs)?;
    let behaviours = regs::behaviours();
    let rep = wire::HelloReply {
        version: wire::VERSION,
        features: 0,
        nr_behaviour: behaviours.len() as u32,
        nr_irqs,
        device_page_size: backend::DEVICE_PAGE_SIZE,
        scanout_size: backend::SCANOUT_SIZE,
        dirty_size: dirty::table_size(),
        posted_offset: backend::POSTED_OFFSET,
        reserved: 0,
    };
    let mut payload = rep.as_bytes().to_vec();
    for be in &behaviours {
        payload.extend_from_slice(be.as_bytes());
    }
    let mut fds = vec![
        shared.page_fd.as_raw_fd(),
        b.kick.as_raw_fd(),
        shared.events_peer.as_raw_fd(),
        shared.scanout_fd.as_raw_fd(),
        shared.dirty_fd.as_raw_fd(),
    ];
    fds.extend(b.irqs.iter().map(|f| f.as_raw_fd()));
    let hdr = wire::Hdr {
        type_: wire::HELLO,
        flags: wire::F_REPLY,
        size: payload.len() as u32,
        seq: msg.hdr.seq,
    };
    sys::send(ctl, &hdr, &payload, &fds, false)?;
    // The VMM holds its own references now.
    drop(shared);

    std::thread::Builder::new()
        .name("reims-vgpu-drain".into())
        .spawn(move || b.worker())?;
    std::thread::Builder::new()
        .name("reims-vgpu-poll".into())
        .spawn(move || b.poller())?;
    std::thread::Builder::new()
        .name("reims-vgpu-actions".into())
        .spawn(move || b.deliverer())?;
    b.set_running();
    let mut name = [0 as libc::c_char; 32];
    // SAFETY: valid buffer.
    unsafe { rabi::reims_vgpu_qemu_backend_name(name.as_mut_ptr(), name.len()) };
    // SAFETY: NUL-terminated by the callee.
    let name = unsafe { std::ffi::CStr::from_ptr(name.as_ptr()) }.to_string_lossy();
    eprintln!(
        "reims-vgpu-backend: device {} up (rail {name}, page shift {}, {} irq lines)",
        b.handle(),
        h.page_shift,
        nr_irqs
    );
    Ok(b)
}

fn mem_table(b: &Backend, msg: sys::Msg) -> io::Result<()> {
    let t = wire::MemTable::read_from(&msg.payload)
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "short MEM_TABLE"))?;
    let nr = t.nr as usize;
    if msg.fds.len() != nr {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "MEM_TABLE fd count"));
    }
    let base = std::mem::size_of::<wire::MemTable>();
    let rsz = std::mem::size_of::<wire::MemRegion>();
    let mut regions = Vec::with_capacity(nr);
    for (i, fd) in msg.fds.into_iter().enumerate() {
        let r = wire::MemRegion::read_from(&msg.payload[base + i * rsz..])
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "short MEM_TABLE region"))?;
        eprintln!(
            "reims-vgpu-backend: guest RAM gpa={:#x} size={:#x} fd_offset={:#x}",
            r.gpa, r.size, r.fd_offset
        );
        regions.push((r.gpa, r.size, r.fd_offset, fd));
    }
    b.ram.write().unwrap().set_table(regions)
}

/// Serve one register access a vCPU is waiting on.
fn access(b: &Backend, ctl: RawFd, msg: &sys::Msg, write: bool) -> io::Result<()> {
    let a = wire::Access::read_from(&msg.payload)
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "short access"))?;
    // The in-process handler ran under QEMU's BQL; see Backend::vmm_lock.
    let _vmm = b.vmm_lock.lock().unwrap();
    // Everything the guest posted before this access happened before it.
    b.apply_posted();
    let h = b.handle();
    CALL.with(|c| *c.borrow_mut() = Some(CallCtx::new(ctl, 0x8000_0000)));
    let mut value = 0u64;
    // SAFETY: valid handle and out pointer.
    let rc = unsafe {
        match (a.window, write) {
            (wire::WINDOW_GFX, false) => rabi::reims_vgpu_qemu_gfx_read(h, a.offset, a.size, &mut value),
            (wire::WINDOW_GFX, true) => rabi::reims_vgpu_qemu_gfx_write(h, a.offset, a.data, a.size),
            (wire::WINDOW_IOSFC, false) => rabi::reims_vgpu_qemu_iosfc_read(h, a.offset, a.size, &mut value),
            (wire::WINDOW_IOSFC, true) => rabi::reims_vgpu_qemu_iosfc_write(h, a.offset, a.data, a.size),
            _ => rabi::REIMS_VGPU_QEMU_ERR_ARGS,
        }
    };
    CALL.with(|c| *c.borrow_mut() = None);
    if write {
        // The C shim delivered actions straight after every register write.
        b.wake_actions();
    }
    if rc != rabi::REIMS_VGPU_QEMU_OK {
        return reply_err(ctl, &msg.hdr, -libc::EIO);
    }
    reply(ctl, &msg.hdr, &wire::Value { value })
}

fn serve(ctl: RawFd) -> io::Result<()> {
    let mut b: Option<&'static Backend> = None;
    loop {
        let Some(msg) = sys::recv(ctl)? else {
            eprintln!("reims-vgpu-backend: VMM closed the connection");
            return Ok(());
        };
        match (msg.hdr.type_, b) {
            (wire::HELLO, None) => b = Some(hello(ctl, &msg)?),
            (wire::MEM_TABLE, Some(dev)) => {
                let hdr = msg.hdr;
                match mem_table(dev, msg) {
                    Ok(()) => reply(ctl, &hdr, &wire::Value::default())?,
                    Err(e) => {
                        eprintln!("reims-vgpu-backend: MEM_TABLE: {e}");
                        reply_err(ctl, &hdr, -libc::EINVAL)?;
                    }
                }
            }
            (wire::READ, Some(dev)) => access(dev, ctl, &msg, false)?,
            (wire::WRITE, Some(dev)) => access(dev, ctl, &msg, true)?,
            (wire::RESET, Some(dev)) => {
                dev.reset();
                reply(ctl, &msg.hdr, &wire::Value::default())?;
            }
            (wire::SHUTDOWN, _) => {
                eprintln!("reims-vgpu-backend: shutdown requested");
                return Ok(());
            }
            (t, _) => {
                eprintln!("reims-vgpu-backend: unexpected message type {t}");
                reply_err(ctl, &msg.hdr, -libc::EPROTO)?;
            }
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut listener: Option<OwnedFd> = None;
    let mut i = 1;
    if args.get(1).map(String::as_str) == Some("--fold-breadcrumbs") {
        use reims_vgpu::config;
        let meta = config::dir_path(config::BREADCRUMB_DIR)
            .map(|d| d.join("reims-vgpu-driver-breadcrumb.txt"))
            .and_then(|p| std::fs::read_to_string(p).ok());
        if reims_vgpu::remote::fold_driver_breadcrumbs() {
            let what = meta
                .as_deref()
                .and_then(|m| m.lines().find_map(|l| l.strip_prefix("what=")))
                .unwrap_or("unknown");
            println!("driver call quarantined: {what}");
        }
        std::process::exit(0);
    }
    while i < args.len() {
        match args[i].as_str() {
            "--socket" if i + 1 < args.len() => {
                listener = Some(listen(&args[i + 1]).unwrap_or_else(|e| {
                    eprintln!("reims-vgpu-backend: {}: {e}", args[i + 1]);
                    std::process::exit(2)
                }));
                i += 2;
            }
            "--listen-fd" if i + 1 < args.len() => {
                let fd: RawFd = args[i + 1].parse().unwrap_or(-1);
                // SAFETY: the caller hands us this fd.
                listener = Some(unsafe { OwnedFd::from_raw_fd(fd) });
                i += 2;
            }
            _ => {
                eprintln!("usage: reims-vgpu-backend --socket PATH | --listen-fd N");
                std::process::exit(2);
            }
        }
    }
    let Some(listener) = listener else {
        eprintln!("usage: reims-vgpu-backend --socket PATH | --listen-fd N");
        std::process::exit(2);
    };
    let ctl = accept(listener.as_raw_fd()).unwrap_or_else(|e| {
        eprintln!("reims-vgpu-backend: accept: {e}");
        std::process::exit(1)
    });
    drop(listener);
    let rc = match serve(ctl.as_raw_fd()) {
        Ok(()) => 0,
        Err(e) => {
            eprintln!("reims-vgpu-backend: {e}");
            1
        }
    };
    if let Some(b) = backend::BACKEND_REF.get() {
        b.stopping.store(true, Ordering::Release);
    }
    // Device teardown (Vulkan objects, the VkPipelineCache flush) runs in the
    // device's own destroy path; process exit releases the rest.
    if let Some(b) = backend::BACKEND_REF.get() {
        // SAFETY: valid handle.
        unsafe { rabi::reims_vgpu_qemu_device_destroy(b.handle()) };
    }
    std::process::exit(rc);
}
