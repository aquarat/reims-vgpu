//! One device, its shared memory, and the threads that drive it.
//!
//! Threads, each the counterpart of something the in-process C shim ran in
//! QEMU:
//!  - the control thread (main.rs) serves register accesses a vCPU waits on
//!    (the shim's MMIO handlers);
//!  - the drain worker waits on the kick eventfd, applies posted writes and
//!    runs `device_drain` (the shim's `reims-vgpu-drain` thread);
//!  - the poll thread runs `device_poll` every 4 ms and keeps the heartbeat
//!    (the shim's 250 Hz poll timer);
//!  - the action thread pops HostActions and delivers them: interrupt lines
//!    through eventfds, scanout frames into the shared buffer (the shim's
//!    action BH).

use std::os::fd::{AsRawFd, OwnedFd, RawFd};
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::{Condvar, Mutex, OnceLock, RwLock};

use reims_vgpu::qemu::abi as rabi;
use reims_vgpu::runtime::host::{HostAction, HostActionKind};
use reims_vgpu_proxy_wire::{self as wire, PostedRing};

use crate::dirty::Dirty;
use crate::guest_ram::GuestRam;
use crate::sys;

pub const DEVICE_PAGE_SIZE: u64 = 64 << 10;
pub const HOT_BLOCK_OFFSET: u32 = 0x100;
pub const POSTED_OFFSET: u32 = 0x1000;

/// Scanout buffers: two of the largest surface the device accepts, plus the
/// cursor glyph. The memfd is sparse; only frames actually painted use memory.
pub const MAX_DIM: u64 = 8192;
pub const SCANOUT_BUFFERS: u64 = 2;
pub const SCANOUT_BUF_SIZE: u64 = MAX_DIM * MAX_DIM * 4;
pub const GLYPH_OFFSET: u64 = SCANOUT_BUFFERS * SCANOUT_BUF_SIZE;
pub const GLYPH_MAX_PIXELS: u64 = 512 * 512;
pub const SCANOUT_SIZE: u64 = GLYPH_OFFSET + GLYPH_MAX_PIXELS * 4;

pub struct Backend {
    pub handle: AtomicU64,
    pub page_shift: u32,
    pub ram: RwLock<GuestRam>,
    pub dirty: Dirty,
    page: *mut u8,
    scanout: *mut u8,
    pub kick: OwnedFd,
    pub irqs: Vec<OwnedFd>,
    pub events: OwnedFd,
    posted_lock: Mutex<()>,
    /// What QEMU's BQL was to the in-process shim. Its MMIO handlers, its poll
    /// timer and its action BH all ran under the BQL and so never overlapped;
    /// the drain worker ran without it. The device relies on that: a register
    /// read takes the device lock with try_lock and falls back to a cached
    /// value, which a concurrent poll would turn into a wrong answer (the
    /// guest's first version read came back 0). Held by exactly those three
    /// paths here, and by reset; never by the drain worker.
    pub vmm_lock: Mutex<()>,
    actions_pending: Mutex<bool>,
    actions_cv: Condvar,
    next_buffer: AtomicU32,
    scanout_gen: AtomicU64,
    doorbells_seen: AtomicU64,
    refused_seen: AtomicU64,
    gpa_misses: AtomicU64,
    pub stopping: AtomicBool,
}

// SAFETY: raw pointers are process-lifetime shared mappings, accessed
// atomically or through copies serialised by the device.
unsafe impl Send for Backend {}
unsafe impl Sync for Backend {}

pub static BACKEND_REF: OnceLock<&'static Backend> = OnceLock::new();

pub fn get() -> &'static Backend {
    BACKEND_REF.get().expect("backend not created")
}

pub struct Shared {
    pub page_fd: OwnedFd,
    pub scanout_fd: OwnedFd,
    pub dirty_fd: OwnedFd,
    pub events_peer: OwnedFd,
}

impl Backend {
    /// Create the shared objects and the device. Called once, on HELLO.
    pub fn create(page_shift: u32, nr_irqs: u32) -> std::io::Result<(&'static Backend, Shared)> {
        let page_fd = sys::memfd("apv-gpu-device-page", DEVICE_PAGE_SIZE)?;
        let page = sys::map_shared(page_fd.as_raw_fd(), DEVICE_PAGE_SIZE, 0)?;
        let scanout_fd = sys::memfd("apv-gpu-scanout", SCANOUT_SIZE)?;
        let scanout = sys::map_shared(scanout_fd.as_raw_fd(), SCANOUT_SIZE, 0)?;
        let dirty_fd = sys::memfd("apv-gpu-dirty", crate::dirty::table_size())?;
        let dirty_base = sys::map_shared(dirty_fd.as_raw_fd(), crate::dirty::table_size(), 0)?;
        let kick = sys::eventfd()?;
        let mut irqs = Vec::new();
        for _ in 0..nr_irqs {
            irqs.push(sys::eventfd()?);
        }
        let mut sv = [0 as RawFd; 2];
        // SAFETY: plain syscall into a 2-element array.
        if unsafe {
            libc::socketpair(
                libc::AF_UNIX,
                libc::SOCK_SEQPACKET | libc::SOCK_CLOEXEC,
                0,
                sv.as_mut_ptr(),
            )
        } != 0
        {
            return Err(sys::last_err("socketpair"));
        }
        use std::os::fd::FromRawFd;
        // SAFETY: fresh fds from socketpair.
        let (events, events_peer) =
            unsafe { (OwnedFd::from_raw_fd(sv[0]), OwnedFd::from_raw_fd(sv[1])) };

        // The hot words must be in place before the device exists: the
        // device's register state takes them at creation.
        // SAFETY: the block lies inside the device page, which is never unmapped.
        let block = unsafe {
            &*(page.add(HOT_BLOCK_OFFSET as usize) as *const reims_vgpu::remote::HotWordBlock)
        };
        if !reims_vgpu::remote::install_external_hot_words(block) {
            return Err(std::io::Error::other("hot words already installed"));
        }

        let backend: &'static Backend = Box::leak(Box::new(Backend {
            handle: AtomicU64::new(0),
            page_shift,
            ram: RwLock::new(GuestRam::default()),
            dirty: Dirty::new(dirty_base),
            page,
            scanout,
            kick,
            irqs,
            events,
            posted_lock: Mutex::new(()),
            vmm_lock: Mutex::new(()),
            actions_pending: Mutex::new(false),
            actions_cv: Condvar::new(),
            next_buffer: AtomicU32::new(0),
            scanout_gen: AtomicU64::new(0),
            doorbells_seen: AtomicU64::new(0),
            refused_seen: AtomicU64::new(0),
            gpa_misses: AtomicU64::new(0),
            stopping: AtomicBool::new(false),
        }));
        let _ = BACKEND_REF.set(backend);

        // Page header.
        backend.hdr_u32(4).store(wire::VERSION, Ordering::Relaxed);
        backend
            .hdr_u32(8)
            .store(wire::STATE_STARTING, Ordering::Relaxed);
        backend
            .hdr_u32(0)
            .store(wire::PAGE_MAGIC, Ordering::Release);

        let ops = crate::host_ops::table(backend);
        let ops: &'static _ = Box::leak(Box::new(ops));
        let info = rabi::ReimsVgpuQemuCreateInfo {
            abi_version: rabi::REIMS_VGPU_QEMU_ABI_VERSION,
            struct_size: std::mem::size_of::<rabi::ReimsVgpuQemuCreateInfo>() as u32,
            host_ops: ops,
            guest_page_shift: page_shift,
        };
        let mut out = rabi::ReimsVgpuQemuDevice {
            abi_version: 0,
            struct_size: 0,
            handle: 0,
        };
        // SAFETY: valid create info and out struct.
        let rc = unsafe { rabi::reims_vgpu_qemu_device_create(&info, &mut out) };
        if rc != rabi::REIMS_VGPU_QEMU_OK || out.handle == 0 {
            return Err(std::io::Error::other(format!("device_create rc={rc}")));
        }
        backend.handle.store(out.handle, Ordering::Release);
        Ok((
            backend,
            Shared {
                page_fd,
                scanout_fd,
                dirty_fd,
                events_peer,
            },
        ))
    }

    pub fn handle(&self) -> u64 {
        self.handle.load(Ordering::Acquire)
    }

    pub fn page_size(&self) -> u64 {
        1 << self.page_shift
    }

    fn hdr_u32(&self, off: usize) -> &AtomicU32 {
        // SAFETY: aligned offset inside the device page.
        unsafe { &*(self.page.add(off) as *const AtomicU32) }
    }

    fn hdr_u64(&self, off: usize) -> &AtomicU64 {
        // SAFETY: aligned offset inside the device page.
        unsafe { &*(self.page.add(off) as *const AtomicU64) }
    }

    pub fn set_running(&self) {
        self.hdr_u32(8)
            .store(wire::STATE_RUNNING, Ordering::Release);
    }

    fn ring(&self) -> *mut PostedRing {
        // SAFETY: inside the device page.
        unsafe { self.page.add(POSTED_OFFSET as usize) as *mut PostedRing }
    }

    pub fn kick_worker(&self) {
        sys::eventfd_signal(self.kick.as_raw_fd());
    }

    pub fn wake_actions(&self) {
        let mut p = self.actions_pending.lock().unwrap();
        *p = true;
        self.actions_cv.notify_one();
    }

    pub fn note_gpa_miss(&self, gpa: u64, len: usize) {
        let n = self.gpa_misses.fetch_add(1, Ordering::Relaxed);
        if n < 16 || n.is_power_of_two() {
            eprintln!("reims-vgpu-backend: guest access outside the RAM table gpa={gpa:#x} len={len} (#{})", n + 1);
        }
    }

    /// Apply every posted register write, in order. Called before a drain and
    /// before serving any synchronous access, so a write the guest made
    /// earlier is always applied before anything it made later.
    pub fn apply_posted(&self) {
        let _g = self.posted_lock.lock().unwrap();
        let ring = self.ring();
        // SAFETY: head/tail are naturally aligned u32 words in the shared ring.
        let head = unsafe { &*(std::ptr::addr_of!((*ring).head) as *const AtomicU32) };
        let tail = unsafe { &*(std::ptr::addr_of!((*ring).tail) as *const AtomicU32) };
        let h = self.handle();
        let mut t = tail.load(Ordering::Relaxed);
        loop {
            let hd = head.load(Ordering::Acquire);
            if t == hd {
                break;
            }
            // SAFETY: entry index within the ring; the producer published it
            // before advancing head (acquire above).
            let e = unsafe {
                std::ptr::read_volatile(&(*ring).entries[(t % wire::POSTED_ENTRIES) as usize])
            };
            // SAFETY: valid handle.
            unsafe {
                match e.window {
                    wire::WINDOW_GFX => {
                        rabi::reims_vgpu_qemu_gfx_write(h, e.offset, e.data, e.size);
                    }
                    wire::WINDOW_IOSFC => {
                        rabi::reims_vgpu_qemu_iosfc_write(h, e.offset, e.data, e.size);
                    }
                    _ => {}
                }
            }
            t = t.wrapping_add(1);
            tail.store(t, Ordering::Release);
        }
    }

    /// Report doorbells the VMM took on the device's behalf to the census.
    fn account_doorbells(&self) {
        let rung = self.hdr_u64(24).load(Ordering::Relaxed);
        let seen = self.doorbells_seen.swap(rung, Ordering::Relaxed);
        if rung > seen {
            reims_vgpu::remote::note_doorbells_lock_free(rung - seen);
        }
        let refused = self.hdr_u64(32).load(Ordering::Relaxed);
        let rseen = self.refused_seen.swap(refused, Ordering::Relaxed);
        if refused > rseen {
            reims_vgpu::remote::note_out_of_range_doorbell(
                self.hdr_u32(40).load(Ordering::Relaxed),
            );
        }
    }

    pub fn worker(&'static self) {
        while sys::eventfd_wait(self.kick.as_raw_fd()) {
            if self.stopping.load(Ordering::Acquire) {
                break;
            }
            self.apply_posted();
            self.account_doorbells();
            // SAFETY: valid handle.
            let rc = unsafe { rabi::reims_vgpu_qemu_device_drain(self.handle()) };
            if rc != rabi::REIMS_VGPU_QEMU_OK {
                eprintln!("reims-vgpu-backend: drain failed rc={rc}");
            }
            self.wake_actions();
        }
    }

    pub fn poller(&'static self) {
        loop {
            if self.stopping.load(Ordering::Acquire) {
                break;
            }
            self.hdr_u64(16)
                .store(sys::now_ns(libc::CLOCK_MONOTONIC), Ordering::Release);
            let polled = {
                let _vmm = self.vmm_lock.lock().unwrap();
                // SAFETY: valid handle.
                unsafe { rabi::reims_vgpu_qemu_device_poll(self.handle()) }
            };
            if polled == rabi::REIMS_VGPU_QEMU_OK {
                self.wake_actions();
            }
            std::thread::sleep(std::time::Duration::from_millis(4));
        }
    }

    pub fn deliverer(&'static self) {
        loop {
            {
                let mut p = self.actions_pending.lock().unwrap();
                while !*p {
                    p = self.actions_cv.wait(p).unwrap();
                }
                *p = false;
            }
            if self.stopping.load(Ordering::Acquire) {
                break;
            }
            let _vmm = self.vmm_lock.lock().unwrap();
            self.deliver_actions();
        }
    }

    pub fn deliver_actions(&self) {
        let h = self.handle();
        loop {
            let mut a = HostAction::default();
            // SAFETY: valid handle and out pointer.
            if unsafe { rabi::reims_vgpu_qemu_device_pop_action(h, &mut a) }
                != rabi::REIMS_VGPU_QEMU_OK
            {
                break;
            }
            self.apply_action(&a);
        }
    }

    fn apply_action(&self, a: &HostAction) {
        match a.kind {
            HostActionKind::IrqGfxPulse => self.raise(0),
            HostActionKind::IrqIosfcPulse => self.raise(1),
            HostActionKind::ScanoutUpdate => {
                self.scanout(a.a0 as u32, a.a1 as u32, a.a2 as u32, a.a3 as u32)
            }
            HostActionKind::CursorUpdate => {
                let ev = wire::Cursor {
                    x: a.a0 as i32,
                    y: a.a1 as i32,
                    visible: u32::from(a.a2 != 0),
                    reserved: 0,
                };
                self.event(wire::EV_CURSOR, &ev);
            }
            HostActionKind::CursorGlyph => self.cursor_glyph(),
            // No host window in this process (yet): its input and close
            // actions have nothing to come from.
            _ => {}
        }
    }

    fn raise(&self, line: usize) {
        if let Some(fd) = self.irqs.get(line) {
            sys::eventfd_signal(fd.as_raw_fd());
        }
    }

    fn event<T: wire::Pod>(&self, type_: u32, body: &T) {
        let hdr = wire::Hdr {
            type_,
            flags: 0,
            size: std::mem::size_of::<T>() as u32,
            seq: 0,
        };
        // Never block the action thread on a VMM that is not reading: a frame
        // event that does not fit is dropped and the next one supersedes it.
        let _ = sys::send(self.events.as_raw_fd(), &hdr, body.as_bytes(), &[], true);
    }

    /// The C shim's apply_scanout: ask whether this present may paint, copy it
    /// into a host buffer (which is also what acknowledges the present when
    /// there is no host window), and show it.
    fn scanout(&self, mapping: u32, width: u32, height: u32, generation: u32) {
        let h = self.handle();
        let mut may = 0u32;
        // SAFETY: valid handle and out pointer.
        let rc = unsafe { rabi::reims_vgpu_qemu_scanout_may_paint(h, mapping, &mut may) };
        if rc != rabi::REIMS_VGPU_QEMU_OK || may == 0 {
            return;
        }
        if width == 0 || height == 0 || u64::from(width) > MAX_DIM || u64::from(height) > MAX_DIM {
            return;
        }
        let buffer = self.next_buffer.fetch_add(1, Ordering::Relaxed) % SCANOUT_BUFFERS as u32;
        let offset = u64::from(buffer) * SCANOUT_BUF_SIZE;
        let stride = width * 4;
        // SAFETY: the buffer lies inside the scanout mapping and holds
        // MAX_DIM^2 pixels.
        let rc = unsafe {
            rabi::reims_vgpu_qemu_scanout_copy(
                h,
                mapping,
                self.scanout.add(offset as usize),
                stride,
                width,
                height,
                generation,
            )
        };
        if rc != rabi::REIMS_VGPU_QEMU_OK {
            // EMPTY: unchanged, the frame already shown stands.
            if rc != rabi::REIMS_VGPU_QEMU_EMPTY {
                eprintln!(
                    "reims-vgpu-backend: scanout_copy mapping={mapping} {width}x{height} rc={rc}"
                );
            }
            return;
        }
        let ev = wire::Scanout {
            buffer,
            width,
            height,
            stride,
            offset,
            generation: self.scanout_gen.fetch_add(1, Ordering::Relaxed) + 1,
        };
        self.event(wire::EV_SCANOUT, &ev);
    }

    fn cursor_glyph(&self) {
        let h = self.handle();
        let mut info = GlyphInfo::default();
        // SAFETY: valid handle; GlyphInfo is the layout twin of the ABI's
        // CursorGlyphInfo (whose type this crate cannot name).
        if unsafe {
            rabi::reims_vgpu_qemu_cursor_glyph_info(h, (&mut info as *mut GlyphInfo).cast())
        } != rabi::REIMS_VGPU_QEMU_OK
        {
            return;
        }
        let n = u64::from(info.pixel_count);
        if info.width == 0
            || info.height == 0
            || n != u64::from(info.width) * u64::from(info.height)
            || n > GLYPH_MAX_PIXELS
        {
            return;
        }
        // SAFETY: the glyph area holds GLYPH_MAX_PIXELS u32s.
        let rc = unsafe {
            rabi::reims_vgpu_qemu_cursor_glyph_copy(
                h,
                self.scanout.add(GLYPH_OFFSET as usize) as *mut u32,
                n as usize,
            )
        };
        if rc != rabi::REIMS_VGPU_QEMU_OK {
            return;
        }
        let ev = wire::CursorGlyph {
            width: info.width,
            height: info.height,
            hot_x: info.hot_x,
            hot_y: info.hot_y,
            offset: GLYPH_OFFSET,
        };
        self.event(wire::EV_CURSOR_GLYPH, &ev);
    }

    /// Device reset (the VMM's reset handler waits for this).
    pub fn reset(&self) {
        let _vmm = self.vmm_lock.lock().unwrap();
        {
            let _g = self.posted_lock.lock().unwrap();
            let ring = self.ring();
            // SAFETY: head/tail words in the shared ring.
            let head = unsafe { &*(std::ptr::addr_of!((*ring).head) as *const AtomicU32) };
            let tail = unsafe { &*(std::ptr::addr_of!((*ring).tail) as *const AtomicU32) };
            tail.store(head.load(Ordering::Acquire), Ordering::Release);
        }
        // SAFETY: valid handle.
        unsafe { rabi::reims_vgpu_qemu_device_reset(self.handle()) };
        let views = self.ram.read().unwrap().unmap_all();
        if views != 0 {
            eprintln!("reims-vgpu-backend: {views} guest page view(s) still mapped at reset");
        }
        self.dirty.release_all();
    }
}

/// `ReimsVgpuCursorGlyphInfo` in the ABI header.
#[repr(C)]
#[derive(Default)]
struct GlyphInfo {
    width: u32,
    height: u32,
    hot_x: u32,
    hot_y: u32,
    pixel_count: u32,
}
