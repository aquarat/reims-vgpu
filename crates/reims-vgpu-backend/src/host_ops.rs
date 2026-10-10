//! The device's host services (`ReimsVgpuHostOps`), answered in this process.
//!
//! In process, QEMU's C shim fills this table. Here every entry is answered
//! from this process's mapping of guest RAM and the shared page, except the two
//! that need the guest CPU (`read_xreg`, `read_kva`): those become callbacks to
//! the VMM, and are answerable only while a vCPU is waiting on a register
//! access this process is serving — exactly the condition the C shim enforces
//! with `current_cpu`.

use std::cell::RefCell;
use std::ffi::c_void;
use std::os::fd::RawFd;

use reims_vgpu::qemu::host_ops::{ReimsVgpuHostOps, ReimsVgpuMapPagesFailure};
use reims_vgpu::runtime::guest_ram::GuestRamRegion;
use reims_vgpu_proxy_wire::{self as wire, Pod};

use crate::backend::Backend;
use crate::sys;

/// The register access a vCPU is blocked on, while this thread serves it.
pub struct CallCtx {
    pub ctl: RawFd,
    pub seq: u32,
    xregs: Option<Option<[u64; 31]>>,
    /// (va block base, pa block base, block size)
    xlate: Vec<(u64, u64, u64)>,
}

impl CallCtx {
    pub fn new(ctl: RawFd, seq: u32) -> Self {
        Self {
            ctl,
            seq,
            xregs: None,
            xlate: Vec::new(),
        }
    }

    /// Send one callback and wait for its reply. The VMM sends nothing else on
    /// this socket while a vCPU waits on us, so the next message is the reply.
    fn call<T: Pod, R: Pod>(&mut self, type_: u32, req: &T) -> Option<R> {
        self.seq = self.seq.wrapping_add(1);
        sys::send_pod(self.ctl, type_, 0, self.seq, req).ok()?;
        let msg = sys::recv(self.ctl).ok()??;
        if msg.hdr.type_ != type_ || msg.hdr.flags & wire::F_REPLY == 0 {
            return None;
        }
        R::read_from(&msg.payload)
    }

    fn xregs(&mut self) -> Option<[u64; 31]> {
        if self.xregs.is_none() {
            let r: Option<wire::Xregs> = self.call(wire::CB_XREGS, &wire::Value::default());
            self.xregs = Some(r.filter(|r| r.ok != 0).map(|r| r.x));
        }
        self.xregs.unwrap()
    }

    fn translate(&mut self, va: u64) -> Option<(u64, u64)> {
        if let Some(&(v, p, size)) = self.xlate.iter().find(|&&(v, _, s)| va >= v && va - v < s) {
            return Some((p + (va - v), size - (va - v)));
        }
        let r: wire::XlateReply = self.call(wire::CB_XLATE, &wire::Xlate { va })?;
        if r.ok == 0 {
            return None;
        }
        // Cache at most a 16 KiB block: larger block mappings are rare for
        // kernel data and the cache is per access anyway.
        let lg = r.lg_page_size.clamp(12, 14);
        let size = 1u64 << lg;
        let vbase = va & !(size - 1);
        let pbase = r.pa & !(size - 1);
        self.xlate.push((vbase, pbase, size));
        Some((pbase + (va - vbase), size - (va - vbase)))
    }
}

thread_local! {
    pub static CALL: RefCell<Option<CallCtx>> = const { RefCell::new(None) };
}

fn be(ctx: *mut c_void) -> &'static Backend {
    // SAFETY: ctx is the leaked &'static Backend given at device create.
    unsafe { &*(ctx as *const Backend) }
}

unsafe extern "C" fn read_gpa(ctx: *mut c_void, gpa: u64, buf: *mut u8, len: usize) -> i32 {
    if len == 0 {
        return 0;
    }
    // SAFETY: caller passes a valid buffer of len bytes.
    let buf = unsafe { std::slice::from_raw_parts_mut(buf, len) };
    if be(ctx).ram.read().unwrap().read(gpa, buf) {
        0
    } else {
        be(ctx).note_gpa_miss(gpa, len);
        -1
    }
}

unsafe extern "C" fn write_gpa(ctx: *mut c_void, gpa: u64, buf: *const u8, len: usize) -> i32 {
    if len == 0 {
        return 0;
    }
    // SAFETY: caller passes a valid buffer of len bytes.
    let buf = unsafe { std::slice::from_raw_parts(buf, len) };
    if be(ctx).ram.read().unwrap().write(gpa, buf) {
        0
    } else {
        be(ctx).note_gpa_miss(gpa, len);
        -1
    }
}

unsafe extern "C" fn mono_ns(_ctx: *mut c_void) -> u64 {
    // QEMU_CLOCK_HOST, which the in-process shim answered with, is real time.
    sys::now_ns(libc::CLOCK_REALTIME)
}

unsafe extern "C" fn schedule_bh(ctx: *mut c_void) {
    be(ctx).kick_worker();
}

unsafe extern "C" fn notify_actions(ctx: *mut c_void) {
    be(ctx).wake_actions();
}

unsafe extern "C" fn read_kva(_ctx: *mut c_void, kva: u64, buf: *mut u8, len: usize) -> i32 {
    if len == 0 {
        return 0;
    }
    CALL.with(|c| {
        let mut c = c.borrow_mut();
        let Some(call) = c.as_mut() else {
            // No vCPU is waiting on this thread: the in-process shim's -2.
            return -2;
        };
        let backend = crate::backend::get();
        let ram = backend.ram.read().unwrap();
        let mut done = 0usize;
        while done < len {
            let Some((pa, avail)) = call.translate(kva + done as u64) else { return -1 };
            let n = (avail as usize).min(len - done);
            // SAFETY: caller passes a valid buffer of len bytes.
            let dst = unsafe { std::slice::from_raw_parts_mut(buf.add(done), n) };
            if !ram.read(pa, dst) {
                return -1;
            }
            done += n;
        }
        0
    })
}

unsafe extern "C" fn read_xreg(_ctx: *mut c_void, index: u32, out: *mut u64) -> i32 {
    CALL.with(|c| {
        let mut c = c.borrow_mut();
        let Some(call) = c.as_mut() else { return -1 };
        match call.xregs() {
            Some(x) if (index as usize) < x.len() => {
                // SAFETY: caller passes a valid out pointer.
                unsafe { *out = x[index as usize] };
                0
            }
            _ => -1,
        }
    })
}

/// Layout twin of `ReimsVgpuMapPagesFailure`, whose fields are crate-private.
#[repr(C)]
struct MapPagesFailure {
    stage: u32,
    host_errno: i32,
    page_index: u64,
}

unsafe extern "C" fn map_pages(
    ctx: *mut c_void,
    gpas: *const u64,
    count: usize,
    out_ptr: *mut *mut c_void,
    failure: *mut ReimsVgpuMapPagesFailure,
) -> i32 {
    let failure = failure as *mut MapPagesFailure;
    let set = |stage: u32, page_index: u64, host_errno: i32| {
        if !failure.is_null() {
            // SAFETY: caller passes a valid failure record.
            unsafe {
                *failure = MapPagesFailure {
                    stage,
                    host_errno,
                    page_index,
                }
            };
        }
    };
    set(0, 0, 0);
    if gpas.is_null() || count == 0 || out_ptr.is_null() {
        set(3, u64::MAX, 0);
        return -1;
    }
    // SAFETY: caller passes count GPAs.
    let gpas = unsafe { std::slice::from_raw_parts(gpas, count) };
    let b = be(ctx);
    match b.ram.read().unwrap().map_pages(gpas, b.page_size()) {
        Ok(p) => {
            // SAFETY: valid out pointer.
            unsafe { *out_ptr = p as *mut c_void };
            0
        }
        Err((stage, page, errno)) => {
            set(stage, page, errno);
            -1
        }
    }
}

unsafe extern "C" fn unmap_pages(ctx: *mut c_void, ptr: *mut c_void, _len: usize) {
    be(ctx).ram.read().unwrap().unmap_pages(ptr as *mut u8);
}

unsafe extern "C" fn is_ram_gpa(ctx: *mut c_void, gpa: u64) -> i32 {
    i32::from(be(ctx).ram.read().unwrap().is_ram(gpa))
}

unsafe extern "C" fn guest_ram_regions(ctx: *mut c_void, out: *mut GuestRamRegion, max: usize) -> i32 {
    const ERR_ARGS: i32 = -1;
    const ERR_NO_RAM: i32 = -2;
    if out.is_null() || max == 0 {
        return ERR_ARGS;
    }
    let ram = be(ctx).ram.read().unwrap();
    let regions = ram.regions();
    if regions.is_empty() {
        return ERR_NO_RAM;
    }
    for (i, r) in regions.iter().take(max).enumerate() {
        // SAFETY: caller passes max entries.
        unsafe {
            *out.add(i) = GuestRamRegion {
                gpa_base: r.gpa,
                host_va: r.hva as u64,
                len: r.size,
            }
        };
    }
    regions.len() as i32
}

unsafe extern "C" fn track_guest_writes(ctx: *mut c_void, gpas: *const u64, count: usize, page_size: usize) -> u64 {
    if gpas.is_null() || count == 0 {
        return 0;
    }
    // SAFETY: caller passes count GPAs.
    let gpas = unsafe { std::slice::from_raw_parts(gpas, count) };
    be(ctx).dirty.track(gpas, page_size as u64)
}

unsafe extern "C" fn untrack_guest_writes(ctx: *mut c_void, token: u64) {
    be(ctx).dirty.untrack(token);
}

unsafe extern "C" fn guest_write_gen(ctx: *mut c_void, token: u64) -> u64 {
    be(ctx).dirty.gen(token)
}

unsafe extern "C" fn guest_written_pages(
    ctx: *mut c_void,
    token: u64,
    since_gen: u64,
    out: *mut u64,
    max: usize,
) -> i64 {
    if out.is_null() {
        return -1;
    }
    // SAFETY: caller passes max entries.
    let out = unsafe { std::slice::from_raw_parts_mut(out, max) };
    be(ctx).dirty.written_since(token, since_gen, out)
}

pub fn table(backend: &'static Backend) -> ReimsVgpuHostOps {
    use reims_vgpu::qemu::abi::REIMS_VGPU_QEMU_ABI_VERSION;
    ReimsVgpuHostOps {
        abi_version: REIMS_VGPU_QEMU_ABI_VERSION,
        struct_size: std::mem::size_of::<ReimsVgpuHostOps>() as u32,
        ctx: backend as *const Backend as *mut c_void,
        read_gpa: Some(read_gpa),
        write_gpa: Some(write_gpa),
        mono_ns: Some(mono_ns),
        schedule_bh: Some(schedule_bh),
        read_kva: Some(read_kva),
        read_xreg: Some(read_xreg),
        map_pages: Some(map_pages),
        unmap_pages: Some(unmap_pages),
        is_ram_gpa: Some(is_ram_gpa),
        guest_ram_regions: Some(guest_ram_regions),
        notify_actions: Some(notify_actions),
        // Packed views are transient here exactly as on the in-process arm64
        // shim; the GPU imports the guest_ram_regions spans instead.
        map_pages_stable: 0,
        track_guest_writes: Some(track_guest_writes),
        untrack_guest_writes: Some(untrack_guest_writes),
        guest_write_gen: Some(guest_write_gen),
        guest_written_pages: Some(guest_written_pages),
        page_alias_census: None,
    }
}
