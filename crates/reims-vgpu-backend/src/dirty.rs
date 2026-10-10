//! Guest-write tracking over the shared dirty table.
//!
//! The device registers page sets and later asks whether, and which of, their
//! pages the guest CPU stored into. Only the hypervisor's dirty log can see
//! such stores, and only the VMM can read it, so the VMM folds the log into
//! this table when it harvests (before handing the device a guest write) and
//! this process reads generations straight out of it: registering a set is a
//! few stores, and every question afterwards is answered without a message.
//!
//! The contract is the one `ReimsVgpuHostOps` documents for the in-process
//! shim: token 0 means "cannot track"; a generation of 0 means "cannot tell
//! yet"; `written_since` answers -1 whenever the answer is not knowable.

use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::Mutex;

use reims_vgpu_proxy_wire::{self as wire, DirtyHeader, DirtySlot};

pub const NR_SLOTS: u32 = 8192;
pub const PAGE_CAPACITY: u32 = 512 * 1024;

pub fn table_size() -> u64 {
    let slots = std::mem::size_of::<DirtyHeader>() as u64
        + NR_SLOTS as u64 * std::mem::size_of::<DirtySlot>() as u64;
    let slots = (slots + 4095) & !4095;
    slots + 2 * PAGE_CAPACITY as u64 * 8
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Own {
    Free,
    Live { incarnation: u32 },
    Releasing,
}

struct Alloc {
    own: Vec<Own>,
    ranges: Vec<(u32, u32)>, // per slot: (first_page, count)
    free: Vec<(u32, u32)>,   // free page ranges (first, count), unsorted
    next_incarnation: u32,
}

pub struct Dirty {
    base: *mut u8,
    alloc: Mutex<Alloc>,
}

// SAFETY: base is a process-lifetime shared mapping accessed atomically.
unsafe impl Send for Dirty {}
unsafe impl Sync for Dirty {}

impl Dirty {
    /// Initialise a fresh (zeroed) table at `base`.
    pub fn new(base: *mut u8) -> Self {
        let slots_offset = std::mem::size_of::<DirtyHeader>() as u32;
        let pages_offset = ((slots_offset as u64
            + NR_SLOTS as u64 * std::mem::size_of::<DirtySlot>() as u64
            + 4095)
            & !4095) as u32;
        let page_gen_offset = pages_offset + PAGE_CAPACITY * 8;
        // SAFETY: base points at a zeroed mapping of table_size() bytes.
        let hdr = unsafe { &mut *(base as *mut DirtyHeader) };
        hdr.version = wire::VERSION;
        hdr.nr_slots = NR_SLOTS;
        hdr.page_capacity = PAGE_CAPACITY;
        hdr.slots_offset = slots_offset;
        hdr.pages_offset = pages_offset;
        hdr.page_gen_offset = page_gen_offset;
        std::sync::atomic::fence(Ordering::Release);
        hdr.magic = wire::DIRTY_MAGIC;
        Self {
            base,
            alloc: Mutex::new(Alloc {
                own: vec![Own::Free; NR_SLOTS as usize],
                ranges: vec![(0, 0); NR_SLOTS as usize],
                free: vec![(0, PAGE_CAPACITY)],
                next_incarnation: 1,
            }),
        }
    }

    fn hdr(&self) -> &DirtyHeader {
        // SAFETY: initialised in new().
        unsafe { &*(self.base as *const DirtyHeader) }
    }

    fn u32_at(&self, off: usize) -> &AtomicU32 {
        // SAFETY: aligned offset inside the table.
        unsafe { &*(self.base.add(off) as *const AtomicU32) }
    }

    fn u64_at(&self, off: usize) -> &AtomicU64 {
        // SAFETY: aligned offset inside the table.
        unsafe { &*(self.base.add(off) as *const AtomicU64) }
    }

    fn slot_off(&self, i: u32) -> usize {
        self.hdr().slots_offset as usize + i as usize * std::mem::size_of::<DirtySlot>()
    }

    fn state(&self, i: u32) -> &AtomicU32 {
        self.u32_at(self.slot_off(i))
    }

    fn set_gen(&self, i: u32) -> &AtomicU64 {
        self.u64_at(self.slot_off(i) + 24)
    }

    fn page_gpa(&self, page: u32) -> &AtomicU64 {
        self.u64_at(self.hdr().pages_offset as usize + page as usize * 8)
    }

    fn page_gen(&self, page: u32) -> &AtomicU64 {
        self.u64_at(self.hdr().page_gen_offset as usize + page as usize * 8)
    }

    fn note_read(&self) {
        // DirtyHeader.gen_reads
        self.u64_at(24).fetch_add(1, Ordering::Release);
    }

    fn harvests(&self) -> u64 {
        // DirtyHeader.harvests
        self.u64_at(16).load(Ordering::Acquire)
    }

    fn token(slot: u32, incarnation: u32) -> u64 {
        (u64::from(incarnation) << 32) | u64::from(slot + 1)
    }

    fn live_slot(alloc: &Alloc, token: u64) -> Option<u32> {
        let slot = (token & 0xffff_ffff) as u32;
        let inc = (token >> 32) as u32;
        if slot == 0 || slot > NR_SLOTS {
            return None;
        }
        match alloc.own[(slot - 1) as usize] {
            Own::Live { incarnation } if incarnation == inc => Some(slot - 1),
            _ => None,
        }
    }

    /// Return slots the VMM has finished releasing to the free pool.
    fn reclaim(&self, a: &mut Alloc) {
        for i in 0..NR_SLOTS {
            if a.own[i as usize] == Own::Releasing && self.state(i).load(Ordering::Acquire) == wire::DIRTY_FREE {
                a.own[i as usize] = Own::Free;
                let r = a.ranges[i as usize];
                if r.1 != 0 {
                    a.free.push(r);
                }
                a.ranges[i as usize] = (0, 0);
            }
        }
    }

    fn take_range(free: &mut Vec<(u32, u32)>, count: u32) -> Option<u32> {
        let i = free.iter().position(|&(_, n)| n >= count)?;
        let (first, n) = free[i];
        if n == count {
            free.swap_remove(i);
        } else {
            free[i] = (first + count, n - count);
        }
        Some(first)
    }

    pub fn track(&self, gpas: &[u64], page_size: u64) -> u64 {
        if gpas.is_empty() || page_size < 4096 || !page_size.is_power_of_two() {
            return 0;
        }
        let mut pages: Vec<u64> = gpas.iter().map(|g| g & !(page_size - 1)).collect();
        pages.sort_unstable();
        pages.dedup();
        let count = pages.len() as u32;
        let mut a = self.alloc.lock().unwrap();
        let mut slot = a.own.iter().position(|o| *o == Own::Free);
        let mut first = Self::take_range(&mut a.free, count);
        if slot.is_none() || first.is_none() {
            if let Some(f) = first.take() {
                a.free.push((f, count));
            }
            self.reclaim(&mut a);
            slot = a.own.iter().position(|o| *o == Own::Free);
            first = Self::take_range(&mut a.free, count);
        }
        let (Some(slot), Some(first)) = (slot, first) else {
            if let Some(f) = first {
                a.free.push((f, count));
            }
            return 0;
        };
        let slot = slot as u32;
        for (k, &gpa) in pages.iter().enumerate() {
            self.page_gpa(first + k as u32).store(gpa, Ordering::Relaxed);
            self.page_gen(first + k as u32).store(0, Ordering::Relaxed);
        }
        let off = self.slot_off(slot);
        self.u32_at(off + 4).store(page_size as u32, Ordering::Relaxed);
        self.u32_at(off + 8).store(first, Ordering::Relaxed);
        self.u32_at(off + 12).store(count, Ordering::Relaxed);
        self.u64_at(off + 16).store(self.harvests() + 1, Ordering::Relaxed);
        self.set_gen(slot).store(0, Ordering::Relaxed);
        self.state(slot).store(wire::DIRTY_PENDING, Ordering::Release);
        // DirtyHeader.slots_used: the VMM scans slots below it.
        self.u32_at(44).fetch_max(slot + 1, Ordering::Release);
        let incarnation = a.next_incarnation;
        a.next_incarnation = a.next_incarnation.wrapping_add(1).max(1);
        a.own[slot as usize] = Own::Live { incarnation };
        a.ranges[slot as usize] = (first, count);
        Self::token(slot, incarnation)
    }

    pub fn untrack(&self, token: u64) {
        let mut a = self.alloc.lock().unwrap();
        if let Some(slot) = Self::live_slot(&a, token) {
            self.state(slot).store(wire::DIRTY_RELEASING, Ordering::Release);
            a.own[slot as usize] = Own::Releasing;
        }
    }

    pub fn gen(&self, token: u64) -> u64 {
        let slot = {
            let a = self.alloc.lock().unwrap();
            Self::live_slot(&a, token)
        };
        self.note_read();
        match slot {
            Some(s) => self.set_gen(s).load(Ordering::Acquire),
            None => 0,
        }
    }

    pub fn written_since(&self, token: u64, since: u64, out: &mut [u64]) -> i64 {
        let a = self.alloc.lock().unwrap();
        self.note_read();
        let Some(slot) = Self::live_slot(&a, token) else { return -1 };
        if since == 0 || self.set_gen(slot).load(Ordering::Acquire) == 0 {
            return -1;
        }
        let (first, count) = a.ranges[slot as usize];
        let mut found = 0usize;
        for p in first..first + count {
            if self.page_gen(p).load(Ordering::Acquire) <= since {
                continue;
            }
            if found == out.len() {
                return -1;
            }
            out[found] = self.page_gpa(p).load(Ordering::Relaxed);
            found += 1;
        }
        found as i64
    }

    /// Drop every set (device reset): they name mappings that no longer exist.
    pub fn release_all(&self) {
        let mut a = self.alloc.lock().unwrap();
        for i in 0..NR_SLOTS {
            if let Own::Live { .. } = a.own[i as usize] {
                self.state(i).store(wire::DIRTY_RELEASING, Ordering::Release);
                a.own[i as usize] = Own::Releasing;
            }
        }
    }
}
