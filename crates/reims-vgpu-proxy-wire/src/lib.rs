// SPDX-License-Identifier: MIT
//! The apv-gpu-proxy protocol, as Rust types.
//!
//! `include/apv-gpu-proxy-protocol.h` is the specification and the C side of
//! the interface; this crate mirrors it field for field. The VMM includes the
//! header and never this crate; the backend uses this crate and never the
//! header. `tests` keeps the two in step: every `#define` in the header must
//! equal the constant of the same name here, and every struct size the header
//! asserts is asserted here too.
#![no_std]

use core::mem::size_of;

pub const VERSION: u32 = 1;
pub const MAX_PAYLOAD: u32 = 8192;
pub const MAX_FDS: u32 = 16;

pub const F_REPLY: u32 = 0x1;
pub const F_ERROR: u32 = 0x2;

pub const HELLO: u32 = 1;
pub const MEM_TABLE: u32 = 2;
pub const READ: u32 = 3;
pub const WRITE: u32 = 4;
pub const RESET: u32 = 5;
pub const SHUTDOWN: u32 = 6;
pub const CB_XREGS: u32 = 16;
pub const CB_XLATE: u32 = 17;
pub const EV_SCANOUT: u32 = 32;
pub const EV_CURSOR: u32 = 33;
pub const EV_CURSOR_GLYPH: u32 = 34;
pub const EV_FATAL: u32 = 35;

pub const WINDOW_GFX: u32 = 0;
pub const WINDOW_IOSFC: u32 = 1;
pub const MAX_WINDOWS: u32 = 4;

pub const R_SYNC: u8 = 0;
pub const R_XCHG0: u8 = 1;
pub const R_LOAD: u8 = 2;
pub const W_POSTED: u8 = 0;
pub const W_SYNC: u8 = 1;
pub const W_ANDNOT: u8 = 2;
pub const W_BITSET: u8 = 3;
pub const B_HARVEST: u8 = 0x1;
pub const B_SIZE4: u8 = 0x2;
pub const OFFSET_DEFAULT: u32 = 0xffff_ffff;

pub const FD_DEVICE_PAGE: u32 = 0;
pub const FD_KICK: u32 = 1;
pub const FD_EVENTS: u32 = 2;
pub const FD_SCANOUT: u32 = 3;
pub const FD_DIRTY: u32 = 4;
pub const FD_IRQ0: u32 = 5;

pub const PAGE_MAGIC: u32 = 0x5056_5041;
pub const STATE_STARTING: u32 = 0;
pub const STATE_RUNNING: u32 = 1;
pub const STATE_FAILED: u32 = 2;
pub const POSTED_ENTRIES: u32 = 256;

pub const DIRTY_MAGIC: u32 = 0x5954_5244;
pub const DIRTY_FREE: u32 = 0;
pub const DIRTY_PENDING: u32 = 1;
pub const DIRTY_ARMED: u32 = 2;
pub const DIRTY_RELEASING: u32 = 3;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Hdr {
    pub type_: u32,
    pub flags: u32,
    pub size: u32,
    pub seq: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct Error {
    pub code: i32,
    pub reserved: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct Hello {
    pub version: u32,
    pub page_shift: u32,
    pub nr_windows: u32,
    pub nr_irqs: u32,
    pub window_size: [u64; MAX_WINDOWS as usize],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Behaviour {
    pub window: u8,
    pub read_kind: u8,
    pub write_kind: u8,
    pub flags: u8,
    pub offset: u32,
    pub word: u32,
    pub arg_lo: u16,
    pub arg_hi: u16,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct HelloReply {
    pub version: u32,
    pub features: u32,
    pub nr_behaviour: u32,
    pub nr_irqs: u32,
    pub device_page_size: u64,
    pub scanout_size: u64,
    pub dirty_size: u64,
    pub posted_offset: u32,
    pub reserved: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct MemRegion {
    pub gpa: u64,
    pub size: u64,
    pub fd_offset: u64,
    pub reserved: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct MemTable {
    pub nr: u32,
    pub generation: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct Access {
    pub window: u32,
    pub size: u32,
    pub offset: u64,
    pub data: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct Value {
    pub value: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct Xregs {
    pub ok: u32,
    pub nr: u32,
    pub x: [u64; 31],
}

impl Default for Xregs {
    fn default() -> Self {
        Self {
            ok: 0,
            nr: 0,
            x: [0; 31],
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct Xlate {
    pub va: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct XlateReply {
    pub ok: u32,
    pub lg_page_size: u32,
    pub pa: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct Scanout {
    pub buffer: u32,
    pub width: u32,
    pub height: u32,
    pub stride: u32,
    pub offset: u64,
    pub generation: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct Cursor {
    pub x: i32,
    pub y: i32,
    pub visible: u32,
    pub reserved: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct CursorGlyph {
    pub width: u32,
    pub height: u32,
    pub hot_x: u32,
    pub hot_y: u32,
    pub offset: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct Fatal {
    pub code: u32,
    pub len: u32,
}

#[repr(C)]
#[derive(Debug, Default)]
pub struct PageHeader {
    pub magic: u32,
    pub version: u32,
    pub state: u32,
    pub reserved0: u32,
    pub heartbeat_ns: u64,
    pub doorbells: u64,
    pub doorbells_refused: u64,
    pub last_refused: u32,
    pub reserved1: u32,
    pub posted_sync_fallbacks: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct PostedEntry {
    pub window: u32,
    pub size: u32,
    pub offset: u64,
    pub data: u64,
    pub seq: u64,
}

#[repr(C)]
pub struct PostedRing {
    pub head: u32,
    pub pad0: [u32; 15],
    pub tail: u32,
    pub pad1: [u32; 15],
    pub entries: [PostedEntry; POSTED_ENTRIES as usize],
}

#[repr(C)]
#[derive(Debug, Default)]
pub struct DirtyHeader {
    pub magic: u32,
    pub version: u32,
    pub nr_slots: u32,
    pub page_capacity: u32,
    pub harvests: u64,
    pub gen_reads: u64,
    pub slots_offset: u32,
    pub pages_offset: u32,
    pub page_gen_offset: u32,
    pub slots_used: u32,
}

#[repr(C)]
#[derive(Debug, Default)]
pub struct DirtySlot {
    pub state: u32,
    pub page_size: u32,
    pub first_page: u32,
    pub count: u32,
    pub arm_harvest: u64,
    pub set_gen: u64,
}

const _: () = {
    assert!(size_of::<Hdr>() == 16);
    assert!(size_of::<Hello>() == 48);
    assert!(size_of::<Behaviour>() == 16);
    assert!(size_of::<HelloReply>() == 48);
    assert!(size_of::<MemRegion>() == 32);
    assert!(size_of::<MemTable>() == 8);
    assert!(size_of::<Access>() == 24);
    assert!(size_of::<Xregs>() == 256);
    assert!(size_of::<XlateReply>() == 16);
    assert!(size_of::<Scanout>() == 32);
    assert!(size_of::<CursorGlyph>() == 24);
    assert!(size_of::<PageHeader>() == 56);
    assert!(size_of::<PostedEntry>() == 32);
    assert!(size_of::<PostedRing>() == 128 + 256 * 32);
    assert!(size_of::<DirtyHeader>() == 48);
    assert!(size_of::<DirtySlot>() == 32);
};

/// Plain-old-data: every bit pattern is a valid value and there is no padding
/// the other side could leave uninitialised. Implemented only for the message
/// structs above, which are all-integer `#[repr(C)]` with no implicit padding
/// (the size asserts would catch padding being introduced).
///
/// # Safety
/// Implementors must be `#[repr(C)]`, contain only integers and arrays of
/// integers, and have no padding.
pub unsafe trait Pod: Copy + Default {
    fn as_bytes(&self) -> &[u8] {
        // SAFETY: Pod guarantees no padding and integer-only fields.
        unsafe { core::slice::from_raw_parts(self as *const Self as *const u8, size_of::<Self>()) }
    }

    /// Read one value from the start of `bytes`, or `None` if it is short.
    fn read_from(bytes: &[u8]) -> Option<Self> {
        if bytes.len() < size_of::<Self>() {
            return None;
        }
        let mut v = Self::default();
        // SAFETY: as above; any bit pattern is valid.
        unsafe {
            core::ptr::copy_nonoverlapping(
                bytes.as_ptr(),
                &mut v as *mut Self as *mut u8,
                size_of::<Self>(),
            );
        }
        Some(v)
    }
}

// SAFETY: each is #[repr(C)], integer-only, unpadded (sizes asserted above).
unsafe impl Pod for Hdr {}
unsafe impl Pod for Error {}
unsafe impl Pod for Hello {}
unsafe impl Pod for Behaviour {}
unsafe impl Pod for HelloReply {}
unsafe impl Pod for MemRegion {}
unsafe impl Pod for MemTable {}
unsafe impl Pod for Access {}
unsafe impl Pod for Value {}
unsafe impl Pod for Xregs {}
unsafe impl Pod for Xlate {}
unsafe impl Pod for XlateReply {}
unsafe impl Pod for Scanout {}
unsafe impl Pod for Cursor {}
unsafe impl Pod for CursorGlyph {}
unsafe impl Pod for Fatal {}
unsafe impl Pod for PostedEntry {}

#[cfg(test)]
mod tests {
    extern crate std;
    use std::string::String;

    const HEADER: &str = include_str!("../include/apv-gpu-proxy-protocol.h");

    fn header_define(name: &str) -> u64 {
        let full = std::format!("APV_PROXY_{name}");
        for line in HEADER.lines() {
            let mut it = line.split_whitespace();
            if it.next() == Some("#define") && it.next() == Some(full.as_str()) {
                let v: String = it
                    .next()
                    .expect("value")
                    .trim_end_matches('u')
                    .into();
                return if let Some(h) = v.strip_prefix("0x") {
                    u64::from_str_radix(h, 16).expect("hex")
                } else {
                    v.parse().expect("dec")
                };
            }
        }
        panic!("{full} not in the header");
    }

    #[test]
    fn every_constant_matches_the_header() {
        use super::*;
        let pairs: &[(&str, u64)] = &[
            ("VERSION", VERSION.into()),
            ("MAX_PAYLOAD", MAX_PAYLOAD.into()),
            ("MAX_FDS", MAX_FDS.into()),
            ("F_REPLY", F_REPLY.into()),
            ("F_ERROR", F_ERROR.into()),
            ("HELLO", HELLO.into()),
            ("MEM_TABLE", MEM_TABLE.into()),
            ("READ", READ.into()),
            ("WRITE", WRITE.into()),
            ("RESET", RESET.into()),
            ("SHUTDOWN", SHUTDOWN.into()),
            ("CB_XREGS", CB_XREGS.into()),
            ("CB_XLATE", CB_XLATE.into()),
            ("EV_SCANOUT", EV_SCANOUT.into()),
            ("EV_CURSOR", EV_CURSOR.into()),
            ("EV_CURSOR_GLYPH", EV_CURSOR_GLYPH.into()),
            ("EV_FATAL", EV_FATAL.into()),
            ("WINDOW_GFX", WINDOW_GFX.into()),
            ("WINDOW_IOSFC", WINDOW_IOSFC.into()),
            ("MAX_WINDOWS", MAX_WINDOWS.into()),
            ("R_SYNC", R_SYNC.into()),
            ("R_XCHG0", R_XCHG0.into()),
            ("R_LOAD", R_LOAD.into()),
            ("W_POSTED", W_POSTED.into()),
            ("W_SYNC", W_SYNC.into()),
            ("W_ANDNOT", W_ANDNOT.into()),
            ("W_BITSET", W_BITSET.into()),
            ("B_HARVEST", B_HARVEST.into()),
            ("B_SIZE4", B_SIZE4.into()),
            ("OFFSET_DEFAULT", OFFSET_DEFAULT.into()),
            ("FD_DEVICE_PAGE", FD_DEVICE_PAGE.into()),
            ("FD_KICK", FD_KICK.into()),
            ("FD_EVENTS", FD_EVENTS.into()),
            ("FD_SCANOUT", FD_SCANOUT.into()),
            ("FD_DIRTY", FD_DIRTY.into()),
            ("FD_IRQ0", FD_IRQ0.into()),
            ("PAGE_MAGIC", PAGE_MAGIC.into()),
            ("STATE_STARTING", STATE_STARTING.into()),
            ("STATE_RUNNING", STATE_RUNNING.into()),
            ("STATE_FAILED", STATE_FAILED.into()),
            ("POSTED_ENTRIES", POSTED_ENTRIES.into()),
            ("DIRTY_MAGIC", DIRTY_MAGIC.into()),
            ("DIRTY_FREE", DIRTY_FREE.into()),
            ("DIRTY_PENDING", DIRTY_PENDING.into()),
            ("DIRTY_ARMED", DIRTY_ARMED.into()),
            ("DIRTY_RELEASING", DIRTY_RELEASING.into()),
        ];
        for (name, value) in pairs {
            assert_eq!(header_define(name), *value, "APV_PROXY_{name}");
        }
    }

    #[test]
    fn every_header_define_is_mirrored() {
        let mirrored = 46;
        let defines = HEADER
            .lines()
            .filter(|l| l.starts_with("#define APV_PROXY_"))
            .count();
        assert_eq!(defines, mirrored, "a define was added to the header without a mirror");
    }
}
