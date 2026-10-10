//! Which registers the VMM serves without a round trip, and how.
//!
//! This is a transcription of the arms `device_gfx_read` / `device_gfx_write`
//! already take without the device lock: the read-to-clear interrupt-status
//! words, the fault and FIFO-consumer words, the status clears and the child
//! doorbells. Everything else in the gfx window is posted (the in-process
//! queue path), and the IOSurface window is synchronous, as its handler takes
//! the device lock and may read the publishing vCPU's registers.

use reims_vgpu::remote as dev;
use reims_vgpu::remote::HotWordBlock;
use reims_vgpu_proxy_wire::{self as wire, Behaviour};

use crate::backend::HOT_BLOCK_OFFSET;

fn word(field: usize) -> u32 {
    HOT_BLOCK_OFFSET + field as u32
}

pub fn behaviours() -> Vec<Behaviour> {
    let disp = word(std::mem::offset_of!(HotWordBlock, intr_status_disp));
    let gpu = word(std::mem::offset_of!(HotWordBlock, intr_status_gpu));
    let fault = word(std::mem::offset_of!(HotWordBlock, intr_fault));
    let fifo = word(std::mem::offset_of!(HotWordBlock, fifo_read));
    let rung = word(std::mem::offset_of!(HotWordBlock, child_doorbell_rung));
    let gfx = |offset: u64, read_kind: u8, write_kind: u8, word: u32, arg_lo: u16, arg_hi: u16| Behaviour {
        window: wire::WINDOW_GFX as u8,
        read_kind,
        write_kind,
        // Every write harvests first, as both in-process shims did before
        // handing the device anything.
        flags: wire::B_HARVEST | wire::B_SIZE4,
        offset: offset as u32,
        word,
        arg_lo,
        arg_hi,
    };
    vec![
        Behaviour {
            window: wire::WINDOW_GFX as u8,
            read_kind: wire::R_SYNC,
            write_kind: wire::W_POSTED,
            flags: wire::B_HARVEST,
            offset: wire::OFFSET_DEFAULT,
            ..Default::default()
        },
        Behaviour {
            window: wire::WINDOW_IOSFC as u8,
            read_kind: wire::R_SYNC,
            write_kind: wire::W_SYNC,
            flags: wire::B_HARVEST,
            offset: wire::OFFSET_DEFAULT,
            ..Default::default()
        },
        gfx(dev::GFX_REG_INTR_STATUS_DISP, wire::R_XCHG0, wire::W_ANDNOT, disp, 0, 0),
        gfx(dev::GFX_REG_INTR_STATUS_GPU, wire::R_XCHG0, wire::W_ANDNOT, gpu, 0, 0),
        gfx(dev::GFX_REG_INTR_FAULT, wire::R_LOAD, wire::W_POSTED, fault, 0, 0),
        gfx(dev::GFX_REG_FIFO_READ, wire::R_LOAD, wire::W_POSTED, fifo, 0, 0),
        gfx(
            dev::GFX_REG_CHILD_DOORBELL,
            wire::R_SYNC,
            wire::W_BITSET,
            rung,
            dev::CHILD_CHANNEL_FIRST as u16,
            dev::CHILD_CHANNEL_LIMIT as u16,
        ),
        gfx(
            dev::GFX_REG_CHILD_REPLAY_DOORBELL,
            wire::R_SYNC,
            wire::W_BITSET,
            rung,
            dev::CHILD_CHANNEL_FIRST as u16,
            dev::CHILD_CHANNEL_LIMIT as u16,
        ),
    ]
}
