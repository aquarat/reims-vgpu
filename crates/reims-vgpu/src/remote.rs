//! The surface an out-of-process host (`reims-vgpu-backend`) needs beyond the
//! QEMU C ABI it already drives.
//!
//! The backend runs this device in its own process and lets the VMM serve the
//! lock-free registers from a shared page. It therefore needs to know *which*
//! registers those are and with what semantics, and to place the words in that
//! page. Both answers live here, next to the device, so the backend's register
//! table is derived from the same constants the in-process handlers branch on
//! rather than from a second copy of the offsets.

use crate::model;

pub use crate::model::{install_external_hot_words, HotWordBlock};

/// Gfx window size; the bound the sparse register store is indexed against.
pub const GFX_MMIO_SIZE: u64 = model::GFX_MMIO_SIZE;
/// Read-to-clear display interrupt status; writes clear the written bits.
pub const GFX_REG_INTR_STATUS_DISP: u64 = model::GFX_REG_INTR_STATUS_DISP;
/// Read-to-clear stamp-signal interrupt status; writes clear the written bits.
pub const GFX_REG_INTR_STATUS_GPU: u64 = model::GFX_REG_INTR_STATUS_GPU;
/// Fault status, host-set and guest-read (not read-to-clear).
pub const GFX_REG_INTR_FAULT: u64 = model::GFX_REG_INTR_FAULT;
/// Main-FIFO consumer counter, host-advanced and polled by the guest producer.
pub const GFX_REG_FIFO_READ: u64 = model::GFX_REG_FIFO_READ;
/// The two child doorbells: the written value names the channel rung.
pub const GFX_REG_CHILD_DOORBELL: u64 = model::GFX_REG_CHILD_DOORBELL;
pub const GFX_REG_CHILD_REPLAY_DOORBELL: u64 = model::GFX_REG_CHILD_REPLAY_DOORBELL;

/// Channels a doorbell may name: `1..MAX_CHANNELS` (channel 0 is the main FIFO,
/// which has its own register). See [`model::is_child_channel`].
pub const CHILD_CHANNEL_FIRST: u32 = 1;
pub const CHILD_CHANNEL_LIMIT: u32 = model::MAX_CHANNELS as u32;

/// Count doorbells the VMM took lock-free on this device's behalf, so the
/// `gfx_doorbell_delay` census keeps meaning what it does in process.
pub fn note_doorbells_lock_free(count: u64) {
    for _ in 0..count {
        crate::runtime::drain::note_doorbell_lock_free();
    }
}

/// Report doorbells the VMM refused because they named no child channel, as
/// the in-process handler does for each one it sees.
pub fn note_out_of_range_doorbell(channel: u32) {
    let _ = model::accept_child_channel(channel, "proxy_child_doorbell");
}
