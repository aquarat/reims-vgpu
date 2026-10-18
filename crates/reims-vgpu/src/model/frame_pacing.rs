//! The display refresh-rate cap: how fast this device's display refreshes, and,
//! when switched on, how fast a guest surface may be refreshed.
//!
//! # What is capped
//!
//! - The display modes this device advertises, and the VBL cadence it delivers,
//!   are at most [`max_refresh_hz`]: 60 unless the operator sets
//!   `REIMS_VGPU_MAX_REFRESH_HZ` (`0` = uncapped, which is the native 120).
//!   A compositor that takes its frame clock from this display's VBL — macOS
//!   13's WindowServer does — draws no faster than that.
//! - With `REIMS_VGPU_FRAME_PACING` on, a packet whose work writes a guest
//!   IOSurface (a mapper-ref texture or a mapping's pages — what a
//!   compositor's frame is) is a **frame** of each surface it writes, and a
//!   surface's frames complete at most once per refresh interval: the
//!   completion words of a frame that arrives early are held, in channel
//!   order, until its refresh comes round, and so is every later word of that
//!   channel. Nothing is dropped and nothing is reordered; the work itself
//!   runs when it arrives.
//!
//! # Why surface pacing exists, and why it is off by default
//!
//! Nothing paces a headless guest's other compositors. macOS 26 never arms
//! this device's VBL or present-complete events, and a CI guest's display
//! sleeps; the iOS simulator's screen composite (a full-screen compute into two
//! guest IOSurfaces) then redraws as fast as the guest's CPU allows — about 67
//! times a second with the device in process and 92 out of process on a
//! 4-vCPU guest — spending CPU the tests need on frames nobody sees.
//!
//! That compositor queues dozens of frames ahead, and nothing short of a full
//! queue holds it back: a hold bounded at one interval changed nothing. So
//! once its queue is full every completion on its channel trails its work by
//! up to that queue's depth in refreshes, about a second — and on macOS 26
//! every guest client shares that channel. Measured (simulator unit tests and
//! UI tests, 2026-10-18), pacing saves unit-test time and costs about 10 % of
//! UI-test time, so it is an operator's choice rather than the default.

use std::collections::VecDeque;

use super::regs::{DISPLAY_REFRESH_HZ, MAX_CHANNELS};

/// The cap this device ships with: what a headless guest's compositors are
/// paced to when the operator says nothing.
pub const DEFAULT_MAX_REFRESH_HZ: u32 = 60;

/// The refresh-rate cap this process runs with, read once.
///
/// `0` means uncapped: the display runs at its native [`DISPLAY_REFRESH_HZ`]
/// and frames complete as soon as their work has.
pub fn max_refresh_hz() -> u32 {
    static CAP: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *CAP.get_or_init(|| {
        use crate::config::RateCap;
        match crate::config::rate_cap(crate::config::MAX_REFRESH_HZ, u64::from(DISPLAY_REFRESH_HZ))
        {
            RateCap::Unset => DEFAULT_MAX_REFRESH_HZ,
            RateCap::Uncapped => 0,
            RateCap::Capped(hz) => hz as u32,
            RateCap::Refused(raw) => {
                crate::observe::fail(format!(
                    "max_refresh_hz_refused value={raw:?} native={DISPLAY_REFRESH_HZ} \
                     (0 for uncapped or 1..={DISPLAY_REFRESH_HZ}; keeping \
                     {DEFAULT_MAX_REFRESH_HZ} Hz)"
                ));
                DEFAULT_MAX_REFRESH_HZ
            }
        }
    })
}

/// Whether guest surfaces are paced to the cap (`REIMS_VGPU_FRAME_PACING`,
/// off unless switched on), read once.
pub fn surface_pacing_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        crate::config::read(crate::config::FRAME_PACING).0 == crate::config::Switch::On
    })
}

/// The refresh rate the display advertises and delivers VBL at.
pub fn display_refresh_hz() -> u32 {
    effective_refresh_hz(max_refresh_hz())
}

/// The display rate under a cap of `cap` (`0` = uncapped).
pub const fn effective_refresh_hz(cap: u32) -> u32 {
    if cap == 0 || cap > DISPLAY_REFRESH_HZ {
        DISPLAY_REFRESH_HZ
    } else {
        cap
    }
}

/// The display's refresh interval in microseconds.
pub fn display_refresh_interval_us() -> u64 {
    1_000_000 / u64::from(display_refresh_hz())
}

/// A completion word held back until its frame's refresh.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HeldWord {
    /// The stamp slot the word is for.
    pub slot: u32,
    /// The value it publishes.
    pub value: u32,
    /// The earliest time (crate clock, µs) it may be published.
    pub due_us: u64,
}

/// What the pacer did, cumulative since the device was created.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PacingCensus {
    /// Packets that wrote a guest surface.
    pub frames: u64,
    /// Of those, frames that arrived before their refresh and were held.
    pub paced: u64,
    /// Total time frames were held, µs.
    pub held_us: u64,
    /// Completion words that waited in a channel's queue.
    pub words_held: u64,
    /// Longest any channel's queue has been.
    pub max_queue: usize,
    /// Longest any frame has been held, µs.
    pub max_hold_us: u64,
}

/// Per-device pacing state. Owned by `DeviceState`; the drain decides when a
/// packet is done and publishes what [`Self::take_due`] hands back.
#[derive(Debug)]
pub struct FramePacing {
    /// Frame interval, µs; `0` = uncapped.
    interval_us: u64,
    /// Guest surfaces (mapping ids) the packet now running on each channel
    /// has written.
    surface_writes: [Vec<u32>; MAX_CHANNELS],
    /// Per surface: the earliest time its next frame may complete.
    next_frame_us: std::collections::HashMap<u32, u64>,
    /// Per channel: completion words waiting, in publication order.
    held: [VecDeque<HeldWord>; MAX_CHANNELS],
    /// Words in all queues; lets the poll skip an idle pacer in one compare.
    queued: usize,
    pub census: PacingCensus,
    /// Frames per surface since the census line was last written.
    pub window_frames: std::collections::BTreeMap<u32, u32>,
    /// When the census line was last written (crate clock, µs).
    pub census_reported_us: u64,
}

impl Default for FramePacing {
    /// Paced at the cap when surface pacing is switched on; otherwise it only
    /// counts surface refreshes for the census.
    fn default() -> Self {
        Self::new(if surface_pacing_enabled() {
            max_refresh_hz()
        } else {
            0
        })
    }
}

impl FramePacing {
    /// A pacer for a cap of `max_hz` frames a second (`0` = uncapped).
    pub fn new(max_hz: u32) -> Self {
        Self {
            interval_us: if max_hz == 0 {
                0
            } else {
                1_000_000 / u64::from(max_hz)
            },
            surface_writes: std::array::from_fn(|_| Vec::new()),
            next_frame_us: std::collections::HashMap::new(),
            held: std::array::from_fn(|_| VecDeque::new()),
            queued: 0,
            census: PacingCensus::default(),
            window_frames: std::collections::BTreeMap::new(),
            census_reported_us: 0,
        }
    }

    /// The frame interval in µs; `0` when uncapped.
    pub fn interval_us(&self) -> u64 {
        self.interval_us
    }

    /// The packet now running on the drain wrote guest surface `mapping_id`.
    ///
    /// Only a child channel mid-packet counts: a surface written by the
    /// IOSurface mapper or the boot console belongs to no channel's frame.
    pub fn note_surface_write(
        &mut self,
        draining_mask: u32,
        draining_channel: u32,
        mapping_id: u32,
    ) {
        let Some(bit) = 1u32.checked_shl(draining_channel) else {
            return;
        };
        if draining_mask & bit == 0 || draining_channel as usize >= MAX_CHANNELS {
            return;
        }
        let writes = &mut self.surface_writes[draining_channel as usize];
        if !writes.contains(&mapping_id) {
            writes.push(mapping_id);
        }
    }

    /// The packet on `domain` has run. If it was a frame — it wrote a guest
    /// surface — that came before that surface's next refresh, returns when
    /// its completion may be published.
    ///
    /// Each surface is a display of its own: it is refreshed at most once per
    /// interval, so two surfaces a compositor redraws alternately are each
    /// paced, not each other's frames.
    pub fn frame_done(&mut self, domain: u32, now_us: u64) -> Option<u64> {
        let d = domain as usize;
        if d >= MAX_CHANNELS || self.surface_writes[d].is_empty() {
            return None;
        }
        let surfaces = std::mem::take(&mut self.surface_writes[d]);
        for mid in &surfaces {
            *self.window_frames.entry(*mid).or_insert(0) += 1;
        }
        if self.interval_us == 0 {
            return None;
        }
        self.census.frames += 1;
        for mid in &surfaces {
            if crate::observe::first_sight(
                "frame_surface",
                u64::from(domain) << 32 | u64::from(*mid),
            ) {
                crate::observe::off(format!("frame_surface ch={domain} mid={mid}"));
            }
        }
        // Not bounded by `now`: a compositor that queues frames ahead of the
        // display — the simulator's does, dozens deep — is only held to the
        // cap once its queue is full, so a bound here would cap nothing. The
        // price is that completions queued behind a held word on the same
        // channel wait with it, up to the guest's queue depth in refreshes.
        let release = surfaces
            .iter()
            .filter_map(|mid| self.next_frame_us.get(mid).copied())
            .fold(now_us, u64::max);
        for mid in &surfaces {
            self.next_frame_us.insert(*mid, release + self.interval_us);
        }
        // A surface not refreshed for a second is no longer being animated;
        // forgetting it keeps the table to the surfaces that are.
        if self.next_frame_us.len() > 64 {
            self.next_frame_us
                .retain(|_, next| next.saturating_add(1_000_000) > now_us);
        }
        if release > now_us {
            self.census.paced += 1;
            self.census.held_us += release - now_us;
            self.census.max_hold_us = self.census.max_hold_us.max(release - now_us);
            Some(release)
        } else {
            None
        }
    }

    /// Whether `domain`'s word must wait; if so it is queued behind the
    /// channel's earlier words. `due` is the frame time [`Self::frame_done`]
    /// gave the packet this word completes, if any.
    pub fn hold(
        &mut self,
        domain: u32,
        slot: u32,
        value: u32,
        due: Option<u64>,
        now_us: u64,
    ) -> bool {
        let d = domain as usize;
        if d >= MAX_CHANNELS {
            return false;
        }
        let queue = &mut self.held[d];
        let behind = queue.back().map_or(0, |w| w.due_us);
        let due_us = due.unwrap_or(0).max(behind);
        if queue.is_empty() && due_us <= now_us {
            return false;
        }
        queue.push_back(HeldWord {
            slot,
            value,
            due_us,
        });
        self.queued += 1;
        self.census.words_held += 1;
        self.census.max_queue = self.census.max_queue.max(queue.len());
        true
    }

    /// Whether any word is waiting.
    pub fn holding(&self) -> bool {
        self.queued != 0
    }

    /// Every word whose refresh has come, as `(domain, word)` in each
    /// channel's publication order.
    pub fn take_due(&mut self, now_us: u64) -> Vec<(u32, HeldWord)> {
        let mut out = Vec::new();
        if self.queued == 0 {
            return out;
        }
        for (d, queue) in self.held.iter_mut().enumerate() {
            while queue.front().is_some_and(|w| w.due_us <= now_us) {
                if let Some(w) = queue.pop_front() {
                    out.push((d as u32, w));
                }
            }
        }
        self.queued -= out.len();
        out
    }

    /// Every waiting word, due or not, in each channel's publication order:
    /// for a device going away, where a late word is better than none.
    pub fn take_all(&mut self) -> Vec<(u32, HeldWord)> {
        self.take_due(u64::MAX)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const HZ60_US: u64 = 1_000_000 / 60;

    fn frame(p: &mut FramePacing, domain: u32, now: u64) -> Option<u64> {
        frame_on(p, domain, 100 + domain, now)
    }

    fn frame_on(p: &mut FramePacing, domain: u32, mid: u32, now: u64) -> Option<u64> {
        p.note_surface_write(1 << domain, domain, mid);
        p.frame_done(domain, now)
    }

    #[test]
    fn a_packet_that_wrote_no_surface_is_not_a_frame() {
        let mut p = FramePacing::new(60);
        assert_eq!(p.frame_done(3, 1_000), None);
        assert_eq!(p.census.frames, 0);
    }

    #[test]
    fn a_write_outside_a_channel_packet_is_not_a_frame() {
        let mut p = FramePacing::new(60);
        p.note_surface_write(0, 3, 7);
        assert_eq!(p.frame_done(3, 1_000), None);
        assert_eq!(p.census.frames, 0);
    }

    #[test]
    fn frames_faster_than_the_cap_complete_one_interval_apart() {
        let mut p = FramePacing::new(60);
        assert_eq!(frame(&mut p, 2, 1_000), None);
        assert_eq!(frame(&mut p, 2, 2_000), Some(1_000 + HZ60_US));
        // The compositor drew its next frame once that one completed.
        let t = 1_000 + HZ60_US + 500;
        assert_eq!(frame(&mut p, 2, t), Some(1_000 + 2 * HZ60_US));
        assert_eq!(p.census.frames, 3);
        assert_eq!(p.census.paced, 2);
    }

    #[test]
    fn frames_queued_ahead_complete_one_interval_apart() {
        let mut p = FramePacing::new(60);
        assert_eq!(frame(&mut p, 2, 1_000), None);
        let mut last = 1_000;
        for now in [1_100, 1_200, 1_300] {
            let due = frame(&mut p, 2, now).expect("held");
            assert_eq!(due, last + HZ60_US);
            last = due;
        }
    }

    #[test]
    fn frames_slower_than_the_cap_are_never_held() {
        let mut p = FramePacing::new(60);
        for i in 0..5 {
            assert_eq!(frame(&mut p, 1, 1_000 + i * 20_000), None);
        }
        assert_eq!(p.census.paced, 0);
    }

    #[test]
    fn channels_are_paced_apart() {
        let mut p = FramePacing::new(60);
        assert_eq!(frame(&mut p, 1, 1_000), None);
        assert_eq!(frame(&mut p, 2, 1_100), None);
        assert!(frame(&mut p, 1, 1_200).is_some());
    }

    #[test]
    fn surfaces_are_paced_apart_even_on_one_channel() {
        let mut p = FramePacing::new(60);
        assert_eq!(frame_on(&mut p, 1, 4, 1_000), None);
        assert_eq!(frame_on(&mut p, 1, 5, 1_100), None);
        assert_eq!(frame_on(&mut p, 1, 4, 1_200), Some(1_000 + HZ60_US));
        // Surface 5 keeps its own refresh.
        assert_eq!(frame_on(&mut p, 1, 5, 1_300), Some(1_100 + HZ60_US));
    }

    #[test]
    fn uncapped_holds_nothing() {
        let mut p = FramePacing::new(0);
        assert_eq!(frame(&mut p, 1, 1_000), None);
        assert_eq!(frame(&mut p, 1, 1_001), None);
        assert!(!p.hold(1, 5, 7, None, 1_002));
    }

    #[test]
    fn held_words_publish_in_channel_order_and_nothing_overtakes_them() {
        let mut p = FramePacing::new(60);
        let due = 10_000;
        assert!(p.hold(4, 4, 100, Some(due), 1_000));
        // A later word of the same channel with no frame of its own still
        // waits behind the frame.
        assert!(p.hold(4, 4, 101, None, 2_000));
        // Another channel is not held.
        assert!(!p.hold(5, 5, 7, None, 2_000));
        assert!(p.take_due(9_999).is_empty());
        let out = p.take_due(due);
        assert_eq!(
            out.iter().map(|(d, w)| (*d, w.value)).collect::<Vec<_>>(),
            vec![(4, 100), (4, 101)]
        );
        assert!(!p.holding());
        // With the queue empty the channel publishes at once again.
        assert!(!p.hold(4, 4, 102, None, due + 1));
    }

    #[test]
    fn a_later_frame_never_releases_an_earlier_word_early() {
        let mut p = FramePacing::new(60);
        assert!(p.hold(1, 1, 1, Some(20_000), 1_000));
        assert!(p.hold(1, 1, 2, Some(15_000), 1_000));
        assert!(p.take_due(15_000).is_empty());
        assert_eq!(p.take_due(20_000).len(), 2);
    }

    #[test]
    fn the_display_rate_is_the_cap_and_never_above_native() {
        assert_eq!(effective_refresh_hz(0), DISPLAY_REFRESH_HZ);
        assert_eq!(effective_refresh_hz(60), 60);
        assert_eq!(
            effective_refresh_hz(DISPLAY_REFRESH_HZ + 1),
            DISPLAY_REFRESH_HZ
        );
    }
}
