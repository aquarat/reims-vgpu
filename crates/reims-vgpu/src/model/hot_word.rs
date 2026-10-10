//! A lock-free register word that may live outside this process's heap.
//!
//! Five guest registers are served without the device lock: the two
//! read-to-clear interrupt-status words (0x1014, 0x1018), the fault word
//! (0x102c), the main-FIFO consumer counter (0x100c) and the child-doorbell
//! mask (0x1020/0x1028). In the in-process build they are heap atomics shared
//! between the registry slot and [`crate::model::GfxRegs`]. When the device runs
//! in its own process, the VMM serves those registers itself from a page both
//! processes map, so the words have to *be* that page: a copy would turn a
//! read-to-clear into two independent clears.
//!
//! [`HotWord`] is either a heap atomic (the default, and what every test sees)
//! or a word inside a block installed once per process with
//! [`install_external_hot_words`]. Both deref to the same `AtomicU32`, so no
//! reader or writer knows which one it holds.

use std::ops::Deref;
use std::sync::atomic::AtomicU32;
use std::sync::{Arc, OnceLock};

/// The externally owned block, in the order the VMM is told about it.
///
/// `#[repr(C)]` because the other process addresses the words by offset.
#[repr(C)]
#[derive(Debug, Default)]
pub struct HotWordBlock {
    pub intr_status_disp: AtomicU32,
    pub intr_status_gpu: AtomicU32,
    pub intr_fault: AtomicU32,
    pub fifo_read: AtomicU32,
    pub child_doorbell_rung: AtomicU32,
}

/// Which word of [`HotWordBlock`] a [`HotWord`] stands for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum HotSlot {
    IntrStatusDisp,
    IntrStatusGpu,
    IntrFault,
    FifoRead,
    ChildDoorbellRung,
}

static EXTERNAL: OnceLock<&'static HotWordBlock> = OnceLock::new();

/// Make every device created from now on use `block` for its hot words.
///
/// Once per process; a second call is refused (returns `false`) rather than
/// silently re-pointing devices that already hold words in the first block.
/// The block must outlive the process, which is what `'static` says: the
/// caller maps it and never unmaps it.
pub fn install_external_hot_words(block: &'static HotWordBlock) -> bool {
    EXTERNAL.set(block).is_ok()
}

#[derive(Clone)]
pub struct HotWord(Inner);

#[derive(Clone)]
enum Inner {
    Owned(Arc<AtomicU32>),
    External(&'static AtomicU32),
}

impl HotWord {
    /// A word for `slot`: inside the installed external block when there is
    /// one, a fresh heap atomic otherwise.
    pub(crate) fn for_slot(slot: HotSlot) -> Self {
        match EXTERNAL.get() {
            Some(block) => Self(Inner::External(match slot {
                HotSlot::IntrStatusDisp => &block.intr_status_disp,
                HotSlot::IntrStatusGpu => &block.intr_status_gpu,
                HotSlot::IntrFault => &block.intr_fault,
                HotSlot::FifoRead => &block.fifo_read,
                HotSlot::ChildDoorbellRung => &block.child_doorbell_rung,
            })),
            None => Self(Inner::Owned(Arc::new(AtomicU32::new(0)))),
        }
    }
}

impl Deref for HotWord {
    type Target = AtomicU32;

    fn deref(&self) -> &AtomicU32 {
        match &self.0 {
            Inner::Owned(word) => word,
            Inner::External(word) => word,
        }
    }
}

impl std::fmt::Debug for HotWord {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let place = match self.0 {
            Inner::Owned(_) => "owned",
            Inner::External(_) => "external",
        };
        write!(
            f,
            "HotWord({place}, {:#x})",
            self.load(std::sync::atomic::Ordering::Relaxed)
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::Ordering;

    #[test]
    fn clones_share_one_word() {
        let a = HotWord(Inner::Owned(Arc::new(AtomicU32::new(0))));
        let b = a.clone();
        a.fetch_or(4, Ordering::AcqRel);
        assert_eq!(b.swap(0, Ordering::AcqRel), 4);
        assert_eq!(a.load(Ordering::Acquire), 0);
    }

    #[test]
    fn external_words_address_the_block() {
        static BLOCK: HotWordBlock = HotWordBlock {
            intr_status_disp: AtomicU32::new(0),
            intr_status_gpu: AtomicU32::new(0),
            intr_fault: AtomicU32::new(0),
            fifo_read: AtomicU32::new(0),
            child_doorbell_rung: AtomicU32::new(0),
        };
        let w = HotWord(Inner::External(&BLOCK.fifo_read));
        w.store(7, Ordering::Release);
        assert_eq!(BLOCK.fifo_read.load(Ordering::Acquire), 7);
        assert_eq!(std::mem::size_of::<HotWordBlock>(), 20);
    }
}
