//! A tightly packed host buffer laid over the guest runs that hold its bytes.
//!
//! # Why this exists
//!
//! The copying rails move a guest window between guest RAM and mapped Vulkan
//! memory (an upload's staging buffer, a readback buffer). They used to do it
//! in two hops through a heap `Vec`: guest pages into the `Vec`, then the `Vec`
//! into the staging buffer; or the readback buffer into a `Vec`, then the `Vec`
//! into guest pages. The `Vec` is never read for anything else on those paths,
//! so the second copy is pure cost — on a compositor-heavy guest it was close to
//! half of the device's copy traffic.
//!
//! [`PackedGuestLayout`] is the description that lets one copy do it: for every
//! byte of a packed buffer `0..packed_len`, the host address of the guest byte it
//! stands for. A producer builds it from the runs it resolved (a page-table
//! walk, a mapping's contiguous view); a consumer that holds the mapped buffer
//! calls [`PackedGuestLayout::gather_into`] or [`PackedGuestLayout::scatter_from`].
//!
//! # The bound
//!
//! Every [`PackedSegment`] is a [`GuestRun`], so no segment can name a byte
//! outside the mapping it was cut from — the same constructor rule the gather
//! rail's runs obey. What this type adds is the *tiling*: the segments ascend
//! and cover the packed buffer exactly, with no hole and no overlap, so a
//! gather leaves no byte of the destination undefined and a scatter writes each
//! guest byte the window names exactly once. A window the runs do not cover is
//! not a layout, and [`PackedGuestLayout::rows`] answers `None` for it rather
//! than producing a partial one.

use crate::GuestRun;

/// One stretch of guest memory a window covers, positioned in that window.
///
/// `window_offset` is the distance from the window's first byte to the run's
/// first byte. The run is already clipped to the window: a walk that maps whole
/// pages cuts the head of the first page and the tail of the last before
/// building one of these, so the offset is never negative.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WindowRun {
    pub window_offset: u64,
    pub run: GuestRun,
}

/// A stretch of the packed buffer and the guest bytes it stands for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PackedSegment {
    packed_offset: u64,
    run: GuestRun,
}

impl PackedSegment {
    /// Where this stretch starts in the packed buffer.
    pub fn packed_offset(&self) -> u64 {
        self.packed_offset
    }

    /// The guest bytes, as the host range that reads and writes them.
    pub fn run(&self) -> GuestRun {
        self.run
    }
}

/// Every byte of a packed buffer, mapped to the guest byte it stands for.
///
/// Built only by [`Self::rows`], which proves the tiling. The pointers inside
/// are only as live as the mappings the runs were cut from: the producer owns
/// those and must keep them until the last [`Self::gather_into`] or
/// [`Self::scatter_from`] returns.
#[derive(Clone, PartialEq, Eq)]
pub struct PackedGuestLayout {
    segments: Vec<PackedSegment>,
    packed_len: u64,
}

impl std::fmt::Debug for PackedGuestLayout {
    /// The shape, never the addresses: the segments are live host pointers
    /// into guest RAM, and a derived `Debug` would put them in any line that
    /// formats a request carrying one.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "PackedGuestLayout({} bytes, {} segments)",
            self.packed_len,
            self.segments.len()
        )
    }
}

impl PackedGuestLayout {
    /// The rows of a strided guest window, packed tight.
    ///
    /// Row `y` is window bytes `y * row_pitch .. y * row_pitch + row_bytes` and
    /// packed bytes `y * row_bytes .. (y + 1) * row_bytes`. Pitch padding is
    /// never part of the layout — a padded row's tail belongs to the guest and
    /// a scatter must leave it alone — and so neither is the padding after the
    /// last row, which the guest need not even have mapped.
    ///
    /// `runs` must ascend by `window_offset` without overlapping. Segments that
    /// continue one another in both the packed buffer and the same run are
    /// merged, so a dense window (`row_pitch == row_bytes`) costs one segment
    /// per run rather than one per row.
    ///
    /// `None` when the geometry is empty or inconsistent (`rows == 0`,
    /// `row_bytes == 0`, `row_pitch < row_bytes`), when the arithmetic would
    /// overflow, when the runs are out of order or overlap, or when any row
    /// byte is not covered by a run.
    pub fn rows(runs: &[WindowRun], rows: u32, row_pitch: u64, row_bytes: u64) -> Option<Self> {
        if rows == 0 || row_bytes == 0 || row_pitch < row_bytes {
            return None;
        }
        let mut previous_end = 0u64;
        for (index, window_run) in runs.iter().enumerate() {
            if window_run.run.is_empty() || (index > 0 && window_run.window_offset < previous_end) {
                return None;
            }
            previous_end = window_run.window_offset.checked_add(window_run.run.len())?;
        }
        let packed_len = u64::from(rows).checked_mul(row_bytes)?;
        // The last row ends here; checked once so every row below can use
        // plain arithmetic on values bounded by it.
        u64::from(rows - 1)
            .checked_mul(row_pitch)?
            .checked_add(row_bytes)?;
        let mut segments: Vec<PackedSegment> = Vec::new();
        // Which run the previous segment was cut from, so a merge never joins
        // two mappings that merely happen to be adjacent in host memory.
        let mut last_run_index = usize::MAX;
        let mut index = 0usize;
        for y in 0..u64::from(rows) {
            let row_lo = y * row_pitch;
            let row_hi = row_lo + row_bytes;
            let mut cursor = row_lo;
            while cursor < row_hi {
                while index < runs.len()
                    && runs[index].window_offset + runs[index].run.len() <= cursor
                {
                    index += 1;
                }
                let window_run = runs.get(index)?;
                if window_run.window_offset > cursor {
                    // A hole: these row bytes have no guest address.
                    return None;
                }
                let run_end = window_run.window_offset + window_run.run.len();
                let n = row_hi.min(run_end) - cursor;
                let within = cursor - window_run.window_offset;
                let packed_offset = y * row_bytes + (cursor - row_lo);
                let host_ptr = window_run.run.host_ptr() + within as usize;
                let merged = match segments.last_mut() {
                    Some(last)
                        if last_run_index == index
                            && last.packed_offset + last.run.len() == packed_offset
                            && last.run.host_ptr() + last.run.len() as usize == host_ptr =>
                    {
                        // Same mapping, so the longer run is still inside it.
                        last.run = GuestRun {
                            host_ptr: last.run.host_ptr(),
                            len: last.run.len() + n,
                        };
                        true
                    }
                    _ => false,
                };
                if !merged {
                    segments.push(PackedSegment {
                        packed_offset,
                        run: GuestRun::in_mapping(
                            window_run.run.host_ptr(),
                            window_run.run.len(),
                            within,
                            n,
                        )?,
                    });
                    last_run_index = index;
                }
                cursor += n;
            }
        }
        Some(Self {
            segments,
            packed_len,
        })
    }

    /// Bytes in the packed buffer.
    pub fn packed_len(&self) -> u64 {
        self.packed_len
    }

    /// The segments, ascending and tiling `0..packed_len`.
    pub fn segments(&self) -> &[PackedSegment] {
        &self.segments
    }

    /// Copy the guest bytes into a packed buffer.
    ///
    /// # Safety
    ///
    /// Every run's mapping must be live and readable, and `dst` must be valid
    /// for [`Self::packed_len`] bytes of writes that overlap no run.
    pub unsafe fn gather_into(&self, dst: *mut u8) {
        for segment in &self.segments {
            std::ptr::copy_nonoverlapping(
                segment.run.host_ptr() as *const u8,
                dst.add(segment.packed_offset as usize),
                segment.run.len() as usize,
            );
        }
    }

    /// Copy a packed buffer out to the guest bytes.
    ///
    /// # Safety
    ///
    /// Every run's mapping must be live and writable, and `src` must be valid
    /// for [`Self::packed_len`] bytes of reads that overlap no run.
    pub unsafe fn scatter_from(&self, src: *const u8) {
        for segment in &self.segments {
            std::ptr::copy_nonoverlapping(
                src.add(segment.packed_offset as usize),
                segment.run.host_ptr() as *mut u8,
                segment.run.len() as usize,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn window_run(buf: &[u8], window_offset: u64, start: usize, len: usize) -> WindowRun {
        WindowRun {
            window_offset,
            run: GuestRun::in_mapping(
                buf.as_ptr() as usize,
                buf.len() as u64,
                start as u64,
                len as u64,
            )
            .expect("fixture run inside its buffer"),
        }
    }

    /// The packed bytes a correct gather produces, computed the slow way from a
    /// window held as one contiguous buffer.
    fn reference_rows(window: &[u8], rows: usize, pitch: usize, row_bytes: usize) -> Vec<u8> {
        let mut out = Vec::new();
        for y in 0..rows {
            out.extend_from_slice(&window[y * pitch..y * pitch + row_bytes]);
        }
        out
    }

    fn gather(layout: &PackedGuestLayout) -> Vec<u8> {
        let mut out = vec![0xEEu8; layout.packed_len() as usize];
        unsafe { layout.gather_into(out.as_mut_ptr()) };
        out
    }

    /// A dense window in one run is one segment and one copy, however many rows.
    #[test]
    fn a_dense_window_in_one_run_is_one_segment() {
        let window: Vec<u8> = (0..64u8).collect();
        let layout =
            PackedGuestLayout::rows(&[window_run(&window, 0, 0, 64)], 8, 8, 8).expect("covered");
        assert_eq!(layout.segments().len(), 1);
        assert_eq!(layout.packed_len(), 64);
        assert_eq!(gather(&layout), window);
    }

    /// A padded pitch takes the tight part of every row and never the padding,
    /// and the last row needs no padding after it to exist.
    #[test]
    fn a_padded_pitch_packs_the_rows_and_skips_the_padding() {
        // 3 rows of 5 bytes at a pitch of 8; the window ends at the last row's
        // last byte (2 * 8 + 5 = 21), so the final row's padding is absent.
        let window: Vec<u8> = (0..21u8).collect();
        let layout = PackedGuestLayout::rows(&[window_run(&window, 0, 0, 21)], 3, 8, 5)
            .expect("the final row needs no padding");
        assert_eq!(layout.segments().len(), 3, "one segment per row");
        assert_eq!(gather(&layout), reference_rows(&window, 3, 8, 5));
    }

    /// A run boundary in the middle of a row splits that row and nothing else,
    /// and the two halves land at the right packed offsets.
    #[test]
    fn a_run_boundary_inside_a_row_splits_only_that_row() {
        // Window of 4 dense rows of 6 bytes = 24 bytes, held as two separate
        // buffers split at byte 9 (row 1, column 3).
        let window: Vec<u8> = (100..124u8).collect();
        let head = window[..9].to_vec();
        let tail = window[9..].to_vec();
        let runs = [window_run(&head, 0, 0, 9), window_run(&tail, 9, 0, 15)];
        let layout = PackedGuestLayout::rows(&runs, 4, 6, 6).expect("covered");
        assert_eq!(layout.segments().len(), 2, "dense rows merge within a run");
        assert_eq!(layout.segments()[1].packed_offset(), 9);
        assert_eq!(gather(&layout), window);
    }

    /// Two runs that are adjacent in host memory but are different runs are not
    /// merged: a merged run would claim one mapping covers both.
    #[test]
    fn adjacent_runs_from_different_mappings_are_not_merged() {
        let backing: Vec<u8> = (0..16u8).collect();
        let runs = [window_run(&backing, 0, 0, 8), window_run(&backing, 8, 8, 8)];
        let layout = PackedGuestLayout::rows(&runs, 1, 16, 16).expect("covered");
        assert_eq!(layout.segments().len(), 2);
        assert_eq!(gather(&layout), backing);
    }

    /// The window's first run may start inside the buffer it was cut from — a
    /// texture that does not begin on a page boundary.
    #[test]
    fn a_run_cut_from_the_middle_of_a_mapping_reads_from_its_offset() {
        let mapping: Vec<u8> = (0..32u8).collect();
        // The window is mapping bytes 5..21: 2 rows of 6 at a pitch of 10
        // (row 0 = 5..11, row 1 = 15..21).
        let layout =
            PackedGuestLayout::rows(&[window_run(&mapping, 0, 5, 16)], 2, 10, 6).expect("covered");
        assert_eq!(gather(&layout), [5, 6, 7, 8, 9, 10, 15, 16, 17, 18, 19, 20]);
    }

    /// Scatter writes the row bytes and leaves every padding byte as it was.
    #[test]
    fn scatter_writes_rows_and_leaves_padding_alone() {
        let mut window = vec![0xAAu8; 2 * 8 + 5];
        let ptr = window.as_mut_ptr() as usize;
        let run = WindowRun {
            window_offset: 0,
            run: GuestRun::in_mapping(ptr, window.len() as u64, 0, window.len() as u64).unwrap(),
        };
        let layout = PackedGuestLayout::rows(&[run], 3, 8, 5).expect("covered");
        let packed: Vec<u8> = (1..=15u8).collect();
        unsafe { layout.scatter_from(packed.as_ptr()) };
        assert_eq!(
            window,
            [
                1, 2, 3, 4, 5, 0xAA, 0xAA, 0xAA, //
                6, 7, 8, 9, 10, 0xAA, 0xAA, 0xAA, //
                11, 12, 13, 14, 15
            ]
        );
    }

    /// Gather then scatter through a fragmented, padded window is the identity
    /// on the row bytes — the shape the compositor actually hands the device.
    #[test]
    fn a_fragmented_padded_window_round_trips() {
        let rows = 7usize;
        let pitch = 13usize;
        let row_bytes = 9usize;
        let window_len = (rows - 1) * pitch + row_bytes;
        let window: Vec<u8> = (0..window_len).map(|i| (i * 7 + 3) as u8).collect();
        // Cut at offsets that land inside rows, on row starts and in padding.
        let cuts = [0usize, 4, 13, 22, 30, 31, 50, 60, window_len];
        let pieces: Vec<Vec<u8>> = cuts
            .windows(2)
            .map(|w| window[w[0]..w[1]].to_vec())
            .collect();
        let runs: Vec<WindowRun> = cuts
            .windows(2)
            .zip(&pieces)
            .map(|(w, piece)| window_run(piece, w[0] as u64, 0, piece.len()))
            .collect();
        let layout =
            PackedGuestLayout::rows(&runs, rows as u32, pitch as u64, row_bytes as u64).unwrap();
        let packed = gather(&layout);
        assert_eq!(packed, reference_rows(&window, rows, pitch, row_bytes));
        // Every segment ascends and the tiling is exact.
        let mut next = 0u64;
        for segment in layout.segments() {
            assert_eq!(segment.packed_offset(), next);
            next += segment.run().len();
        }
        assert_eq!(next, layout.packed_len());

        let mut out: Vec<Vec<u8>> = pieces.iter().map(|p| vec![0u8; p.len()]).collect();
        let out_runs: Vec<WindowRun> = cuts
            .windows(2)
            .zip(out.iter_mut())
            .map(|(w, piece)| WindowRun {
                window_offset: w[0] as u64,
                run: GuestRun::whole(piece.as_mut_ptr() as usize, piece.len() as u64).unwrap(),
            })
            .collect();
        let out_layout =
            PackedGuestLayout::rows(&out_runs, rows as u32, pitch as u64, row_bytes as u64)
                .unwrap();
        unsafe { out_layout.scatter_from(packed.as_ptr()) };
        let rebuilt: Vec<u8> = out.concat();
        assert_eq!(
            reference_rows(&rebuilt, rows, pitch, row_bytes),
            reference_rows(&window, rows, pitch, row_bytes)
        );
    }

    /// A row byte no run covers is a refusal, not a partial layout — including
    /// a short final row, the case a window cut one byte too early produces.
    #[test]
    fn uncovered_row_bytes_refuse() {
        let buf = vec![0u8; 64];
        // Hole at window bytes 8..10, inside row 1 (8..14 at pitch 8, 6 bytes).
        let runs = [window_run(&buf, 0, 0, 8), window_run(&buf, 10, 10, 30)];
        assert_eq!(PackedGuestLayout::rows(&runs, 3, 8, 6), None);
        // Padding holes are fine: row bytes 0..6, 8..14, 16..22 only.
        let runs = [
            window_run(&buf, 0, 0, 6),
            window_run(&buf, 8, 8, 6),
            window_run(&buf, 16, 16, 6),
        ];
        assert!(PackedGuestLayout::rows(&runs, 3, 8, 6).is_some());
        // The window stops one byte short of the last row's end.
        assert_eq!(
            PackedGuestLayout::rows(&[window_run(&buf, 0, 0, 21)], 3, 8, 6),
            None
        );
    }

    /// Runs out of order or overlapping are malformed input, refused rather
    /// than resolved by guessing which one names the byte.
    #[test]
    fn unordered_or_overlapping_runs_refuse() {
        let buf = vec![0u8; 32];
        let overlap = [window_run(&buf, 0, 0, 10), window_run(&buf, 8, 8, 10)];
        assert_eq!(PackedGuestLayout::rows(&overlap, 1, 18, 18), None);
        let reversed = [window_run(&buf, 10, 10, 10), window_run(&buf, 0, 0, 10)];
        assert_eq!(PackedGuestLayout::rows(&reversed, 1, 20, 20), None);
    }

    /// Degenerate geometry is not a layout.
    #[test]
    fn degenerate_geometry_refuses() {
        let buf = vec![0u8; 16];
        let runs = [window_run(&buf, 0, 0, 16)];
        assert_eq!(PackedGuestLayout::rows(&runs, 0, 4, 4), None);
        assert_eq!(PackedGuestLayout::rows(&runs, 2, 4, 0), None);
        assert_eq!(
            PackedGuestLayout::rows(&runs, 2, 3, 4),
            None,
            "pitch below row"
        );
        assert_eq!(PackedGuestLayout::rows(&runs, u32::MAX, u64::MAX, 4), None);
    }
}
