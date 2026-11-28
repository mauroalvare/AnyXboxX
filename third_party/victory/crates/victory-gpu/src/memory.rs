//! Title memory by address, for translated shaders that read and write
//! through raw 40-bit addresses (`victory_shader::emit::SLOT_MEMORY`: the
//! console's ray-tracing structure builder and the programs that walk its
//! structures; the structures themselves lie in plain title memory, in no
//! buffer object).
//!
//! A 64 KiB page in a title buffer that holds all of it is in that buffer's
//! copy ([`crate::shadow`]), which every other access uses. A page outside
//! buffers has a page of its own in an **arena**: memory of ours with a copy
//! on the card ([`crate::shadow::by_address`]). The streamed structures lie
//! anywhere between other data (6.6 GB of title memory in 4 MiB windows, run
//! light-1; 8 GB in windows of 256 KiB, light-2), so a page is what is kept.
//!
//! This module keeps what a shader needs to find the pages:
//!
//! - the **directory**: per 64 KiB page of title address space the number
//!   of the copy that holds the page (0: none) and the page's byte offset
//!   in it; one for all lists, entries only ever go from empty to filled.
//!   The CPU writes it in memory of its own and shaders read a copy of that
//!   on the card, brought up to date as title memory is (over the bus, every
//!   step of every ray waited for it);
//! - per view heap a run of slots with a raw unordered access view of each
//!   numbered copy (slot 0 of the run: a null view, for pages without a
//!   copy);
//! - the **faults**: the page numbers shaders found empty, read back here
//!   once a frame, so the pages are there from the next frame on. A pointer a
//!   shader takes out of memory cannot be known before it runs.
//!
//! **A page that is not there when a shader touches it is a lost access**:
//! the read gives 0, the store is dropped, and the fault note only helps
//! the next frame. For the structure builder that is the same as a dropped
//! stage, so before one of its dispatches the host makes ready what it can
//! see the dispatch name:
//!
//! - 256 KiB from each root view outside buffers, and where title memory has a
//!   packed ("dehydrated") structure there, all of it: the title ships its
//!   structures packed, and the builder's rehydration stages unpack each in
//!   place once it is streamed in;
//! - 4 MiB from each unordered access view outside buffers (structures built);
//! - the page of each pair of root constants that is a title address and of
//!   each root constant that is one in 256-byte units;
//! - the first page of every structure a list's upload memory names
//!   (a top-level build's instances).
//!
//! **What comes back to title memory**: what the builder stores in title
//! buffers of at most [`BACK`] bytes it names (the sizes the title reads to
//! make a structure smaller). Structures and scratch memory stay on the card
//! (the first whole run of the builder wrote garbage over the title's
//! objects). So an arena's page is the only place a structure is whole, and
//! title memory must not go over it: of a page the title wrote only the
//! dwords that changed since they were last taken go up. A packed structure
//! goes up whole when the builder comes to it, whatever is known of it: the
//! same bytes streamed in again do not differ from those last taken, and the
//! card holds them unpacked (run light-3: a quarter of the structures
//! unpacked a second time, into garbage all over their neighbours).
//!
//! **Stores are bounded**: a structure the builder misreads makes it store
//! anywhere. Pages of title buffers take its stores only once one of its
//! dispatches was given that buffer as an unordered access view (its scratch
//! memory, its results); what else a store reaches is an arena's.
//!
//! A packed structure is gone when title memory has no header of it any more
//! (looked at over [`SWEEP`] frames); the pages it alone has are given back.
//!
//! shortcut: a page held for anything else is kept for the rest of the run.
//! shortcut: pages a list does not name are looked at again in turn, over
//! [`SWEEP`] frames.

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};

use victory_shader::emit::{MEMORY_FAULTS, MEMORY_PAGES, MEMORY_READ_ONLY};
use windows::Win32::Graphics::Direct3D12::*;

use crate::descriptor::host::HostHeap;
use crate::host::{HostList, host};
use crate::shadow::Shadow;

/// Copies that can be numbered (the run of heap slots is one longer).
const COPIES: u32 = (1 << 12) - 1;
/// A directory page.
const PAGE: u64 = 1 << 16;
/// An arena's bytes.
const ARENA: u64 = 64 << 20;
/// Pages in one arena (`64 MiB / 64 KiB = 1024`).
const ARENA_PAGES: usize = (ARENA / PAGE) as usize;
/// `u64` bitmap words covering [`ARENA_PAGES`].
const ARENA_WORDS: usize = ARENA_PAGES / 64;
/// What is compared at a time.
const PIECE: usize = 4096;
/// Pages held, at most (3 GiB).
const MOST: usize = 3 << 14;
/// Pages of title memory with small title buffers in them, at most.
const WINDOWS: usize = 512;
/// The biggest packed structure taken for one.
const PACKED: u64 = 64 << 20;
/// For so many frames what a page holds may still be on its way.
const YOUNG: u64 = 4;
/// Free pages there are, at least, when a list is recorded.
const SPARE: usize = 1024;
/// A packed structure not named for so many frames is another one.
const AGAIN: u64 = 8;
/// From a root view; from one that is written (a structure being built).
const VIEW: u64 = 256 << 10;
const BUILT: u64 = 4 << 20;
/// The biggest title buffer whose builder results come back.
const BACK: u64 = 256 << 10;
/// Pages taken from the faults, of one frame and in all, at most: a ray in a
/// structure that is not right goes anywhere, and 2,048 a frame took all the
/// pages there are in the first seconds (run light-4).
const FAULTS: usize = 128;
const FAULTS_ALL: usize = 2048;

/// Pages taken from the faults in all (`VICTORY_BVH_FAULTS_ALL=<n>`; unset:
/// [`FAULTS_ALL`]). With ray tracing the bound is met in the first minute
/// (run play-1126-0921: at 63 s), and from then on 130-500 pages a frame that
/// the walking programs touch have no copy for the rest of the run.
fn faults_all() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| {
        std::env::var("VICTORY_BVH_FAULTS_ALL").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(FAULTS_ALL)
    })
}
/// Frames over which every page is looked at once.
const SWEEP: usize = 128;
/// Maximum gap (in 64 KiB pages) and span when coalescing nearby pages into a
/// single `watch::writes` query.
const WATCH_GAP: u64 = 64;
const WATCH_SPAN: u64 = 1024;

/// What a draw or dispatch does with title memory by address.
#[derive(Clone, Copy, PartialEq)]
pub(crate) enum Use {
    Reads,
    /// Stores by address, as the structure builder; `back`: its results in
    /// small title buffers come back.
    Builds { back: bool },
}

/// A page of title memory in an arena.
struct Held {
    /// Byte offset inside the arena's backing buffer (`0..ARENA`, 64 KiB aligned),
    /// and the arena's place in `copies`.
    offset: usize,
    copy: usize,
    /// The write-watch asking after which it was last taken (0: never).
    taken: u64,
    /// The frame it was last looked at in, and the one it came in.
    looked: u64,
    born: u64,
    /// Packed structures that have bytes in it, and whether it is held for
    /// them alone.
    owners: u32,
    theirs: bool,
}

/// A packed structure the builder came to.
struct Packed {
    /// The frame it last did.
    frame: u64,
    len: u64,
    /// Its first two dwords in title memory.
    header: u64,
    /// The write-watch asking at which `header` was last confirmed (0: never).
    checked: u64,
}

/// One 64 MiB arena and its per-page free bitmap.
/// Bit `k` of `free_bits[k / 64]` is 1 when slot `k` (`base + k * PAGE`) is free.
struct Arena {
    copy: usize,
    base: usize,
    free_bits: [u64; ARENA_WORDS],
    free_count: usize,
}

impl Arena {
    fn new(copy: usize, base: usize) -> Self {
        Self {
            copy,
            base,
            free_bits: [u64::MAX; ARENA_WORDS],
            free_count: ARENA_PAGES,
        }
    }

    fn take_one(&mut self) -> Option<usize> {
        if self.free_count == 0 {
            return None;
        }
        for (wi, w) in self.free_bits.iter_mut().enumerate() {
            if *w != 0 {
                let bit = w.trailing_zeros() as usize;
                *w &= !(1u64 << bit);
                self.free_count -= 1;
                return Some(self.base + (wi * 64 + bit) * PAGE as usize);
            }
        }
        None
    }

    fn clear_run(&mut self, start: usize, len: usize) {
        let end = start + len;
        let mut slot = start;
        while slot < end {
            let wi = slot >> 6;
            let bit = slot & 63;
            let take = (64 - bit).min(end - slot);
            let mask = if take == 64 { u64::MAX } else { ((1u64 << take) - 1) << bit };
            debug_assert_eq!(self.free_bits[wi] & mask, mask);
            self.free_bits[wi] &= !mask;
            slot += take;
        }
        self.free_count -= len;
    }

    /// Allocates the lowest contiguous run of `need` free pages in this arena.
    fn take_run(&mut self, need: usize) -> Option<usize> {
        if need == 0 || need > self.free_count {
            return None;
        }
        if need == 1 {
            return self.take_one();
        }
        let mut run_start = 0usize;
        let mut run_len = 0usize;
        for (wi, &word) in self.free_bits.iter().enumerate() {
            if word == 0 {
                run_len = 0;
                continue;
            }
            if word == u64::MAX {
                if run_len == 0 {
                    run_start = wi * 64;
                }
                run_len += 64;
                if run_len >= need {
                    self.clear_run(run_start, need);
                    return Some(self.base + run_start * PAGE as usize);
                }
                continue;
            }
            let mut bit = 0usize;
            while bit < 64 {
                let shifted = word >> bit;
                if shifted == 0 {
                    run_len = 0;
                    break;
                }
                if shifted & 1 == 0 {
                    run_len = 0;
                    bit += shifted.trailing_zeros() as usize;
                    continue;
                }
                let ones = shifted.trailing_ones() as usize;
                if run_len == 0 {
                    run_start = wi * 64 + bit;
                }
                run_len += ones;
                if run_len >= need {
                    self.clear_run(run_start, need);
                    return Some(self.base + run_start * PAGE as usize);
                }
                bit += ones;
            }
        }
        None
    }

    /// Returns `at` (which belongs to this arena) back to the free bitmap.
    fn give_back(&mut self, at: usize) -> bool {
        let slot = (at - self.base) / PAGE as usize;
        debug_assert!(slot < ARENA_PAGES);
        let mask = 1u64 << (slot & 63);
        if self.free_bits[slot >> 6] & mask == 0 {
            self.free_bits[slot >> 6] |= mask;
            self.free_count += 1;
            true
        } else {
            false
        }
    }
}

struct Memory {
    /// The CPU's entries (number, offset) directly in the directory's
    /// `L0 WRITE_BACK` staging memory, and the copy shaders read.
    directory: (*mut u32, Arc<Shadow>),
    faults: (ID3D12Resource, *mut u32, u64),
    /// Copy `n` is `copies[n - 1]`.
    copies: Vec<Arc<Shadow>>,
    /// Places of title buffers' copies among `copies`.
    numbers: HashMap<usize, usize>,
    /// Arenas in creation order; allocations prefer lower arenas first so the
    /// active working set packs densely into the fewest physical arenas.
    arenas: Vec<Arena>,
    /// Total free pages across `arenas`.
    free_pages: usize,
    /// Reclaimed arena pages waiting for in-flight GPU submissions to finish
    /// before returning to their arena bitmap: `(retire_gate, copy, at)`.
    retiring: VecDeque<(crate::host::BvhRetire, usize, usize)>,
    /// Non-arena copies (title buffers and small-buffer windows) in `copies`
    /// that mirror title memory and need `Shadow::refresh`.
    title_copies: Vec<usize>,
    /// By title page.
    pages: HashMap<u64, Held>,
    /// Their numbers in the order they came, and where the sweep is.
    order: Vec<u64>,
    sweep: usize,
    /// Windows of title memory itself ([`Memory::map`]).
    windows: usize,
    /// Packed structures the builder came to: the frame it last did.
    packed: HashMap<u64, Packed>,
    /// Their addresses in the order they came, and where the sweep is.
    packed_order: Vec<u64>,
    packed_sweep: usize,
    /// Copies of title buffers the builder may store in.
    open: std::collections::HashSet<usize>,
    /// Pages taken from faults.
    faulted: usize,
    /// Per host view heap (by address): the first slot of its run, and how
    /// many copies have their view there.
    heaps: HashMap<usize, (u32, u32)>,
    /// The frame the faults were last read in.
    frame: u64,
    /// Pages asked for this frame that are not the title's memory.
    refused: std::collections::HashSet<u64>,
    /// Reusable scratch buffers for coalesced write-watch queries.
    scratch_pages: Vec<u64>,
    scratch_watch: Vec<(u64, Option<(u64, u64)>)>,
    stats: Stats,
    #[cfg(feature = "diagnostics")]
    dump: dump::Dump,
    #[cfg(feature = "diagnostics")]
    trace: trace::Trace,
}

/// Since the last report.
#[derive(Default)]
struct Stats {
    from: Option<(std::time::Instant, u64)>,
    uses: [u64; 2],
    faults: u64,
    unknown: u64,
    back: u64,
    /// Pages looked at, bytes that changed, packed structures sent.
    looked: u64,
    changed: u64,
    packed: u64,
    packed_bytes: u64,
    gone: u64,
    /// Pages that came for an address in constants, a written view, another
    /// view, a structure named in upload memory.
    came: [u64; 4],
    micros: u64,
    /// Pages found without a copy that were left so because of the bounds on
    /// pages taken from faults, and the different pages faulted.
    left: u64,
    distinct: std::collections::HashSet<u64>,
}

// SAFETY: the mappings are of memory the struct owns; all access is under
// the mutex.
unsafe impl Send for Memory {}

static MEMORY: Mutex<Option<Memory>> = Mutex::new(None);

/// The synthetic titles share the same GPU memory manager. Each needs a fresh
/// directory because Windows can reuse the virtual addresses from a prior test.
#[cfg(test)]
fn fresh_test_memory() -> std::sync::MutexGuard<'static, ()> {
    static TEST: Mutex<()> = Mutex::new(());
    let guard = TEST.lock().unwrap_or_else(|e| e.into_inner());
    *MEMORY.lock().unwrap_or_else(|e| e.into_inner()) = None;
    guard
}

/// What a draw binds.
pub(crate) struct Ready {
    pub(crate) directory: u64,
    pub(crate) faults: u64,
    /// The view heap slot of the null view; copy `n`'s view is `n` further.
    pub(crate) base: u32,
}

impl Memory {
    fn new() -> Option<Memory> {
        let host = host()?;
        let bytes = u64::from(MEMORY_PAGES) * 8;
        // Directly backed by L0 WRITE_BACK staging + card VRAM (no duplicate
        // write-watched host mirror). Zeroed once on creation; every entry
        // update marks only its 8 bytes dirty on the card.
        let directory = crate::shadow::by_address_owned(bytes)?;
        let entries = directory.data_mut();
        // SAFETY: `entries` maps `bytes` of writable CPU-cached staging memory.
        unsafe { std::ptr::write_bytes(entries, 0, bytes as usize) };
        crate::host::name(&directory.buffer, || "title memory by address: directory".into());
        let (faults, data) = host.shared_buffer(u64::from(MEMORY_FAULTS) * 4)?;
        crate::host::name(&faults, || "title memory by address: faults".into());
        // SAFETY: a fresh mapping of at least that many bytes.
        unsafe { std::ptr::write_bytes(data, 0, MEMORY_FAULTS as usize * 4) };
        // SAFETY: a live buffer.
        let gpu = unsafe { faults.GetGPUVirtualAddress() };
        tracing::info!("title memory by address: a shader reads or writes through raw addresses");
        Some(Memory {
            directory: (entries.cast::<u32>(), directory),
            faults: (faults, data.cast::<u32>(), gpu),
            copies: Vec::new(),
            numbers: HashMap::new(),
            arenas: Vec::new(),
            free_pages: 0,
            retiring: VecDeque::new(),
            title_copies: Vec::new(),
            pages: HashMap::new(),
            order: Vec::new(),
            sweep: 0,
            windows: 0,
            packed: HashMap::new(),
            packed_order: Vec::new(),
            packed_sweep: 0,
            open: Default::default(),
            faulted: 0,
            heaps: HashMap::new(),
            frame: u64::MAX,
            refused: Default::default(),
            scratch_pages: Vec::new(),
            scratch_watch: Vec::new(),
            stats: Stats::default(),
            #[cfg(feature = "diagnostics")]
            dump: Default::default(),
            #[cfg(feature = "diagnostics")]
            trace: Default::default(),
        })
    }

    /// The directory's entry of `page`.
    fn entry(&self, page: u64) -> *mut u32 {
        debug_assert!(page < u64::from(MEMORY_PAGES));
        // SAFETY: inside the directory.
        unsafe { self.directory.0.add(page as usize * 2) }
    }

    /// Marks directory entries `[first, last]` (inclusive) dirty for upload.
    fn mark_directory(&self, first: u64, last: u64) {
        if first <= last {
            self.directory.1.mark_card_dirty(first as usize * 8, (last as usize + 1) * 8);
        }
    }

    /// Writes a writable directory entry for `page` and marks its 8 bytes dirty.
    fn set_entry(&self, page: u64, n: usize, offset: u64) {
        fill(self.entry(page), n, offset);
        self.mark_directory(page, page);
    }

    /// Writes a read-only directory entry for `page` and marks its 8 bytes dirty.
    fn close_entry(&self, page: u64, n: usize, offset: u64) {
        close(self.entry(page), n, offset);
        self.mark_directory(page, page);
    }

    /// The place in `copies` of the copy that holds `page`, if one does.
    fn copy_of(&self, page: u64) -> Option<usize> {
        if page >= u64::from(MEMORY_PAGES) {
            return None;
        }
        // SAFETY: an entry of the directory.
        ((unsafe { self.entry(page).read() } & !MEMORY_READ_ONLY) as usize).checked_sub(1)
    }

    /// Whether `page` can be given a place: in the directory, not turned down
    /// this frame, and the title's GPU memory. (A shader's pointer, or a
    /// constant taken for an address, may be anything.)
    fn possible(&mut self, page: u64) -> bool {
        // Not the last entry: addresses past the directory read it, and it
        // stays empty.
        if page < 0x1_0000 || page >= u64::from(MEMORY_PAGES) - 1 || self.refused.contains(&page) {
            return false;
        }
        // Pages already mapped in the directory (matching `map()`'s early
        // return) skip redundant `VirtualQuery` + `GetWriteWatch` revalidation
        // during dispatch recording: `bring_pages()` reads via SEH-safe
        // `copy_mapped`, `land()` validates `committed()` on any dirty/unknown
        // header, and `look_again()` still calls `there()` (which checks
        // `committed()`) so pages decommitted without a write are reclaimed.
        if self.copy_of(page).is_some() {
            return true;
        }
        let at = (page * PAGE) as usize;
        let ours = crate::texture::committed(at as *const u8, PAGE as usize)
            && crate::watch::writes(at, at + PAGE as usize, true).is_some();
        if !ours {
            self.refused.insert(page);
            self.stats.unknown += 1;
        }
        ours
    }

    /// Gives a copy a number, if one is left.
    fn number(&mut self, shadow: Arc<Shadow>, owned_arena: bool) -> Option<usize> {
        let key = Arc::as_ptr(&shadow) as usize;
        if let Some(&n) = self.numbers.get(&key) {
            return Some(n);
        }
        if self.copies.len() >= COPIES as usize {
            static ONCE: std::sync::Once = std::sync::Once::new();
            ONCE.call_once(|| tracing::warn!(COPIES, "title memory by address: no number left for another copy; its pages read zeros"));
            return None;
        }
        // SAFETY: a live buffer.
        let flags = unsafe { shadow.buffer.GetDesc() }.Flags;
        if flags.0 & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS.0 == 0 {
            return None;
        }
        self.copies.push(shadow);
        let n = self.copies.len() - 1;
        self.numbers.insert(key, n);
        if !owned_arena {
            self.title_copies.push(n);
        }
        Some(n)
    }

    /// Gives `page` a place, if it has none and the memory is the title's;
    /// whether it has one then.
    fn map(&mut self, page: u64) -> bool {
        if self.copy_of(page).is_some() {
            return true;
        }
        if !self.possible(page) {
            return false;
        }
        let start = page * PAGE;
        if crate::shadow::in_buffer(start, PAGE).is_some() {
            let Some(shadow) = crate::shadow::find(start, PAGE) else {
                return false;
            };
            let fresh = !self.numbers.contains_key(&(Arc::as_ptr(&shadow) as usize));
            let Some(n) = self.number(shadow.clone(), false) else {
                return false;
            };
            // Every page the copy holds whole, not only the one asked for.
            // shortcut: a page of a title buffer inside this one that has a
            // copy of its own is given this copy too.
            if fresh {
                let (first, len) = shadow.span();
                let mut at = first.next_multiple_of(PAGE);
                let mut dirty: Option<(u64, u64)> = None;
                while at + PAGE <= first + len && at / PAGE < u64::from(MEMORY_PAGES) - 1 {
                    let p = at / PAGE;
                    let entry = self.entry(p);
                    // SAFETY: an entry of the directory.
                    if unsafe { entry.read() } == 0 {
                        close(entry, n, at - first);
                        dirty = Some(match dirty {
                            Some((lo, hi)) => (lo.min(p), hi.max(p)),
                            None => (p, p),
                        });
                    }
                    at += PAGE;
                }
                if let Some((lo, hi)) = dirty {
                    self.mark_directory(lo, hi);
                }
            }
            // SAFETY: as above.
            if unsafe { self.entry(page).read() } == 0 {
                self.close_entry(page, n, start - shadow.span().0);
            }
            return true;
        }
        // Small title buffers in it: what the builder stores in them comes
        // back, so title memory itself.
        if self.windows < WINDOWS && crate::shadow::small_buffer_in(start, PAGE, BACK) {
            if let Some(shadow) = crate::shadow::by_address(start, PAGE)
                && let Some(n) = self.number(shadow, false)
            {
                self.windows += 1;
                self.set_entry(page, n, 0);
                return true;
            }
            return false;
        }
        if self.pages.len() >= MOST {
            static ONCE: std::sync::Once = std::sync::Once::new();
            ONCE.call_once(|| tracing::warn!(MOST, "title memory by address: no more pages are taken; others read zeros"));
            return false;
        }
        let Some((copy, offset)) = self.alloc_one(true) else {
            return false;
        };
        self.hold(page, offset, copy);
        true
    }

    /// Allocates the lowest free page from the lowest-numbered arena with space,
    /// optionally creating a new arena when all existing arenas are full.
    fn alloc_one(&mut self, allow_new: bool) -> Option<(usize, usize)> {
        self.reclaim_retired();
        for arena in &mut self.arenas {
            if let Some(at) = arena.take_one() {
                self.free_pages -= 1;
                return Some((arena.copy, at));
            }
        }
        if allow_new && self.another() {
            let arena = self.arenas.last_mut()?;
            if let Some(at) = arena.take_one() {
                self.free_pages -= 1;
                return Some((arena.copy, at));
            }
        }
        None
    }

    /// Allocates a contiguous run of `need` pages in the lowest-numbered arena
    /// that fits; falls back to a new arena only if `allow_new` and total free
    /// pages across existing arenas are fewer than `need`.
    fn alloc_run(&mut self, need: usize, allow_new: bool) -> Option<(usize, usize)> {
        if need == 0 || need > ARENA_PAGES {
            return None;
        }
        self.reclaim_retired();
        for arena in &mut self.arenas {
            if let Some(at) = arena.take_run(need) {
                self.free_pages -= need;
                return Some((arena.copy, at));
            }
        }
        if allow_new
            && self.free_pages < need
            && self.pages.len() + self.free_pages + self.retiring.len() < MOST
            && self.another()
        {
            let arena = self.arenas.last_mut()?;
            if let Some(at) = arena.take_run(need) {
                self.free_pages -= need;
                return Some((arena.copy, at));
            }
        }
        None
    }

    /// Allocates `need` pages from the lowest single arena that holds at least
    /// `need` free slots (even if non-contiguous), keeping all pages under one
    /// descriptor.
    fn alloc_single_arena(&mut self, need: usize) -> Option<(usize, Vec<usize>)> {
        if need == 0 || need > ARENA_PAGES {
            return None;
        }
        for arena in &mut self.arenas {
            if arena.free_count >= need {
                let mut addrs = Vec::with_capacity(need);
                for _ in 0..need {
                    addrs.push(arena.take_one()?);
                }
                self.free_pages -= need;
                return Some((arena.copy, addrs));
            }
        }
        None
    }

    /// Returns a freed page back to its owning arena's free bitmap.
    fn give_back(&mut self, copy: usize, at: usize) {
        if let Some(arena) = self.arenas.iter_mut().find(|a| a.copy == copy)
            && arena.give_back(at)
        {
            self.free_pages += 1;
        }
    }

    /// Moves retired arena pages whose GPU submissions have completed back
    /// into their owning arenas' free bitmaps.
    fn reclaim_retired(&mut self) {
        while let Some(front) = self.retiring.front_mut() {
            if !crate::host::bvh_retire_ready(&mut front.0) {
                break;
            }
            let (_, copy, at) = self.retiring.pop_front().unwrap();
            self.give_back(copy, at);
        }
    }

    /// Maps pages `first..=last` (stopping at the first page not belonging to
    /// the title), preferring a contiguous run in the lowest arena when
    /// multiple unmapped pure-arena pages are requested together.
    fn map_contiguous(&mut self, first: u64, last: u64, mark_theirs: bool) -> u64 {
        let count = (last.saturating_sub(first) + 1) as usize;
        if count > 1 && count <= ARENA_PAGES && self.pages.len() + count <= MOST {
            let mut unmapped: Vec<u64> = Vec::with_capacity(count);
            let mut all_arena = true;
            for page in first..=last {
                if self.copy_of(page).is_some() {
                    continue;
                }
                if !self.possible(page) {
                    all_arena = false;
                    break;
                }
                let start = page * PAGE;
                if crate::shadow::in_buffer(start, PAGE).is_some()
                    || (self.windows < WINDOWS && crate::shadow::small_buffer_in(start, PAGE, BACK))
                {
                    all_arena = false;
                    break;
                }
                unmapped.push(page);
            }
            if all_arena && unmapped.len() > 1 {
                if let Some((copy, base)) = self.alloc_run(unmapped.len(), true) {
                    for (k, page) in unmapped.into_iter().enumerate() {
                        self.hold(page, base + k * PAGE as usize, copy);
                        if mark_theirs && let Some(held) = self.pages.get_mut(&page) {
                            held.theirs = true;
                        }
                    }
                    return last + 1;
                }
                if let Some((copy, addrs)) = self.alloc_single_arena(unmapped.len()) {
                    for (page, at) in unmapped.into_iter().zip(addrs) {
                        self.hold(page, at, copy);
                        if mark_theirs && let Some(held) = self.pages.get_mut(&page) {
                            held.theirs = true;
                        }
                    }
                    return last + 1;
                }
            }
        }
        let mut end = first;
        while end <= last {
            let new = self.copy_of(end).is_none();
            if !self.map(end) {
                break;
            }
            if mark_theirs && new && let Some(held) = self.pages.get_mut(&end) {
                held.theirs = true;
            }
            end += 1;
        }
        end
    }

    /// Another arena's pages.
    fn another(&mut self) -> bool {
        let Some(shadow) = crate::shadow::by_address_owned(ARENA) else {
            return false;
        };
        let Some(n) = self.number(shadow, true) else {
            return false;
        };
        self.arenas.push(Arena::new(n, 0));
        self.free_pages += ARENA_PAGES;
        true
    }

    /// `page` is held at byte `offset` in the arena that is `copy`.
    fn hold(&mut self, page: u64, offset: usize, copy: usize) {
        self.pages.insert(page, Held { offset, copy, taken: 0, looked: u64::MAX, born: self.frame, owners: 0, theirs: false });
        self.order.push(page);
        self.set_entry(page, copy, offset as u64);
    }

    /// The packed structure at `address` is gone: the pages it alone had are
    /// invalidated in the directory immediately and fence-retired until every
    /// in-flight or queued GPU submission completes before returning to their
    /// arena bitmap.
    fn gone(&mut self, address: u64) {
        let Some(packed) = self.packed.remove(&address) else { return };
        let mut target = crate::host::bvh_retire_target();
        let ready_now = crate::host::bvh_retire_ready(&mut target);
        let mut cleared: Option<(u64, u64)> = None;
        for page in address / PAGE..=(address + packed.len - 1) / PAGE {
            let Some(held) = self.pages.get_mut(&page) else { continue };
            held.owners = held.owners.saturating_sub(1);
            if held.owners == 0 && held.theirs {
                let (copy, offset) = (held.copy, held.offset);
                self.pages.remove(&page);
                let entry = self.entry(page);
                // SAFETY: an entry of the directory, under the mutex. Clear
                // number first so any concurrent observer stops using the slot.
                unsafe {
                    entry.write_volatile(0);
                    entry.add(1).write_volatile(0);
                }
                cleared = Some(match cleared {
                    Some((lo, hi)) => (lo.min(page), hi.max(page)),
                    None => (page, page),
                });
                if ready_now {
                    self.give_back(copy, offset);
                } else {
                    self.retiring.push_back((target, copy, offset));
                }
            }
        }
        if let Some((lo, hi)) = cleared {
            self.mark_directory(lo, hi);
        }
        self.stats.gone += 1;
    }

    /// Whether the packed structure at `address` is still in title memory.
    fn there(&self, address: u64) -> bool {
        let mut header = [0u8; 8];
        crate::texture::committed(address as *const u8, 8) && {
            crate::host::copy_mapped(header.as_mut_ptr(), address as *const u8, 8, false);
            self.packed.get(&address).is_some_and(|p| p.header == u64::from_le_bytes(header))
        }
    }

    /// Looks at some of the pages and packed structures again.
    fn look_again(&mut self) {
        let mut pages = std::mem::take(&mut self.scratch_pages);
        pages.clear();
        for _ in 0..self.order.len().div_ceil(SWEEP) {
            self.sweep = (self.sweep + 1) % self.order.len();
            let page = self.order[self.sweep];
            if self.pages.contains_key(&page) {
                pages.push(page);
            } else {
                self.order.swap_remove(self.sweep);
                if self.order.is_empty() {
                    break;
                }
            }
        }
        self.scratch_pages = self.bring_pages(pages);

        // Keep `there()` (`VirtualQuery` + header check) on the slow `SWEEP`
        // cadence (~1/128 packed entries per frame) so a packed page
        // decommitted with `VirtualFree(..., MEM_DECOMMIT)` without any CPU
        // write (which `GetWriteWatch` does not report as dirty on a still-
        // reserved `MEM_WRITE_WATCH` VAD) is still detected and reclaimed.
        for _ in 0..self.packed_order.len().div_ceil(SWEEP) {
            self.packed_sweep = (self.packed_sweep + 1) % self.packed_order.len();
            let address = self.packed_order[self.packed_sweep];
            if !self.packed.contains_key(&address) {
                self.packed_order.swap_remove(self.packed_sweep);
                if self.packed_order.is_empty() {
                    break;
                }
            } else if !self.there(address) {
                self.gone(address);
            }
        }
    }

    /// A dispatch of the builder was given `address` to write: if it is in a
    /// title buffer's copy, stores by address reach that from here on.
    fn opens(&mut self, address: u64) {
        let Some(n) = self.map(address / PAGE).then(|| self.copy_of(address / PAGE)).flatten() else {
            return;
        };
        if self.pages.contains_key(&(address / PAGE)) || !self.open.insert(n) {
            return;
        }
        // (Its pages hold what the CPU writes beside the builder's scratch.)
        self.copies[n].keep_stores();
        let (first, len) = self.copies[n].span();
        tracing::info!(
            start = format_args!("{first:#x}"),
            len,
            "title memory by address: the structure builder may store in a title buffer"
        );
        let mut dirty: Option<(u64, u64)> = None;
        for page in first.div_ceil(PAGE)..(first + len) / PAGE {
            if self.copy_of(page) == Some(n) {
                fill(self.entry(page), n, page * PAGE - first);
                dirty = Some(match dirty {
                    Some((lo, hi)) => (lo.min(page), hi.max(page)),
                    None => (page, page),
                });
            }
        }
        if let Some((lo, hi)) = dirty {
            self.mark_directory(lo, hi);
        }
    }

    /// [`Memory::map`] of the pages of `[address, address + len)`, up to the
    /// first that is not the title's, pushing any arena pages not yet checked
    /// this frame into `pending` for a single coalesced [`Memory::bring_pages`].
    fn map_range(&mut self, address: u64, len: u64, pending: &mut Vec<u64>) {
        if len == 0 {
            return;
        }
        let first = address / PAGE;
        let last = (address + len.saturating_sub(1)) / PAGE;
        let end = self.map_contiguous(first, last, false);
        let frame = self.frame;
        pending.extend((first..end).filter(|page| self.pages.get(page).is_some_and(|h| h.looked != frame)));
    }

    /// Takes the arena pages in `pages` from title memory where it wrote them,
    /// sorting, deduplicating, and coalescing nearby pages into batched
    /// `watch::writes` queries.
    fn bring_pages(&mut self, mut pages: Vec<u64>) -> Vec<u64> {
        if pages.is_empty() {
            return pages;
        }
        pages.sort_unstable();
        pages.dedup();
        let frame = self.frame;
        pages.retain(|page| self.pages.get(page).is_some_and(|h| h.looked != frame));
        if pages.is_empty() {
            return pages;
        }
        let started = std::time::Instant::now();
        let mut watch = std::mem::take(&mut self.scratch_watch);
        query_page_writes(&pages, &mut watch);
        let mut stale_pages = Vec::new();
        let mut part_queries = Vec::new();
        for &(page, answer) in &watch {
            let Some(held) = self.pages.get_mut(&page) else {
                continue;
            };
            held.looked = frame;
            self.stats.looked += 1;
            let prev = held.taken;
            if prev != 0 && answer.is_some_and(|(_, wrote)| wrote <= prev) {
                continue;
            }
            held.taken = answer.map_or(1, |(now, _)| now.max(1));
            let parts = if prev != 0 && answer.is_some() {
                let index = part_queries.len();
                part_queries.push((page as usize, prev));
                Some(index)
            } else {
                None
            };
            stale_pages.push((page, prev == 0, parts));
        }

        let masks = if part_queries.is_empty() {
            Vec::new()
        } else {
            crate::watch::parts_written(&part_queries)
        };
        let mut title = [0u8; PIECE];
        for (page, fresh, parts) in stale_pages {
            let mask = parts.map_or(u16::MAX, |i| masks[i]);
            if mask == 0 {
                continue;
            }
            let held = &self.pages[&page];
            let shadow = &self.copies[held.copy];
            let data = shadow.data_mut();
            let offset = held.offset;
            let mut dirty = Vec::new();
            for piece in (0..16).filter(|i| (mask >> i) & 1 != 0).map(|i| i * PIECE) {
                let from = (page * PAGE) as usize + piece;
                crate::host::copy_mapped(title.as_mut_ptr(), from as *const u8, PIECE, false);
                // SAFETY: inside the arena staging mapping, under the memory mutex.
                let ours = unsafe { std::slice::from_raw_parts_mut(data.add(offset + piece), PIECE) };
                if fresh {
                    ours.copy_from_slice(&title);
                } else {
                    for (at, len) in crate::shadow::word_diff_runs(piece, &title, ours) {
                        ours[at..at + len].copy_from_slice(&title[at..at + len]);
                        self.stats.changed += len as u64;
                        dirty.push((offset + piece + at, offset + piece + at + len));
                    }
                }
            }
            if fresh {
                shadow.mark_card_dirty(offset, offset + PAGE as usize);
                self.stats.changed += PAGE;
            } else if !dirty.is_empty() {
                shadow.mark_card_dirty_runs(dirty.into_iter());
            }
        }
        watch.clear();
        self.scratch_watch = watch;
        self.stats.micros += started.elapsed().as_micros() as u64;
        pages.clear();
        pages
    }

    /// The bytes of a packed structure title memory has at `address` by now,
    /// and its first two dwords.
    fn packed_at(&self, address: u64) -> Option<(u64, u64)> {
        if address & 0xFF != 0 {
            return None;
        }
        let mut header = [0u8; 8];
        crate::host::copy_mapped(header.as_mut_ptr(), address as *const u8, 8, false);
        let header = u64::from_le_bytes(header);
        Some((packed_len(header)?, header))
    }

    /// Batched submission-time [`Memory::land`] for all builder addresses of a
    /// dispatch: checks header pages with coalesced write-watch queries so
    /// unchanged known packed structures within [`AGAIN`] skip `VirtualQuery`
    /// and `copy_mapped` completely, while any written or replaced header runs
    /// the full `land` path.
    fn land_batch(&mut self, addrs: &[u64]) {
        let mut header_pages = std::mem::take(&mut self.scratch_pages);
        header_pages.clear();
        header_pages.extend(addrs.iter().map(|&a| a / PAGE));
        header_pages.sort_unstable();
        header_pages.dedup();
        let mut watch = std::mem::take(&mut self.scratch_watch);
        query_page_writes(&header_pages, &mut watch);
        header_pages.clear();
        self.scratch_pages = header_pages;

        for &address in addrs {
            let answer = lookup_watch(&watch, address / PAGE);
            if let Some((now, wrote)) = answer
                && let Some(known) = self.packed.get_mut(&address)
                && known.frame + AGAIN >= self.frame
                && known.checked != 0
                && wrote <= known.checked
            {
                known.frame = self.frame;
                known.checked = now;
                continue;
            }
            self.land(address, answer.map_or(0, |(now, _)| now));
        }

        watch.clear();
        self.scratch_watch = watch;
    }

    /// At the submission of a list whose builder dispatch was given `address`
    /// outside buffers, in the order of its copies: what title memory has
    /// there now goes up. When the dispatch is recorded it may not be there
    /// yet: the title's queue waits for its streaming. `asking` is the
    /// write-watch asking taken immediately before reading `address` (0 if
    /// unwatched).
    fn land(&mut self, address: u64, asking: u64) {
        let page = address / PAGE;
        if !crate::texture::committed((page * PAGE) as *const u8, PAGE as usize) {
            return;
        }
        let Some((len, header)) = self.packed_at(address) else {
            // Anything else: what came since its page was first taken.
            for page in page..=(address + VIEW - 1) / PAGE {
                if self.pages.get(&page).is_some_and(|h| h.born + YOUNG >= self.frame) {
                    self.take(page, 0, PAGE as usize, false);
                }
            }
            return;
        };
        if let Some(known) = self.packed.get_mut(&address) {
            let soon = known.frame + AGAIN >= self.frame;
            known.frame = self.frame;
            if known.header == header && soon {
                known.checked = asking;
                return;
            }
            if known.header != header {
                self.gone(address);
            } else {
                known.checked = asking;
            }
        }
        let last = (address + len - 1) / PAGE;
        if !self.packed.contains_key(&address) {
            self.packed.insert(address, Packed { frame: self.frame, len, header, checked: asking });
            self.packed_order.push(address);
            self.place_packed_pages(page, last);
            for p in page..=last {
                let from = (address.max(p * PAGE) - p * PAGE) as usize;
                let to = ((address + len).min((p + 1) * PAGE) - p * PAGE) as usize;
                let need_margins = if let Some(held) = self.pages.get_mut(&p) {
                    let just_born = held.owners == 0 && held.born == self.frame;
                    held.owners += 1;
                    held.theirs |= just_born;
                    let init = held.taken == 0;
                    if init {
                        held.taken = 1;
                    }
                    init
                } else {
                    false
                };
                if need_margins {
                    if from > 0 {
                        self.take(p, 0, from, true);
                    }
                    if to < PAGE as usize {
                        self.take(p, to, PAGE as usize, true);
                    }
                }
            }
            self.stats.packed_bytes += len;
        }
        for p in page..=last {
            let from = address.max(p * PAGE) - p * PAGE;
            let to = (address + len).min((p + 1) * PAGE) - p * PAGE;
            self.take(p, from as usize, to as usize, true);
        }
        self.stats.packed += 1;
    }

    /// Finishes placing `page` into an already-allocated arena slot `(copy, at)`
    /// at submission time (shared by [`Memory::place`] and
    /// [`Memory::place_packed_pages`]).
    fn finish_placed_page(&mut self, page: u64, copy: usize, offset: usize) {
        self.hold(page, offset, copy);
        // Capture the write stamp so a later refresh preserves builder output.
        let now = place_taken()
            .then(|| crate::watch::writes((page * PAGE) as usize, ((page + 1) * PAGE) as usize, true))
            .flatten()
            .map(|(now, _)| now);
        self.take(page, 0, PAGE as usize, true);
        if let Some(now) = now
            && let Some(held) = self.pages.get_mut(&page)
        {
            held.taken = now.max(1);
        }
    }

    /// Places newly streamed pages in a contiguous run when possible.
    fn place_packed_pages(&mut self, first: u64, last: u64) {
        let count = (last.saturating_sub(first) + 1) as usize;
        if count > 1 && count <= ARENA_PAGES {
            let unmapped: Vec<u64> = (first..=last)
                .filter(|&page| self.copy_of(page).is_none() && Self::can_place(page))
                .collect();
            if unmapped.len() > 1 && self.pages.len() + unmapped.len() <= MOST
                && let Some((copy, offset)) = self.alloc_run(unmapped.len(), false)
            {
                for (i, page) in unmapped.into_iter().enumerate() {
                    self.finish_placed_page(page, copy, offset + i * PAGE as usize);
                }
                return;
            }
        }
        for page in first..=last {
            if self.copy_of(page).is_none() {
                self.place(page);
            }
        }
    }

    fn can_place(page: u64) -> bool {
        page >= 0x1_0000
            && page < u64::from(MEMORY_PAGES) - 1
            && crate::texture::committed((page * PAGE) as *const u8, PAGE as usize)
            && crate::shadow::in_buffer(page * PAGE, PAGE).is_none()
    }

    /// A place in an arena for a page of committed title memory, at a
    /// submission: only one there is already (the list keeps the arenas it
    /// knew when it was recorded).
    fn place(&mut self, page: u64) -> bool {
        if self.pages.len() >= MOST || !Self::can_place(page) {
            return false;
        }
        let Some((copy, at)) = self.alloc_one(false) else {
            return false;
        };
        self.finish_placed_page(page, copy, at);
        true
    }

    /// Bytes `from..to` of `page` from title memory directly into its arena's
    /// staging page, at a submission; `whole`: sent to the card whether they
    /// differ from staging or not (the card may hold an unpacked version).
    fn take(&mut self, page: u64, from: usize, to: usize, whole: bool) {
        if from >= to {
            return;
        }
        let Some(held) = self.pages.get(&page) else { return };
        let shadow = &self.copies[held.copy];
        let base = held.offset;
        let data = shadow.data_mut();
        let mut title = [0u8; PIECE];
        let mut dirty_runs: Vec<(usize, usize)> = Vec::new();
        let mut at = from;
        while at < to {
            let n = (PIECE - at % PIECE).min(to - at);
            crate::host::copy_mapped(title.as_mut_ptr(), (page as usize * PAGE as usize + at) as *const u8, n, false);
            // SAFETY: inside the arena's L0 WRITE_BACK staging mapping, under the mutex.
            let ours = unsafe { std::slice::from_raw_parts_mut(data.add(base + at), n) };
            if whole {
                if *ours != title[..n] {
                    ours.copy_from_slice(&title[..n]);
                }
            } else {
                let runs = crate::shadow::word_diff_runs(at, &title[..n], ours);
                for &(a, len) in &runs {
                    ours[a..a + len].copy_from_slice(&title[a..a + len]);
                    dirty_runs.push((base + at + a, base + at + a + len));
                }
            }
            at += n;
        }
        if whole {
            shadow.mark_card_dirty(base + from, base + to);
        } else if !dirty_runs.is_empty() {
            shadow.mark_card_dirty_runs(dirty_runs.into_iter());
        }
    }

    /// The pages shaders found empty since the last frame.
    fn take_faults(&mut self) {
        let mut pages = Vec::new();
        for i in 1..MEMORY_FAULTS as usize {
            // SAFETY: inside the faults' mapping; the GPU may be writing,
            // a number missed now is written again next frame.
            let at = unsafe { self.faults.1.add(i) };
            // SAFETY: as above.
            let page = unsafe { at.read_volatile() };
            if page != 0 {
                pages.push(u64::from(page));
                // SAFETY: as above.
                unsafe { at.write_volatile(0) };
            }
        }
        if pages.is_empty() {
            return;
        }
        pages.sort_unstable();
        pages.dedup();
        let mut made = 0;
        let all = faults_all();
        let mut bring = std::mem::take(&mut self.scratch_pages);
        bring.clear();
        for &page in &pages {
            self.stats.distinct.insert(page);
            if self.copy_of(page).is_some() {
                continue;
            }
            if made < FAULTS && self.faulted < all {
                if self.map(page) {
                    made += 1;
                    self.faulted += 1;
                    bring.push(page);
                }
            } else {
                self.stats.left += 1;
            }
        }
        self.scratch_pages = self.bring_pages(bring);
        self.stats.faults += pages.len() as u64;
        // The first frames that leave pages without a copy: all of them, and
        // which of them hold a packed structure's first bytes or lie in a
        // title buffer (to tell walked structures from stray pointers).
        static WHOLE: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        if made < pages.len() && self.faulted >= faults_all() && WHOLE.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 3 {
            let mut text = String::new();
            for &page in pages.iter().filter(|&&p| self.copy_of(p).is_none()) {
                let at = page * PAGE;
                let committed = crate::texture::committed(at as *const u8, PAGE as usize);
                let buffer = crate::shadow::in_buffer(at, 1).is_some();
                let packed = committed && self.packed_at(at).is_some();
                text += &format!(" {at:#x}{}{}{}", if committed { "" } else { "?" }, if buffer { "b" } else { "" }, if packed { "p" } else { "" });
            }
            tracing::warn!(frame = self.frame, "title memory by address: pages left without a copy by the bounds (?: not committed, b: in a title buffer, p: a packed structure there):{text}");
        }
        // Each is an access that read 0 or was dropped.
        static SAID: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        let said = SAID.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if said < 40 || said.is_multiple_of(100) {
            let some: Vec<String> = pages
                .iter()
                .step_by(pages.len().div_ceil(12))
                .map(|&p| format!("{:#x}", p * PAGE))
                .collect();
            tracing::warn!(
                frame = self.frame,
                pages = pages.len(),
                made,
                held = self.pages.len(),
                ?some,
                "title memory by address: shaders touched pages that had no copy (those accesses read 0 or were dropped); those that are the title's are there now"
            );
        }
    }

    /// The structures named in a list's upload memory around `address`
    /// (instances of a top-level build: a structure's address is their last
    /// 8 bytes): the first page of each, queued into `pending` for coalesced
    /// [`Memory::bring_pages`].
    fn named_in_upload(&mut self, address: u64, pending: &mut Vec<u64>) {
        let Some((from, len)) = crate::upload::block(address as usize) else {
            return;
        };
        // SAFETY: the block is handed out and stays mapped; the title wrote
        // it before it recorded the dispatch.
        let words = unsafe { std::slice::from_raw_parts(from as *const u64, len / 8) };
        let frame = self.frame;
        let mut last_page = u64::MAX;
        for &word in words {
            if word & 0xFF == 0 && (1 << 32..1 << 37).contains(&word) {
                let page = word / PAGE;
                if page == last_page {
                    continue;
                }
                last_page = page;
                if self.copy_of(page).is_some() || self.map(page) {
                    if self.pages.get(&page).is_some_and(|h| h.looked != frame) {
                        pending.push(page);
                    }
                }
            }
        }
    }

    /// A small title buffer whose results come back, if its pages are
    /// windows of title memory itself (or a copy of the buffer's own).
    fn small(&mut self, list: &HostList, start: u64, len: u64) {
        let mut at = start;
        while at < start + len {
            let page = at / PAGE;
            let n = ((page + 1) * PAGE).min(start + len) - at;
            self.map(page);
            if let Some(k) = self.copy_of(page)
                && !self.pages.contains_key(&page)
            {
                self.copies[k].refresh(list, at, n);
                self.copies[k].gpu_writes(list, at, n);
                self.stats.back += 1;
            }
            at += n;
        }
    }

    /// Keeps the directory and all owned arenas alive on `list` through GPU
    /// execution, and refreshes only non-arena title-buffer copies/windows
    /// (owned arenas and the directory mark their own dirty ranges directly).
    fn bring_up(&self, list: &HostList) {
        list.keep(&self.directory.1.buffer);
        for arena in &self.arenas {
            list.keep(&self.copies[arena.copy].buffer);
        }
        for &n in &self.title_copies {
            let shadow = &self.copies[n];
            let (start, len) = shadow.span();
            shadow.refresh(list, start, len);
        }
    }

    /// Every ten seconds or so, what was used.
    fn report(&mut self) {
        let now = std::time::Instant::now();
        let (since, frame) = *self.stats.from.get_or_insert((now, self.frame));
        if now.duration_since(since).as_secs_f64() < 10.0 {
            return;
        }
        let frames = self.frame.wrapping_sub(frame).max(1) as f64;
        let s = std::mem::take(&mut self.stats);
        tracing::info!(
            "title memory by address: {} pages ({} MB) in {} arenas, {} copies in all; per frame {:.0} dispatches that read, {:.0} that build, \
             {:.1} pages that had no copy, {:.1} pages turned down, {:.1} results brought back; {:.0} pages looked at, {:.1} KB changed, \
             {:.1} packed structures ({} there), {:.2} ms; {} windows; since the last line {} KB of packed structures came and {} of them went, \
             and pages for constants {}, views written {}, others {}, upload memory {}, faults in all {}; \
             since the last line {} different pages faulted, {:.1} a frame left without a copy by the bounds ({} in all)",
            self.pages.len(),
            (self.pages.len() as u64 * PAGE) >> 20,
            self.arenas.len(),
            self.copies.len(),
            s.uses[0] as f64 / frames,
            s.uses[1] as f64 / frames,
            s.faults as f64 / frames,
            s.unknown as f64 / frames,
            s.back as f64 / frames,
            s.looked as f64 / frames,
            s.changed as f64 / 1024.0 / frames,
            s.packed as f64 / frames,
            self.packed.len(),
            s.micros as f64 / 1000.0 / frames,
            self.windows,
            s.packed_bytes >> 10,
            s.gone,
            s.came[0],
            s.came[1],
            s.came[2],
            s.came[3],
            self.faulted,
            s.distinct.len(),
            s.left as f64 / frames,
            faults_all()
        );
        self.stats.from = Some((now, self.frame));
    }
}

/// A list with this key ([`HostList::used`]) asks the builder for sizes.
const SIZES: usize = usize::MAX - (9 << 33);

/// Whether a submission that asks for the sizes of structures is waited for,
/// and landed, before the title goes on (`VICTORY_BVH_SIZES_WAIT=1`; off).
///
/// Off: it does not help. In run light-15, with it, 33 of 154 copies had the
/// slot's old size still (all of them recorded one frame after the size was
/// asked for) against 53 of 230 in light-11, 48 of 275 dynamic structures
/// were good, and the title asks in every frame, so every frame waits (109 ms
/// against 90). The title reads the slot before our submission of the asking
/// list is through, or reads it for another purpose; not known.
///
/// The title asks for a new structure's compacted size (a program of one
/// thread stores it into a small buffer), reads it a frame or two later
/// without looking whether it is there, takes memory of that size and copies
/// the structure into it. With submissions that land up to two frames late it
/// read what the slot held before, some other structure's size, and the copy
/// wrote over its neighbours: of 230 such copies of run light-11, 99 were
/// asked for two frames earlier and landed in that frame, none of those with
/// the right size.
fn sizes_wait() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_BVH_SIZES_WAIT").is_some_and(|v| v == "1"))
}

/// Whether a dispatch of the builder is its program of a structure's size:
/// one thread, the structure's address and what to store (1: the compacted
/// size) as constants, a small title buffer to store into.
fn asks_size(written: &[u64], constants: &[u32], threads: Option<[u32; 3]>) -> bool {
    threads == Some([1, 1, 1])
        && matches!(*constants, [_, hi, 1..=3, ..] if (1..0x20).contains(&hi))
        && written.first().is_some_and(|&at| crate::shadow::in_buffer(at, 1).is_some_and(|b| b.1 <= BACK))
}

/// Whether one of `lists` asks for sizes.
pub(crate) fn asked_sizes<'a>(lists: impl Iterator<Item = &'a HostList>) -> bool {
    let mut lists = lists;
    let asked = sizes_wait() && lists.any(|l| l.used(SIZES));
    if asked {
        static TIMES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let n = TIMES.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
        if n.is_power_of_two() || n.is_multiple_of(100) {
            tracing::info!(n, frame = crate::frame::current(), "submissions that ask for structures' sizes, waited for");
        }
    }
    asked
}

/// Whether a page placed at submission remembers its write-watch stamp.
fn place_taken() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_PLACE_TAKEN").is_none_or(|v| v != "0"))
}

/// [`fill`] of a page stores by address do not reach.
fn close(entry: *mut u32, n: usize, offset: u64) {
    fill(entry, n | MEMORY_READ_ONLY as usize, offset);
}

/// The offset first: a shader that sees the number sees it too.
fn fill(entry: *mut u32, n: usize, offset: u64) {
    // SAFETY: an entry of the directory, under the mutex.
    unsafe {
        entry.add(1).write_volatile(offset as u32);
        entry.write_volatile(n as u32 + 1);
    }
}

/// The bytes of a packed structure whose first two dwords these are: the
/// version (6) with the marks of a bottom level and of one that is packed,
/// and its size unpacked in 64-byte nodes.
fn packed_len(header: u64) -> Option<u64> {
    let (version, nodes) = (header as u32, header >> 32);
    let marks = version >> 16;
    // The header counts its nodes of 64 bytes without the last one: the
    // builder's own program of a structure's size stores one more.
    let len = (nodes + 1) * 64;
    (version & 0xFFFF == 6 && marks & 0x8100 == 0x8100 && nodes > 0 && len <= PACKED).then_some(len)
}

/// Queries write-watch state for sorted, deduplicated 64 KiB `pages`,
/// coalescing nearby pages (within [`WATCH_GAP`] pages and [`WATCH_SPAN`] total
/// span) into batched `watch::writes` calls. Results are written to `out` in
/// the same sorted page order as `(page, Some((now, wrote)))` or `(page, None)`.
fn query_page_writes(pages: &[u64], out: &mut Vec<(u64, Option<(u64, u64)>)>) {
    out.clear();
    out.reserve(pages.len());
    let mut i = 0;
    while i < pages.len() {
        let start_page = pages[i];
        let mut j = i + 1;
        while j < pages.len()
            && pages[j] - pages[j - 1] <= WATCH_GAP
            && pages[j] - start_page < WATCH_SPAN
        {
            j += 1;
        }
        query_cluster(&pages[i..j], out);
        i = j;
    }
}

/// Queries one sorted cluster of pages with a single `watch::writes` call,
/// splitting by halves if the span crosses separate `VirtualAlloc` reservations
/// or unwatched holes.
fn query_cluster(chunk: &[u64], out: &mut Vec<(u64, Option<(u64, u64)>)>) {
    if chunk.is_empty() {
        return;
    }
    let first = chunk[0];
    let end = chunk[chunk.len() - 1] + 1;
    let contiguous = end - first == chunk.len() as u64;
    // Only pass `guarded = true` when there are no gaps between pages, so we
    // never guard unrelated memory in a gap.
    match crate::watch::writes((first * PAGE) as usize, (end * PAGE) as usize, contiguous) {
        Some((now, written)) => {
            let mut wi = 0;
            for &page in chunk {
                let p = page as usize;
                while wi < written.len() && written[wi].0 < p {
                    wi += 1;
                }
                let wrote = if wi < written.len() && written[wi].0 == p {
                    written[wi].1
                } else {
                    0
                };
                out.push((page, Some((now, wrote))));
            }
        }
        None if chunk.len() > 1 => {
            let mid = chunk.len() / 2;
            query_cluster(&chunk[..mid], out);
            query_cluster(&chunk[mid..], out);
        }
        None => {
            out.push((first, None));
        }
    }
}

/// Binary-search lookup in the sorted output of [`query_page_writes`].
fn lookup_watch(watch: &[(u64, Option<(u64, u64)>)], page: u64) -> Option<(u64, u64)> {
    watch.binary_search_by_key(&page, |&(p, _)| p).ok().and_then(|i| watch[i].1)
}

/// Puts what `title` holds into `ours` (what was last taken): the dwords that
/// differ. How many.
#[cfg(test)]
fn merge(title: &[u8; PIECE], ours: &mut [u8], never: bool) -> usize {
    if never {
        ours.copy_from_slice(title);
        return PIECE / 4;
    }
    if ours == title {
        return 0;
    }
    let mut changed = 0;
    for (t, o) in title.chunks_exact(4).zip(ours.chunks_exact_mut(4)) {
        if t != o {
            o.copy_from_slice(t);
            changed += 1;
        }
    }
    changed
}

/// Before a draw whose shader reads title memory by address: what its
/// root arguments name is made ready (`addresses`: its root views, of them
/// `written` those it writes; `constants`: its root constants, see the
/// module), every numbered copy
/// has its view in `heap`'s run and is brought up to date when `list` is
/// submitted. None if the host cannot hold it; the draw is then dropped.
pub(crate) fn ready(
    list: &HostList,
    heap: &HostHeap,
    addresses: &[u64],
    written: &[u64],
    constants: &[u32],
    use_: Use,
    threads: Option<[u32; 3]>,
) -> Option<Ready> {
    let mut guard = MEMORY.lock().unwrap_or_else(|e| e.into_inner());
    if guard.is_none() {
        *guard = Memory::new();
    }
    let m = guard.as_mut()?;
    let frame = crate::frame::current();
    if m.frame != frame {
        m.frame = frame;
        m.refused.clear();
        m.reclaim_retired();
        m.take_faults();
        m.look_again();
        m.report();
        #[cfg(feature = "diagnostics")]
        m.trace.look(frame);
    }
    #[cfg(feature = "diagnostics")]
    if use_ != Use::Reads {
        m.trace.dispatch(frame, list, addresses, written, constants, threads);
    }
    let builds = use_ != Use::Reads;
    if builds && sizes_wait() && asks_size(written, constants, threads) {
        list.first_use(SIZES);
    }
    m.reclaim_retired();
    while builds && m.free_pages < SPARE && m.pages.len() + m.free_pages + m.retiring.len() < MOST && m.another() {}
    m.stats.uses[usize::from(builds)] += 1;
    // Once per list and page named: the builder dispatches hundreds of times
    // a frame with the same arguments.
    const READ: usize = usize::MAX - 0x6d;
    let named = |kind: usize, at: u64| list.first_use(usize::MAX - (kind << 26) - (at / PAGE) as usize);
    // A title address in two constants (low dword, then bits 32 to 39), or
    // in one in 256-byte units (the builder's alone). A constant that only
    // looks like one names memory nothing reads; one that is not the title's
    // memory is refused.
    let held = |m: &Memory| m.pages.len() as u64;
    let mut pending = std::mem::take(&mut m.scratch_pages);
    pending.clear();
    let mut land_addrs: Vec<u64> = Vec::new();
    let before = held(m);
    for (i, &c) in constants.iter().enumerate() {
        if let Some(&high) = constants.get(i + 1)
            && (1..=0x1F).contains(&high)
            && c >= 0x1_0000
        {
            let at = u64::from(c) | u64::from(high) << 32;
            if named(1, at) {
                m.map_range(at, 1, &mut pending);
            }
        }
        if builds && (0x0100_0000..0x2000_0000).contains(&c) {
            let at = u64::from(c) << 8;
            if named(1, at) {
                m.map_range(at, 1, &mut pending);
            }
        }
    }
    let mut came = [held(m) - before, 0, 0, 0];
    for &at in addresses.iter().filter(|&&a| a >= 1 << 32) {
        // (Several structures to a page.)
        let mut packed = false;
        if builds
            && list.first_use(usize::MAX - (3 << 33) - (at >> 8) as usize)
            && crate::shadow::in_buffer(at, 1).is_none()
            && m.possible(at / PAGE)
        {
            // Bind the arenas during recording. The batched callback
            // checks the latest headers and stages bytes at submission.
            let len = m.packed.get(&at)
                .filter(|known| known.frame + AGAIN >= frame && known.checked != 0)
                .map(|known| known.len)
                .or_else(|| m.packed_at(at).map(|(len, _)| len));
            if let Some(len) = len {
                packed = true;
                let first = at / PAGE;
                let last = (at + len - 1) / PAGE;
                let end = m.map_contiguous(first, last, true);
                pending.extend((first..end).filter(|page| {
                    m.pages.get(page).is_some_and(|h| h.looked != frame)
                }));
            }
            land_addrs.push(at);
        }
        if builds && written.contains(&at) && list.first_use(usize::MAX - (4 << 33) - (at >> 2 & 0x7FFF_FFFF) as usize) {
            if let Some((start, len)) = crate::shadow::in_buffer(at, 1)
                && len <= BACK
                && use_ == (Use::Builds { back: true })
                && list.first_use(usize::MAX - (5 << 33) - (start >> 2 & 0x7FFF_FFFF) as usize)
            {
                m.small(list, start, len);
            }
            m.opens(at);
            if !packed && crate::shadow::in_buffer(at, 1).is_none() {
                #[cfg(feature = "diagnostics")]
                {
                    let built = m.dump.built.entry(at).or_insert((frame, frame, 0));
                    (built.1, built.2) = (frame, built.2 + 1);
                }
                let before = held(m);
                m.map_range(at, BUILT, &mut pending);
                came[1] += held(m).saturating_sub(before);
            }
        }
        if packed || !named(2, at) {
            continue;
        }
        let before = held(m);
        match crate::shadow::in_buffer(at, 1) {
            Some(_) => {
                m.map_range(at, 1, &mut pending);
                m.named_in_upload(at, &mut pending);
                came[3] += held(m).saturating_sub(before);
            }
            None => {
                m.map_range(at, if builds { VIEW } else { PAGE }, &mut pending);
                came[2] += held(m).saturating_sub(before);
            }
        }
    }
    m.scratch_pages = m.bring_pages(pending);
    // Ordering invariant: all `addresses` in this `ready()` call belong to the
    // same dispatch, so no GPU command runs between them. `land_batch` runs
    // `land()` in recorded `addresses` order for all non-buffer addresses and
    // is followed immediately by `bring_up(list)` so `directory` + all `copies`
    // are refreshed before the dispatch. `m.small()` inside the loop only
    // touches non-arena title-buffer copies (`!pages.contains_key`), disjoint
    // from the arena pages written by `land_batch`. Across separate dispatches
    // or copies on the same list, each `ready()` queues its own `defer_host` +
    // `bring_up` in exact recorded order.
    if !land_addrs.is_empty() {
        list.defer_host(move || {
            if let Some(m) = MEMORY.lock().unwrap_or_else(|e| e.into_inner()).as_mut() {
                m.land_batch(&land_addrs);
            }
        });
        m.bring_up(list);
        // `m.copies` is held behind `MEMORY` and cannot grow between here and
        // the `first_use` check below (all `map`/`another`/`opens`/`small`
        // calls finished in the loop above). Recording the same key avoids a
        // duplicate `bring_up(list)` for the same `m.copies.len()` on this
        // dispatch while still firing `bring_up` if a later `ready()` on this
        // list allocates another copy.
        let _ = list.first_use(usize::MAX - (6 << 33) - m.copies.len());
    }
    for (sum, n) in m.stats.came.iter_mut().zip(came) {
        *sum += n;
    }
    let key = std::ptr::from_ref(heap) as usize;
    let (base, made, is_new_heap) = match m.heaps.get(&key) {
        Some(&(base, made)) => (base, made, false),
        None => {
            // Held for the heap's life: not given back with a list's slots.
            let base = heap.dynamic(COPIES + 1)?;
            // Every slot of the run: a number shows no view of another kind.
            for n in 0..=COPIES {
                heap.null_raw_uav_at(base + n);
            }
            (base, 0, true)
        }
    };
    let copies = m.copies.len() as u32;
    if is_new_heap || made < copies {
        for (n, shadow) in m.copies.iter().enumerate().skip(made as usize) {
            let (_, len) = shadow.span();
            heap.raw_uav_at(base + 1 + n as u32, &shadow.buffer, 0, len);
        }
        m.heaps.insert(key, (base, copies));
    }
    // Every copy by every list that has come to by-address memory, whoever
    // numbered it: an arena another list made while this one was being
    // recorded was brought up to date by that list's submission alone, which
    // may run after this one's (structures unpacked from zeros, and then
    // written over by the arena's first upload: in frames that stream much in
    // half of them, run light-8).
    let first_read = list.first_use(READ);
    if list.first_use(usize::MAX - (6 << 33) - m.copies.len()) {
        m.bring_up(list);
    }
    if first_read {
        list.keep(&m.faults.0);
        #[cfg(feature = "diagnostics")]
        m.dump(list, addresses);
    }
    #[cfg(feature = "diagnostics")]
    if use_ == Use::Reads {
        m.dump.walked(addresses);
    }
    let directory = &m.directory.1;
    Some(Ready {
        directory: directory.gpu(directory.span().0),
        faults: m.faults.2,
        base,
    })
}

/// `VICTORY_BVH_TRACE=1`: the sizes the builder is asked for (its program of
/// one thread that stores a structure's compacted size) and when title memory
/// has them, and the copies to a place of that size.
#[cfg(feature = "diagnostics")]
mod trace {
    use super::*;

    #[derive(Default)]
    pub(super) struct Trace {
        /// Frame, where the size goes, the structure, what title memory had.
        asked: Vec<(u64, u64, u64, u64)>,
        /// By structure, what its size was when last seen to change.
        sizes: HashMap<u64, (u64, u64)>,
        said: u32,
    }

    fn on() -> bool {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ON.get_or_init(|| std::env::var_os("VICTORY_BVH_TRACE").is_some_and(|v| v == "1"))
    }

    fn value(at: u64) -> u64 {
        let mut b = [0u8; 8];
        crate::host::copy_mapped(b.as_mut_ptr(), at as *const u8, 8, false);
        u64::from_le_bytes(b)
    }

    impl Trace {
        fn say(&mut self, text: String) {
            self.said += 1;
            if self.said < 6000 {
                tracing::info!("bvh trace: {text}");
            }
        }

        pub(super) fn look(&mut self, frame: u64) {
            if !on() {
                return;
            }
            let mut gone = Vec::new();
            for i in 0..self.asked.len() {
                let (then, at, structure, was) = self.asked[i];
                let now = value(at);
                if now != was {
                    self.sizes.insert(structure, (now, frame));
                    gone.push(i);
                    self.say(format!("frame {frame}: size of {structure:#x} at {at:#x} asked in {then}: {was:#x} -> {now:#x}"));
                } else if frame > then + 60 {
                    gone.push(i);
                    self.say(format!("frame {frame}: size of {structure:#x} at {at:#x} asked in {then}: still {was:#x}"));
                }
            }
            for i in gone.into_iter().rev() {
                self.asked.swap_remove(i);
            }
        }

        pub(super) fn dispatch(
            &mut self,
            frame: u64,
            list: &HostList,
            addresses: &[u64],
            written: &[u64],
            constants: &[u32],
            threads: Option<[u32; 3]>,
        ) {
            if !on() {
                return;
            }
            let _ = addresses;
            let list = std::ptr::from_ref(list) as usize & 0xFFFF_FFFF;
            if threads == Some([1, 1, 1])
                && let [lo, hi, mode, ..] = *constants
                && (1..=3).contains(&mode)
                && hi < 0x20
                && let Some(&at) = written.first()
                && crate::shadow::in_buffer(at, 1).is_some_and(|b| b.1 <= BACK)
            {
                let structure = u64::from(lo) | u64::from(hi) << 32;
                let was = value(at);
                self.say(format!("frame {frame} list {list:x}: asks mode {mode} of {structure:#x} to {at:#x}, there {was:#x}"));
                if mode == 1 {
                    self.asked.push((frame, at, structure, was));
                }
            } else if threads == Some([32, 1, 1])
                && let [from, to] = *written
                && [from, to].iter().all(|&a| a >= 1 << 32 && crate::shadow::in_buffer(a, 1).is_none())
            {
                let known = self.sizes.get(&from).copied();
                self.say(format!("frame {frame} list {list:x}: copies {from:#x} to {to:#x} constants {:x?}, its size was {known:x?}", &constants[..constants.len().min(4)]));
            }
        }
    }
}

/// `VICTORY_MEMORY_DUMP=<folder>`: when a file `go` appears there, the arenas
/// as the card holds them are copied out and written (`arena-<n>.bin`), with
/// `pages.txt` (title page, arena, offset) and the structures the walking
/// programs were given: to look at the structures offline.
#[cfg(feature = "diagnostics")]
mod dump {
    use super::*;

    #[derive(Default)]
    pub(super) struct Dump {
        walked: std::collections::BTreeSet<u64>,
        /// Views the builder was given to write outside buffers: the first
        /// and the last frame, how often.
        pub(super) built: std::collections::BTreeMap<u64, (u64, u64, u64)>,
        /// The frame it is written at, and the copies.
        due: Option<(u64, Vec<(ID3D12Resource, usize)>)>,
        done: bool,
    }

    impl Dump {
        pub(super) fn walked(&mut self, addresses: &[u64]) {
            if !self.done && self.walked.len() < 64 {
                self.walked.extend(addresses);
            }
        }
    }

    fn folder() -> Option<&'static std::path::PathBuf> {
        static FOLDER: std::sync::OnceLock<Option<std::path::PathBuf>> = std::sync::OnceLock::new();
        FOLDER.get_or_init(|| std::env::var_os("VICTORY_MEMORY_DUMP").map(Into::into)).as_ref()
    }

    fn turn(buffer: &ID3D12Resource, before: D3D12_RESOURCE_STATES, after: D3D12_RESOURCE_STATES) -> D3D12_RESOURCE_BARRIER {
        D3D12_RESOURCE_BARRIER {
            Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
            Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
            Anonymous: D3D12_RESOURCE_BARRIER_0 {
                Transition: std::mem::ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                    // SAFETY: the same pointer, not counted; the barrier is
                    // used while the buffer lives.
                    pResource: unsafe { std::mem::transmute_copy(buffer) },
                    Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                    StateBefore: before,
                    StateAfter: after,
                }),
            },
        }
    }

    impl Memory {
        /// At a list's first use: the copies are recorded into it.
        pub(super) fn dump(&mut self, list: &HostList, _: &[u64]) {
            let Some(folder) = folder() else { return };
            if let Some((due, _)) = &self.dump.due {
                if self.frame < *due {
                    return;
                }
                let copies = self.dump.due.take().map(|d| d.1).unwrap_or_default();
                let mut text = String::new();
                for a in &self.dump.walked {
                    text += &format!("walked {a:#x}\n");
                }
                for (a, (first, last, n)) in &self.dump.built {
                    text += &format!("built {a:#x} {first} {last} {n}\n");
                }
                for (a, p) in &self.packed {
                    text += &format!("packed {a:#x} {} {}\n", p.len, p.frame);
                }
                text += &format!("frame {}\n", self.frame);
                for (page, held) in &self.pages {
                    let arena = self.arenas.iter().position(|a| a.copy == held.copy).unwrap_or(usize::MAX);
                    let at = held.offset;
                    text += &format!("{:#x} {arena} {at:#x}\n", page * PAGE);
                }
                let _ = std::fs::write(folder.join("pages.txt"), text);
                for (n, (_, data)) in copies.iter().enumerate() {
                    // SAFETY: a mapping of an arena's bytes; the copies ran
                    // frames ago.
                    let bytes = unsafe { std::slice::from_raw_parts(*data as *const u8, ARENA as usize) };
                    let _ = std::fs::write(folder.join(format!("arena-{n}.bin")), bytes);
                }
                tracing::info!(arenas = copies.len(), "title memory by address: the arenas are written out");
                return;
            }
            if self.dump.done || !folder.join("go").exists() {
                return;
            }
            self.dump.done = true;
            let Some(host) = host() else { return };
            let mut copies = Vec::new();
            for arena in &self.arenas {
                let buffer = &self.copies[arena.copy].buffer;
                let Some((to, data)) = host.shared_buffer(ARENA) else { break };
                // SAFETY: an open list; live buffers of those sizes. The list
                // has not used the arena yet.
                unsafe {
                    list.list.ResourceBarrier(&[turn(buffer, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_SOURCE)]);
                    list.list.CopyBufferRegion(&to, 0, buffer, 0, ARENA);
                    list.list.ResourceBarrier(&[turn(buffer, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COMMON)]);
                }
                list.keep(&to);
                copies.push((to, data as usize));
            }
            self.dump.due = Some((self.frame + 12, copies));
        }
    }
}

#[cfg(test)]
#[path = "memory_tests.rs"]
mod gpu_tests;

#[cfg(test)]
#[path = "memory_locality_tests.rs"]
mod locality_tests;

#[cfg(test)]
#[path = "memory_prep_tests.rs"]
mod prep_tests;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_piece_takes_what_changed() {
        let mut title = [7u8; PIECE];
        let mut ours = vec![0u8; PIECE];
        assert_eq!(merge(&title, &mut ours, true), PIECE / 4);
        assert_eq!(ours, title);
        assert_eq!(merge(&title, &mut ours, false), 0);
        // A dword of it.
        title[41] = 8;
        assert_eq!(merge(&title, &mut ours, false), 1);
        assert_eq!(ours, title);
    }

    #[test]
    fn a_packed_structure_is_known_by_its_header() {
        let header = |version: u32, nodes: u32| u64::from(version) | u64::from(nodes) << 32;
        assert_eq!(packed_len(header(0x8102_0006, 0x9a1a)), Some(0x9a1b * 64));
        // Unpacked, a top level, another version, no size, too big.
        assert_eq!(packed_len(header(0x8002_0006, 0x9a1a)), None);
        assert_eq!(packed_len(header(0x0000_0006, 5)), None);
        assert_eq!(packed_len(header(0x8102_0007, 5)), None);
        assert_eq!(packed_len(header(0x8102_0006, 0)), None);
        assert_eq!(packed_len(header(0x8102_0006, 1 << 21)), None);
    }
}
