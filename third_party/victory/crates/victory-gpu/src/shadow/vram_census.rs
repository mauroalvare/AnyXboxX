//! VRAM census: what each of the host's reserves and pools takes of the
//! card's own memory, as the system counts it (DXGI `QueryVideoMemoryInfo`,
//! local segment group), made through the paths a run uses; then a churn
//! that makes and lets go of them in a loop and looks for memory, or room,
//! that is not given back. Device tests, ignored: run each in its own
//! process with the launcher's 8 GB preset in the environment (and
//! `VICTORY_BURY_LAG=0`: what is let go is dropped at once in a process that
//! submits nothing), e.g.
//! `cargo test -p victory-gpu --lib --release -- --ignored --exact shadow::vram_census::census --nocapture`.

use super::*;
use windows::Win32::Graphics::Direct3D12::*;

const MB: u64 = 1 << 20;

/// The card's budget and what this process has in use of it.
fn local() -> (u64, u64) {
    let [budget, used, ..] = host().and_then(|h| h.video_memory()).unwrap_or_default();
    (budget, used)
}

/// What `make` adds to the card's memory in use, said under `name`; the
/// things it made are handed back so they stay alive.
fn step<T>(name: &str, make: impl FnOnce() -> T) -> T {
    let (_, before) = local();
    let made = make();
    // A beat for the system's count.
    std::thread::sleep(std::time::Duration::from_millis(300));
    let (budget, after) = local();
    eprintln!(
        "census | {name:<64} | {:>+6} MB | in use {:>5} of {:>5} MB ({:>3} %)",
        (after as i64 - before as i64) / MB as i64,
        after / MB,
        budget / MB,
        after * 100 / budget.max(1)
    );
    made
}

/// Whether a buffer lies in the card's own memory (a default heap).
fn on_card(buffer: &ID3D12Resource) -> bool {
    let mut heap = D3D12_HEAP_PROPERTIES::default();
    // SAFETY: a live committed resource.
    unsafe { buffer.GetHeapProperties(Some(&mut heap), None) }.is_ok() && heap.Type == D3D12_HEAP_TYPE_DEFAULT
}

/// Title buffers of `size` bytes at fresh title memory, registered and their
/// copies made (`find`), up to `n`: the copies with a card side.
fn buffer_copies(size: u64, n: usize, id: &mut usize) -> Vec<(usize, u64, Arc<Shadow>)> {
    let mut out = Vec::new();
    for _ in 0..n {
        let Some(at) = crate::upload::reserve(size as usize) else { break };
        *id += 1;
        register(*id, at as u64, size);
        match find(at as u64, size) {
            Some(shadow) => out.push((*id, at as u64, shadow)),
            None => unregister(*id, at as u64),
        }
    }
    out
}

/// The title releases them and nothing else holds their copies.
fn release_buffers(copies: Vec<(usize, u64, Arc<Shadow>)>) {
    for (id, at, shadow) in copies {
        unregister(id, at);
        drop(shadow);
    }
}

/// Windows of title memory outside buffers, each [`WINDOW`] bytes of fresh
/// memory, up to `n`.
fn windows(n: usize) -> Vec<((u64, u64), Arc<Shadow>)> {
    let mut out = Vec::new();
    for _ in 0..n {
        let Some(at) = crate::upload::reserve(WINDOW as usize) else { break };
        let (start, len) = window_of(at as u64, at as u64 + 64, WINDOW);
        if let Some(shadow) = find(at as u64, 64) {
            out.push(((start, len), shadow));
        }
    }
    out
}

/// What the sweep does to idle windows ([`let_go_idle`]), at once.
fn release_windows(made: Vec<((u64, u64), Arc<Shadow>)>) {
    let mut map = LOOSE.write().unwrap_or_else(|e| e.into_inner());
    for (key, shadow) in made {
        map.remove(&key);
        drop(shadow);
    }
}

/// Copies by address (ray-tracing arenas) of `size` bytes, `n` of them; how
/// many lie on the card.
fn arenas(size: u64, n: usize) -> (Vec<Arc<Shadow>>, usize) {
    let made: Vec<Arc<Shadow>> = (0..n).filter_map(|_| {
        let at = crate::upload::reserve(size as usize)?;
        by_address(at as u64, size)
    }).collect();
    let carded = made.iter().filter(|s| on_card(&s.buffer)).count();
    (made, carded)
}

/// A small BC7 texture's description, for the texture pool.
fn small_texture() -> D3D12_RESOURCE_DESC {
    D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
        Width: 1024,
        Height: 1024,
        DepthOrArraySize: 1,
        MipLevels: 1,
        Format: windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_BC7_UNORM,
        SampleDesc: windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        ..Default::default()
    }
}

/// `n` textures (1 MiB each) from the texture pool.
fn pool_textures(n: usize) -> Vec<ID3D12Resource> {
    let host = host().unwrap();
    let desc = small_texture();
    // SAFETY: a valid description.
    (0..n).filter_map(|_| unsafe { crate::pool::texture(&host.device, &desc, D3D12_RESOURCE_STATE_COMMON, true) }).collect()
}

/// Host lists as the title's lists use them: indirect arguments clamped
/// (`indirect_scratch`, `bytes` of them) and `batches` batch places.
fn lists(n: usize, bytes: u64, batches: usize) -> Vec<crate::host::HostList> {
    let host = host().unwrap();
    (0..n)
        .filter_map(|_| {
            let list = host.new_list().ok()?;
            list.indirect_scratch(bytes)?;
            for _ in 0..batches {
                list.batch_place()?;
            }
            Some(list)
        })
        .collect()
}

/// Copies by address (ray-tracing arenas) give their room on the card back
/// when they go: made and let go one after another, 72 of 64 MiB (past the
/// 4,096 MB `VICTORY_MEMORY_CARD_MB` allows by default), each lies where the
/// first one did. Their count only ever grew: from the 65th on every arena
/// went to the PC's memory with the card all but empty.
#[test]
fn arenas_give_their_card_room_back() {
    let Some(host) = host() else { return };
    if !cards() || !host.card_ready() {
        return;
    }
    let mut placed = Vec::new();
    for _ in 0..72 {
        let (made, carded) = arenas(64 * MB, 1);
        assert_eq!(made.len(), 1, "no copy by address");
        placed.push(carded);
    }
    assert!(placed.iter().all(|&n| n == placed[0]), "where each arena lay: {placed:?}");
}

#[test]
#[ignore = "VRAM census on the real card: a minute, and it fills part of the card's budget"]
fn census() {
    let Some(host) = host() else { return };
    let (budget, start) = local();
    eprintln!(
        "census | budget {} MB, in use with the device made {} MB; card buffers {:?} (least, most), windows {} MB, room for buffers kept {}",
        budget / MB,
        start / MB,
        card_limits(),
        window_card_most() / MB,
        reserve_card_budgets()
    );
    step("card work: compare pass and its lists, results (PC memory)", || host.card_ready());
    let mut id = 0x5000_0000;
    let copies = step("card copies of title buffers, 4 MiB each, until refused", || buffer_copies(4 * MB, 512, &mut id));
    let carded = copies.iter().filter(|c| c.2.card.is_some()).count();
    eprintln!("census |   {} copies, {carded} with a card side ({} MB counted)", copies.len(), CARD_BYTES[0].load(Ordering::Relaxed) / MB);
    let win = step("window card sides (4 MiB windows) until refused", || windows(512));
    let carded = win.iter().filter(|w| w.1.card.is_some()).count();
    eprintln!("census |   {} windows, {carded} with a card side ({} MB counted)", win.len(), WINDOW_CARD_BYTES.load(Ordering::Relaxed) / MB);
    let (by_addr, carded) = step("ray-tracing arenas by address, 64 MiB each, 16 of them", || arenas(64 * MB, 16));
    eprintln!("census |   {} arenas, {carded} on the card; window bytes counted now {} MB", by_addr.len(), WINDOW_CARD_BYTES.load(Ordering::Relaxed) / MB);
    // The host's once-a-second look has run in a game by now.
    crate::pool::room(host.video_memory());
    let textures = step("texture pool: 64 textures of 1 MiB (and blocks made ahead)", || pool_textures(64));
    eprintln!("census |   {} textures; free in texture blocks {} MB", textures.len(), crate::pool::texture_room() / MB);
    let view_heap = step("one title view heap (1,000,000 slots asked)", || crate::descriptor::host::HostHeap::new(false, 1_000_000));
    let sampler_heap = step("one title sampler heap (2,048 slots)", || crate::descriptor::host::HostHeap::new(true, 2048));
    let host_lists = step("256 host lists, each 256 KiB of clamped arguments, 2 batches", || lists(256, 256 << 10, 2));
    let (_, end) = local();
    eprintln!("census | all of it: {} MB over the device's own, {} % of the budget", (end - start) / MB, end * 100 / budget.max(1));
    step("the lists go", || drop(host_lists));
    step("the heaps go", || drop((view_heap, sampler_heap)));
    step("the textures go (back in the pool after its rest)", || drop(textures));
    std::thread::sleep(std::time::Duration::from_millis(2600));
    step("the pool's once-a-second look", || crate::pool::room(host.video_memory()));
    step("the arenas go", || drop(by_addr));
    step("the windows go", || release_windows(win));
    step("the card copies go", || release_buffers(copies));
    let (_, after) = local();
    eprintln!(
        "census | at the end {} MB in use against {} MB at the start; free in texture blocks {} MB",
        after / MB,
        start / MB,
        crate::pool::texture_room() / MB
    );
}

/// Makes and lets go of card copies, windows, arenas, pool textures and host
/// lists in a loop: what is in use of the card must come back to where it
/// was, and what the host counts as taken must too, so that the same things
/// made again lie on the card again.
#[test]
#[ignore = "VRAM churn on the real card: about a minute"]
fn churn() {
    let Some(host) = host() else { return };
    if !host.card_ready() {
        return;
    }
    let cycles = std::env::var("VRAM_CHURN_CYCLES").ok().and_then(|v| v.parse().ok()).unwrap_or(24);
    let (budget, start) = local();
    eprintln!("churn | budget {} MB, in use at the start {} MB, {cycles} cycles", budget / MB, start / MB);
    let mut id = 0x6000_0000;
    let mut seen = Vec::new();
    let mut arenas_carded = Vec::new();
    for cycle in 0..cycles {
        let copies = buffer_copies(4 * MB, 32, &mut id);
        let win = windows(32);
        let (by_addr, carded) = arenas(64 * MB, 4);
        let textures = pool_textures(64);
        let host_lists = lists(16, (64 << 10) << (cycle % 6), 1 + cycle % 4);
        let (_, peak) = local();
        drop(host_lists);
        drop(textures);
        drop(by_addr);
        release_windows(win);
        release_buffers(copies);
        std::thread::sleep(std::time::Duration::from_millis(2600));
        crate::pool::room(host.video_memory());
        let (_, used) = local();
        eprintln!(
            "churn | cycle {cycle:3}: peak {:5} MB, after {:5} MB, arenas on the card {carded} of 4, counted: buffers {} MB, windows {} MB, free texture blocks {} MB",
            peak / MB,
            used / MB,
            CARD_BYTES[0].load(Ordering::Relaxed) / MB,
            WINDOW_CARD_BYTES.load(Ordering::Relaxed) / MB,
            crate::pool::texture_room() / MB
        );
        seen.push(used.saturating_sub(crate::pool::texture_room()));
        arenas_carded.push(carded);
    }
    let half = seen.len() / 2;
    let mean = |s: &[u64]| s.iter().sum::<u64>() / s.len().max(1) as u64;
    let (first, second) = (mean(&seen[..half]), mean(&seen[half..]));
    eprintln!(
        "churn | in use less empty texture blocks: first half {} MB, second half {} MB, start {} MB; arenas on the card per cycle {:?}",
        first / MB,
        second / MB,
        start / MB,
        arenas_carded
    );
    assert_eq!(CARD_BYTES[0].load(Ordering::Relaxed), 0, "card copies' bytes still counted");
    assert_eq!(WINDOW_CARD_BYTES.load(Ordering::Relaxed), 0, "windows' card bytes still counted");
    assert!(second <= first + 64 * MB, "creep: {} -> {} MB", first / MB, second / MB);
    assert!(
        arenas_carded.iter().all(|&n| n == arenas_carded[0]),
        "arenas made after others went lie on the card less often: {arenas_carded:?}"
    );
}
