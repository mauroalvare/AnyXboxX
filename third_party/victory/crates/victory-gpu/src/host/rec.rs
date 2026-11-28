//! A host list's D3D12 list, with barrier calls joined
//! (`VICTORY_JOIN_BARRIERS=1`; off by default; A/B-able with
//! `VICTORY_AB=<s>,VICTORY_JOIN_BARRIERS`).
//!
//! The state changes the host asks for (`crate::list::ensure_state`, the
//! aliasing before a texture's first upload) and the title's own barriers
//! wait here instead of going to the list at once; they go, in the order
//! they were asked for, in one call, at the next use of the list for
//! anything else. Every use goes through [`Rec`]'s `Deref`, so nothing is
//! recorded before a barrier asked for earlier. In a replay of capture
//! replay5 joining the calls that had no work between them saved 2.4 ms of
//! 79.5 (2,490 calls joined, 1,619 of them a texture's state change after
//! its upload followed by the next texture's aliasing).

use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering::Relaxed};

use windows::Win32::Graphics::Direct3D12::*;

/// Whether barriers are held now (`VICTORY_JOIN_BARRIERS=1`, or its phase
/// of `VICTORY_AB`).
pub(crate) fn on() -> bool {
    #[cfg(feature = "diagnostics")]
    if let Some(on) = crate::ab::pick("VICTORY_JOIN_BARRIERS") {
        return on;
    }
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_JOIN_BARRIERS").is_some_and(|v| v == "1"))
}

/// Barriers held. Their resources are borrowed: the list's states hold the
/// textures (`HostList::transition`), and the title's barriers name what
/// the title keeps alive until the list has run.
struct Held(Vec<D3D12_RESOURCE_BARRIER>);

// SAFETY: borrowed interface pointers of free-threaded objects.
unsafe impl Send for Held {}

/// Since the last frame-rate line: barrier calls asked for while held, and
/// calls made of them.
pub(crate) static STATS: [std::sync::atomic::AtomicU64; 2] = [const { std::sync::atomic::AtomicU64::new(0) }; 2];

pub(crate) struct Rec {
    list: ID3D12GraphicsCommandList,
    any: AtomicBool,
    /// True when a global NULL UAV barrier has already been emitted or held
    /// since the last non-barrier command on this list, so another UAV barrier
    /// before any intervening GPU work is redundant.
    uav_synced: AtomicBool,
    held: Mutex<Held>,
}

/// True if `b` is a UAV barrier with `pResource == None` (synchronizes all UAVs).
fn is_global_uav(b: &D3D12_RESOURCE_BARRIER) -> bool {
    if b.Type != D3D12_RESOURCE_BARRIER_TYPE_UAV {
        return false;
    }
    // SAFETY: `Type == UAV` selects the `UAV` union arm.
    unsafe { b.Anonymous.UAV.pResource.is_none() }
}

/// Resource pointer named by a per-resource UAV barrier (`None` for global UAV
/// or non-UAV barriers).
fn uav_resource(b: &D3D12_RESOURCE_BARRIER) -> Option<usize> {
    if b.Type != D3D12_RESOURCE_BARRIER_TYPE_UAV {
        return None;
    }
    // SAFETY: `Type == UAV` selects the `UAV` union arm.
    unsafe {
        b.Anonymous
            .UAV
            .pResource
            .as_ref()
            .map(|r| windows::core::Interface::as_raw(r) as usize)
    }
}

/// Filters/merges `incoming` into `out`, collapsing redundant UAV barriers
/// without altering any transition or aliasing barrier order. Returns whether
/// `out` (or a previously emitted barrier via `already_uav_synced`) now covers
/// all UAVs globally.
fn push_deduped(
    out: &mut Vec<D3D12_RESOURCE_BARRIER>,
    incoming: &[D3D12_RESOURCE_BARRIER],
    already_uav_synced: bool,
) -> bool {
    let mut has_global = already_uav_synced || out.iter().any(is_global_uav);
    if !has_global && incoming.iter().any(is_global_uav) {
        // A global NULL UAV barrier in this contiguous batch subsumes any
        // per-resource UAV barriers already queued in `out`.
        out.retain(|b| b.Type != D3D12_RESOURCE_BARRIER_TYPE_UAV);
    }
    for b in incoming {
        if b.Type == D3D12_RESOURCE_BARRIER_TYPE_UAV {
            if has_global {
                continue;
            }
            if is_global_uav(b) {
                has_global = true;
            } else if let Some(res) = uav_resource(b)
                && out.iter().any(|existing| uav_resource(existing) == Some(res))
            {
                continue;
            }
        }
        // SAFETY: plain copy of D3D12_RESOURCE_BARRIER with borrowed ManuallyDrop pointers.
        out.push(unsafe { std::ptr::read(b) });
    }
    has_global
}

impl Rec {
    pub(crate) fn new(list: ID3D12GraphicsCommandList) -> Self {
        Self {
            list,
            any: AtomicBool::new(false),
            uav_synced: AtomicBool::new(false),
            held: Mutex::new(Held(Vec::new())),
        }
    }

    /// Closes an unused recording for reset without emitting held barriers.
    ///
    /// # Safety
    /// The caller owns the recording. The list is open and will not be submitted.
    pub(super) unsafe fn close_discarded(&self) -> windows::core::Result<()> {
        self.held.lock().unwrap_or_else(|e| e.into_inner()).0.clear();
        self.any.store(false, Relaxed);
        self.uav_synced.store(false, Relaxed);
        // SAFETY: the caller's open, unsubmitted list.
        unsafe { self.list.Close() }
    }

    /// Whether a global NULL UAV barrier has already been emitted or held with
    /// no intervening commands on this list.
    pub(crate) fn is_uav_synced(&self) -> bool {
        self.uav_synced.load(Relaxed)
    }

    /// Records `barriers` now, or holds them for the next use of the list,
    /// collapsing redundant UAV barriers when no GPU work intervened.
    ///
    /// # Safety
    /// As `ResourceBarrier`'s: an open list, live resources.
    pub(crate) unsafe fn barriers(&self, barriers: &[D3D12_RESOURCE_BARRIER]) {
        if barriers.is_empty() {
            return;
        }
        let already_synced = self.uav_synced.load(Relaxed);
        if !on() {
            // (What was held while it was on goes first.)
            if self.any.load(Relaxed) {
                self.flush();
            }
            let mut filtered = Vec::with_capacity(barriers.len());
            let now_synced = push_deduped(&mut filtered, barriers, already_synced);
            if !filtered.is_empty() {
                // SAFETY: the caller's.
                unsafe { self.list.ResourceBarrier(&filtered) };
            }
            if now_synced {
                self.uav_synced.store(true, Relaxed);
            }
            return;
        }
        let mut held = self.held.lock().unwrap_or_else(|e| e.into_inner());
        // A resource whose state changes again: what is held goes first, so
        // that no call holds two changes of one resource.
        if barriers.iter().filter_map(named).any(|r| held.0.iter().filter_map(named).any(|h| h == r)) {
            // SAFETY: as in `flush`.
            unsafe { self.list.ResourceBarrier(&held.0) };
            held.0.clear();
            STATS[1].fetch_add(1, Relaxed);
        }
        let now_synced = push_deduped(&mut held.0, barriers, already_synced);
        if now_synced {
            self.uav_synced.store(true, Relaxed);
        }
        STATS[0].fetch_add(1, Relaxed);
        if !held.0.is_empty() {
            self.any.store(true, Relaxed);
        }
    }

    /// What is held goes to the list, in one call.
    fn flush(&self) {
        let mut held = self.held.lock().unwrap_or_else(|e| e.into_inner());
        self.any.store(false, Relaxed);
        if held.0.is_empty() {
            return;
        }
        // SAFETY: the barriers were asked for on this open list, in order;
        // their resources are alive (`Held`).
        unsafe { self.list.ResourceBarrier(&held.0) };
        held.0.clear();
        STATS[1].fetch_add(1, Relaxed);
    }
}

/// The resource a state change or aliasing names (its interface pointer).
fn named(b: &D3D12_RESOURCE_BARRIER) -> Option<usize> {
    // SAFETY: the union's member is the one `Type` says; a pointer read.
    unsafe {
        let p: Option<&ID3D12Resource> = match b.Type {
            D3D12_RESOURCE_BARRIER_TYPE_TRANSITION => b.Anonymous.Transition.pResource.as_ref(),
            D3D12_RESOURCE_BARRIER_TYPE_ALIASING => b.Anonymous.Aliasing.pResourceAfter.as_ref(),
            _ => None,
        };
        p.map(|r| windows::core::Interface::as_raw(r) as usize)
    }
}

impl std::ops::Deref for Rec {
    type Target = ID3D12GraphicsCommandList;

    /// The list, with what is held recorded first. Any command recorded
    /// through `Deref` may perform GPU writes, so `uav_synced` resets.
    fn deref(&self) -> &ID3D12GraphicsCommandList {
        if self.any.load(Relaxed) {
            self.flush();
        }
        self.uav_synced.store(false, Relaxed);
        &self.list
    }
}

/// For the frame-rate line.
pub(crate) fn line(frames: f64) {
    let [asked, made] = [0, 1].map(|i| STATS[i].swap(0, Relaxed) as f64 / frames);
    if asked > 0.0 {
        tracing::info!(
            "barrier calls per frame (VICTORY_JOIN_BARRIERS): {asked:.0} asked for, {made:.0} made of them"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn global_uav() -> D3D12_RESOURCE_BARRIER {
        D3D12_RESOURCE_BARRIER {
            Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
            Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
            Anonymous: D3D12_RESOURCE_BARRIER_0 {
                UAV: std::mem::ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER {
                    pResource: std::mem::ManuallyDrop::new(None),
                }),
            },
        }
    }

    fn transition_dummy() -> D3D12_RESOURCE_BARRIER {
        D3D12_RESOURCE_BARRIER {
            Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
            Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
            Anonymous: D3D12_RESOURCE_BARRIER_0 {
                Transition: std::mem::ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                    pResource: std::mem::ManuallyDrop::new(None),
                    Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                    StateBefore: D3D12_RESOURCE_STATE_COMMON,
                    StateAfter: D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                }),
            },
        }
    }

    #[test]
    fn consecutive_global_uav_barriers_collapse_while_keeping_transitions() {
        let mut out = Vec::new();
        assert!(push_deduped(&mut out, &[global_uav(), global_uav()], false));
        assert_eq!(out.len(), 1);
        assert!(is_global_uav(&out[0]));

        // Already synced: another global UAV is dropped, transition is preserved.
        let mut next = Vec::new();
        assert!(push_deduped(&mut next, &[global_uav(), transition_dummy()], true));
        assert_eq!(next.len(), 1);
        assert_eq!(next[0].Type, D3D12_RESOURCE_BARRIER_TYPE_TRANSITION);
    }

    /// Representative multi-pass RT tile-class GPU benchmark comparing un-deduped
    /// global UAV barriers (old behavior) against scoped/deduplicated UAV
    /// barriers (`Rec` dedup + per-resource UAV scoping) at equal dispatch count,
    /// verifying exact bit-identical producer-consumer output.
    /// Run with:
    /// `cargo test -p victory-gpu --release --lib host::rec::tests::rt_barrier_scoping_and_dedup_bench -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn rt_barrier_scoping_and_dedup_bench() {
        use std::mem::ManuallyDrop;
        use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
        use windows::Win32::Graphics::Direct3D::{D3D_FEATURE_LEVEL_12_0, ID3DBlob};
        use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT_UNKNOWN, DXGI_SAMPLE_DESC};
        use windows::Win32::Graphics::Dxgi::{
            CreateDXGIFactory2, DXGI_ADAPTER_FLAG_SOFTWARE, DXGI_CREATE_FACTORY_FLAGS,
            IDXGIFactory4,
        };
        use windows::Win32::System::Threading::{CreateEventW, INFINITE, WaitForSingleObject};
        use windows::core::{Interface, s};

        const HLSL: &str = r#"
#define RS "RootConstants(num32BitConstants=4, b0), UAV(u0)"
cbuffer C : register(b0) { uint pass_id; uint count; uint occluder_x; uint pad; };
RWByteAddressBuffer buf : register(u0);
[RootSignature(RS)]
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    if (id.x >= count) return;
    uint prev = (pass_id == 0u) ? 0u : buf.Load(id.x * 4u);
    uint blocked = (id.x >= occluder_x && id.x < occluder_x + 1024u) ? 0u : 1u;
    buf.Store(id.x * 4u, prev + blocked + pass_id);
}
"#;
        const STREAMS: usize = 8;
        const PASSES_PER_STREAM: u32 = 32; // 256 RT/resolve compute dispatches per frame
        const ELEMENTS: u32 = 16_384; // 64 KiB per independent RT tile stream
        const WARMUP: usize = 10;
        const ITERS: usize = 40;
        const BUDGET_MS: f64 = 33.333;

        // Read-only enhanced barrier group (SRV -> INDIRECT_ARGUMENT) that the
        // title may issue between independent tile dispatches of the same pass.
        let ro_buf_barrier = D3D12_BUFFER_BARRIER {
            SyncBefore: D3D12_BARRIER_SYNC_COMPUTE_SHADING,
            SyncAfter: D3D12_BARRIER_SYNC_COMPUTE_SHADING,
            AccessBefore: D3D12_BARRIER_ACCESS_SHADER_RESOURCE,
            AccessAfter: D3D12_BARRIER_ACCESS_INDIRECT_ARGUMENT,
            pResource: ManuallyDrop::new(None),
            Offset: 0,
            Size: u64::MAX,
        };
        let ro_group = D3D12_BARRIER_GROUP {
            Type: D3D12_BARRIER_TYPE_BUFFER,
            NumBarriers: 1,
            Anonymous: D3D12_BARRIER_GROUP_0 {
                pBufferBarriers: &ro_buf_barrier,
            },
        };
        // Producer -> consumer enhanced barrier group between passes (UAV -> UAV).
        let uav_buf_barrier = D3D12_BUFFER_BARRIER {
            SyncBefore: D3D12_BARRIER_SYNC_COMPUTE_SHADING,
            SyncAfter: D3D12_BARRIER_SYNC_COMPUTE_SHADING,
            AccessBefore: D3D12_BARRIER_ACCESS_UNORDERED_ACCESS,
            AccessAfter: D3D12_BARRIER_ACCESS_UNORDERED_ACCESS,
            pResource: ManuallyDrop::new(None),
            Offset: 0,
            Size: u64::MAX,
        };
        let uav_group = D3D12_BARRIER_GROUP {
            Type: D3D12_BARRIER_TYPE_BUFFER,
            NumBarriers: 1,
            Anonymous: D3D12_BARRIER_GROUP_0 {
                pBufferBarriers: &uav_buf_barrier,
            },
        };

        let (device, adapter_name) = unsafe {
            let factory: IDXGIFactory4 = CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0)).unwrap();
            let mut found = None;
            for i in 0.. {
                let Ok(adapter) = factory.EnumAdapters1(i) else { break };
                let Ok(desc) = adapter.GetDesc1() else { continue };
                if desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 != 0 {
                    continue;
                }
                let len = desc.Description.iter().position(|&c| c == 0).unwrap_or(desc.Description.len());
                let name = String::from_utf16_lossy(&desc.Description[..len]);
                let mut dev: Option<ID3D12Device> = None;
                if D3D12CreateDevice(&adapter, D3D_FEATURE_LEVEL_12_0, &mut dev).is_ok()
                    && let Some(d) = dev
                {
                    found = Some((d, name));
                    break;
                }
            }
            found.expect("hardware D3D12 device")
        };

        let (queue, allocator, raw_list, fence, event, ts_freq, root, pso, buffers, readback, ts_heap, ts_rb) = unsafe {
            let queue: ID3D12CommandQueue = device
                .CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
                    Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
                    ..Default::default()
                })
                .unwrap();
            let allocator: ID3D12CommandAllocator = device
                .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)
                .unwrap();
            let list: ID3D12GraphicsCommandList = device
                .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &allocator, None)
                .unwrap();
            list.Close().unwrap();
            let fence: ID3D12Fence = device.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
            let event = CreateEventW(None, false, false, None).unwrap();
            let ts_freq = queue.GetTimestampFrequency().unwrap();

            let mut code: Option<ID3DBlob> = None;
            D3DCompile(
                HLSL.as_ptr().cast(),
                HLSL.len(),
                s!("barrier_bench"),
                None,
                None,
                s!("main"),
                s!("cs_5_1"),
                0,
                0,
                &mut code,
                None,
            )
            .unwrap();
            let blob = code.unwrap();
            let bytes = std::slice::from_raw_parts(blob.GetBufferPointer().cast::<u8>(), blob.GetBufferSize());
            let root: ID3D12RootSignature = device.CreateRootSignature(0, bytes).unwrap();
            let pso: ID3D12PipelineState = device
                .CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                    pRootSignature: std::mem::transmute_copy(&root),
                    CS: D3D12_SHADER_BYTECODE {
                        pShaderBytecode: bytes.as_ptr().cast(),
                        BytecodeLength: bytes.len(),
                    },
                    ..Default::default()
                })
                .unwrap();

            let make_buf = |bytes: u64, heap: D3D12_HEAP_TYPE, flags: D3D12_RESOURCE_FLAGS, state: D3D12_RESOURCE_STATES| {
                let mut res: Option<ID3D12Resource> = None;
                device
                    .CreateCommittedResource(
                        &D3D12_HEAP_PROPERTIES { Type: heap, ..Default::default() },
                        D3D12_HEAP_FLAG_NONE,
                        &D3D12_RESOURCE_DESC {
                            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                            Width: bytes,
                            Height: 1,
                            DepthOrArraySize: 1,
                            MipLevels: 1,
                            Format: DXGI_FORMAT_UNKNOWN,
                            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                            Flags: flags,
                            ..Default::default()
                        },
                        state,
                        None,
                        &mut res,
                    )
                    .unwrap();
                res.unwrap()
            };

            let buffers: Vec<ID3D12Resource> = (0..STREAMS)
                .map(|_| {
                    make_buf(
                        u64::from(ELEMENTS) * 4,
                        D3D12_HEAP_TYPE_DEFAULT,
                        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                        D3D12_RESOURCE_STATE_COMMON,
                    )
                })
                .collect();
            let readback = make_buf(
                u64::from(ELEMENTS) * 4,
                D3D12_HEAP_TYPE_READBACK,
                D3D12_RESOURCE_FLAG_NONE,
                D3D12_RESOURCE_STATE_COPY_DEST,
            );
            let mut ts_heap: Option<ID3D12QueryHeap> = None;
            device
                .CreateQueryHeap(
                    &D3D12_QUERY_HEAP_DESC {
                        Type: D3D12_QUERY_HEAP_TYPE_TIMESTAMP,
                        Count: 4,
                        NodeMask: 0,
                    },
                    &mut ts_heap,
                )
                .unwrap();
            let ts_rb = make_buf(
                32,
                D3D12_HEAP_TYPE_READBACK,
                D3D12_RESOURCE_FLAG_NONE,
                D3D12_RESOURCE_STATE_COPY_DEST,
            );
            (queue, allocator, list, fence, event, ts_freq, root, pso, buffers, readback, ts_heap.unwrap(), ts_rb)
        };

        let rec = Rec::new(raw_list.clone());
        let mut fence_val = 0u64;

        let mut run_mode = |optimized: bool, verify_readback: bool| -> (f64, f64, Vec<u32>) {
            let cpu_start = std::time::Instant::now();
            unsafe {
                allocator.Reset().unwrap();
                rec.Reset(&allocator, None).unwrap();
                rec.EndQuery(&ts_heap, D3D12_QUERY_TYPE_TIMESTAMP, 0);
                rec.SetComputeRootSignature(&root);
                rec.SetPipelineState(&pso);
                for pass in 0..PASSES_PER_STREAM {
                    for (s, buf) in buffers.iter().enumerate() {
                        let c = [pass, ELEMENTS, 2048 + (s as u32) * 256, 0];
                        rec.SetComputeRoot32BitConstants(0, 4, c.as_ptr().cast(), 0);
                        rec.SetComputeRootUnorderedAccessView(1, buf.GetGPUVirtualAddress());
                        rec.Dispatch(ELEMENTS.div_ceil(64), 1, 1);
                        // Between independent tile streams of the same pass, the
                        // title emits a read-only enhanced barrier group.
                        if optimized {
                            if crate::list::barrier_groups_need_uav(1, (&raw const ro_group).cast()) {
                                rec.barriers(&[global_uav()]);
                            }
                        } else {
                            // Old behavior: every barrier_groups(count > 0) emitted
                            // a global NULL UAV barrier, serializing all tile streams.
                            raw_list.ResourceBarrier(&[global_uav()]);
                        }
                    }
                    // Between producer pass `pass` and consumer pass `pass + 1`,
                    // title emits an enhanced UAV barrier + sync + pre-clamp UAV barrier.
                    if optimized {
                        if crate::list::barrier_groups_need_uav(1, (&raw const uav_group).cast()) {
                            rec.barriers(&[global_uav()]);
                        }
                        // Consecutive sync() + pre-clamp UAV barrier collapsed by `Rec::barriers`.
                        rec.barriers(&[global_uav()]);
                        if !rec.is_uav_synced() {
                            rec.barriers(&[global_uav()]);
                        }
                    } else {
                        raw_list.ResourceBarrier(&[global_uav()]);
                        raw_list.ResourceBarrier(&[global_uav()]);
                        raw_list.ResourceBarrier(&[global_uav()]);
                    }
                }
                rec.EndQuery(&ts_heap, D3D12_QUERY_TYPE_TIMESTAMP, 1);
                rec.ResolveQueryData(&ts_heap, D3D12_QUERY_TYPE_TIMESTAMP, 0, 2, &ts_rb, 0);
                if verify_readback {
                    rec.ResourceBarrier(&[crate::host::transition(
                        &buffers[0],
                        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                        D3D12_RESOURCE_STATE_COPY_SOURCE,
                    )]);
                    rec.CopyBufferRegion(&readback, 0, &buffers[0], 0, u64::from(ELEMENTS) * 4);
                    rec.ResourceBarrier(&[crate::host::transition(
                        &buffers[0],
                        D3D12_RESOURCE_STATE_COPY_SOURCE,
                        D3D12_RESOURCE_STATE_COMMON,
                    )]);
                }
                rec.Close().unwrap();
                let cpu_rec_ms = cpu_start.elapsed().as_secs_f64() * 1000.0;
                let cmd: ID3D12CommandList = raw_list.cast().unwrap();
                queue.ExecuteCommandLists(&[Some(cmd)]);
                fence_val += 1;
                queue.Signal(&fence, fence_val).unwrap();
                fence.SetEventOnCompletion(fence_val, event).unwrap();
                WaitForSingleObject(event, INFINITE);

                let mut ts = [0u64; 2];
                let mut ptr: *mut std::ffi::c_void = std::ptr::null_mut();
                ts_rb.Map(0, None, Some(&mut ptr)).unwrap();
                std::ptr::copy_nonoverlapping(ptr.cast::<u64>(), ts.as_mut_ptr(), 2);
                ts_rb.Unmap(0, None);
                let ms = ts[1].saturating_sub(ts[0]) as f64 * 1000.0 / ts_freq as f64;

                let mut out = Vec::new();
                if verify_readback {
                    out.resize(ELEMENTS as usize, 0);
                    readback.Map(0, None, Some(&mut ptr)).unwrap();
                    std::ptr::copy_nonoverlapping(ptr.cast::<u32>(), out.as_mut_ptr(), ELEMENTS as usize);
                    readback.Unmap(0, None);
                }
                (ms, cpu_rec_ms, out)
            }
        };

        let (_, _, baseline_out) = run_mode(false, true);
        let (_, _, opt_out) = run_mode(true, true);
        assert_eq!(baseline_out, opt_out, "filtered/deduped barriers must preserve exact producer-consumer output");

        let median = |mut v: Vec<f64>| {
            v.sort_by(|a, b| a.total_cmp(b));
            v[v.len() / 2]
        };

        for _ in 0..WARMUP {
            run_mode(false, false);
        }
        let before: Vec<_> = (0..ITERS).map(|_| run_mode(false, false)).collect();
        let before_gpu = median(before.iter().map(|x| x.0).collect());
        let before_cpu = median(before.iter().map(|x| x.1).collect());

        for _ in 0..WARMUP {
            run_mode(true, false);
        }
        let after: Vec<_> = (0..ITERS).map(|_| run_mode(true, false)).collect();
        let after_gpu = median(after.iter().map(|x| x.0).collect());
        let after_cpu = median(after.iter().map(|x| x.1).collect());
        let delta_gpu = after_gpu - before_gpu;
        let pct_gpu = (1.0 - after_gpu / before_gpu) * 100.0;
        let pct_cpu = (1.0 - after_cpu / before_cpu) * 100.0;

        println!(
            "adapter={adapter_name} dispatches={} streams={STREAMS} budget_30fps={BUDGET_MS:.3}ms",
            STREAMS as u32 * PASSES_PER_STREAM
        );
        println!(
            "BEFORE (unfiltered + un-deduped UAV) gpu={before_gpu:.4}ms ({:.2}% of 33.3ms) cpu_record={before_cpu:.4}ms",
            before_gpu / BUDGET_MS * 100.0
        );
        println!(
            "AFTER  (filtered + deduped UAV)      gpu={after_gpu:.4}ms ({:.2}% of 33.3ms) cpu_record={after_cpu:.4}ms",
            after_gpu / BUDGET_MS * 100.0
        );
        println!(
            "DELTA gpu={delta_gpu:+.4}ms ({pct_gpu:+.1}%) cpu_record={:+.4}ms ({pct_cpu:+.1}%) bit_exact=true",
            after_cpu - before_cpu
        );
    }
}
