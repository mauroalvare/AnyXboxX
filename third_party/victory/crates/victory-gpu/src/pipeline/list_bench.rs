//! Per-call cost of the D3D12.X list methods as the title calls them
//! (#2586): through the list's method table, with a host device, title
//! buffers in write-watched memory, a shader-visible title view heap, and
//! pipelines whose translations read root views, root constants and
//! descriptor tables as the title's do. Run with
//! `cargo test -p victory-gpu --release list_bench -- --ignored --nocapture`.
//!
//! Each method is timed alone, over many calls on an open list (reset and
//! re-recorded every [`PER_LIST`] calls, as a frame's lists are), then a
//! draw sequence as the title records one, on one thread and on five.

use super::*;
use crate::object::{S_OK, slot};
use std::time::{Duration, Instant};
use victory_shader::emit::{HostShader, UserData};
use victory_shader::{HwStage, ResourceType, RootType, UsedResource};
use windows::Win32::Graphics::Direct3D::{Fxc::D3DCompile, ID3DBlob};
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::core::{GUID, Interface, PCSTR, s};

/// A write-watched allocation of `pages` pages (as the runtime's pools),
/// on a 64 KiB boundary (the allocation granularity). Never freed: the
/// bench's buffers live in it until the process ends.
fn pool(pages: usize) -> usize {
    use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc};
    const MEM_WRITE_WATCH: u32 = 0x20_0000;
    // SAFETY: a fresh allocation.
    let p = unsafe {
        VirtualAlloc(
            std::ptr::null(),
            pages * crate::watch::PAGE,
            MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH,
            PAGE_READWRITE,
        )
    } as usize;
    assert!(p != 0 && p.is_multiple_of(crate::watch::PAGE));
    p
}

/// Calls recorded into one list before it is closed and reset.
const PER_LIST: usize = 2000;

fn compile(source: &str, target: PCSTR) -> Vec<u8> {
    let mut code: Option<ID3DBlob> = None;
    // SAFETY: the arguments are live and `code` is the output pointer.
    unsafe {
        D3DCompile(
            source.as_ptr().cast(),
            source.len(),
            PCSTR::null(),
            None,
            None,
            s!("main"),
            target,
            0,
            0,
            &mut code,
            None,
        )
        .unwrap();
    }
    let code = code.unwrap();
    // SAFETY: the blob owns these bytes.
    unsafe {
        std::slice::from_raw_parts(code.GetBufferPointer().cast::<u8>(), code.GetBufferSize())
    }
    .to_vec()
}

/// A shader whose translation is `bytes` (host code that reads nothing, so
/// any root signature takes it) with the bindings a title shader has: what
/// the draw path does for it on the CPU is the same.
fn preset(
    stage: &'static str,
    hw: HwStage,
    bytes: &[u8],
    user_data: Vec<UserData>,
    root_cbvs: Vec<u8>,
    used: Vec<UsedResource>,
) -> Arc<Shader> {
    let shader = shader(stage, bytes);
    shader.translated.get_or_init(|| {
        Some(Translated {
            host: HostShader {
                stage: hw,
                dxil: bytes.to_vec(),
                user_data,
                samplers: Vec::new(),
                static_samplers: Vec::new(),
                root_cbvs,
                root_srvs: Vec::new(),
                root_uavs: Vec::new(),
                root_indexed_views: Vec::new(),
            },
            threads: (hw == HwStage::Cs).then_some([64, 1, 1]),
            tables: crate::draw::TablePlan::new(&used, &[]),
            used,
            stores: true,
            stores_by_address: false,
            picks: Default::default(),
        })
    });
    shader
}

fn table_entry(kind: ResourceType, slot: u8, k: u32) -> UsedResource {
    UsedResource {
        kind,
        root: RootType::Table,
        table_offset_dwords: 8 * k,
        lower_bound: 0,
        stride: 0,
        dwords: 8,
        user_data: 0,
        slot,
    }
}

fn user(index: u32, slot: u8, root: RootType, dword: u32) -> UserData {
    UserData {
        index,
        slot,
        root,
        dword,
    }
}

fn made(pso: This) -> &'static host::HostPipeline {
    let started = Instant::now();
    loop {
        if let Some(p) = host_pipeline(pso) {
            return p;
        }
        assert!(
            started.elapsed() < Duration::from_secs(60),
            "pipeline not made"
        );
        std::thread::sleep(Duration::from_millis(10));
    }
}

#[repr(C, align(8))]
struct Depth1(u32, D3D12_DEPTH_STENCIL_DESC1);

fn graphics_pipeline(vs: &Arc<Shader>, ps: &Arc<Shader>) -> This {
    #[repr(C)]
    struct Stream {
        vs_kind: u64,
        vs: Bytecode,
        ps_kind: u64,
        ps: Bytecode,
        depth: Depth1,
        topology: [u32; 2],
        dsv: [u32; 2],
    }
    let stream = Stream {
        vs_kind: 1,
        vs: Bytecode {
            code: vs.container.as_ptr(),
            size: vs.container.len(),
        },
        ps_kind: 2,
        ps: Bytecode {
            code: ps.container.as_ptr(),
            size: ps.container.len(),
        },
        depth: Depth1(
            21,
            D3D12_DEPTH_STENCIL_DESC1 {
                DepthEnable: true.into(),
                DepthWriteMask: D3D12_DEPTH_WRITE_MASK_ALL,
                DepthFunc: D3D12_COMPARISON_FUNC_LESS_EQUAL,
                ..Default::default()
            },
        ),
        topology: [14, D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE.0 as u32],
        dsv: [16, DXGI_FORMAT_D32_FLOAT.0 as u32],
    };
    let desc = StreamDesc {
        size: size_of_val(&stream),
        stream: (&raw const stream).cast(),
    };
    let mut pso = std::ptr::null();
    assert_eq!(
        create_stream(std::ptr::null(), &desc, &PIPELINE_IIDS[0], &mut pso),
        S_OK
    );
    made(pso);
    pso
}

fn compute_pipeline(cs: &Arc<Shader>) -> This {
    let head = ComputeHead {
        root: std::ptr::null(),
        stage: Bytecode {
            code: cs.container.as_ptr(),
            size: cs.container.len(),
        },
    };
    let mut pso = std::ptr::null();
    assert_eq!(
        create_compute(std::ptr::null(), &head, &PIPELINE_IIDS[0], &mut pso),
        S_OK
    );
    made(pso);
    pso
}

fn iid(g: GUID) -> Iid {
    guid(g.data1, g.data2, g.data3, g.data4)
}

const RESOURCE_IID: GUID = GUID::from_values(
    0x696442be,
    0xa72e,
    0x4059,
    [0xbc, 0x79, 0x5b, 0x5c, 0x98, 0x04, 0x0f, 0xad],
);

/// A title buffer of `bytes` at `address`.
fn buffer_at(address: u64, bytes: u64) -> This {
    let desc: [u32; 14] = [
        1,
        0,
        0,
        0,
        bytes as u32,
        (bytes >> 32) as u32,
        1,
        1 | 1 << 16,
        0,
        1,
        0,
        1,
        0,
        0,
    ];
    let mut out = std::ptr::null();
    let made = crate::resource::create_at_address(
        std::ptr::null(),
        address,
        desc.as_ptr().cast(),
        0,
        std::ptr::null(),
        &iid(RESOURCE_IID),
        &mut out,
    );
    assert_eq!(made, S_OK);
    out
}

/// A 64x64 RGBA8 title texture at `address`.
fn texture_at(address: u64) -> This {
    let words: [u32; 14] = [3, 0, 0, 0, 64, 0, 64, 1 | 1 << 16, 28, 1, 0, 0, 0, 0];
    let mut out = std::ptr::null();
    let made = crate::resource::create_at_address(
        std::ptr::null(),
        address,
        words.as_ptr().cast(),
        0,
        std::ptr::null(),
        &iid(RESOURCE_IID),
        &mut out,
    );
    assert_eq!(made, S_OK);
    out
}

/// Title memory of `bytes`, 64 KiB aligned, not write-watched.
fn title_memory(bytes: usize) -> u64 {
    let layout = std::alloc::Layout::from_size_align(bytes, 1 << 16).unwrap();
    // SAFETY: a non-zero size; leaked, as title memory outlives a test.
    unsafe { std::alloc::alloc_zeroed(layout) as u64 }
}

/// What every timed list shares: the title's objects.
struct Scene {
    heap: This,
    heap_start: u64,
    /// Constant ring (write-watched), index buffer, compute output.
    ring: u64,
    index: u64,
    out: u64,
    args: This,
    args_at: u64,
    textures: Vec<This>,
    draw: [This; 2],
    dispatch: This,
    indexed_signature: This,
    dispatch_signature: This,
}

// SAFETY: the title objects are free-threaded, as the title uses them.
unsafe impl Sync for Scene {}

const RING_PAGES: usize = 16;

impl Scene {
    fn new() -> Scene {
        let null = std::ptr::null();
        let vs_code = compile(
            "float4 main(uint id : SV_VertexID) : SV_Position { return float4(id, 0, 0, 1); }",
            s!("vs_5_1"),
        );
        let ps_code = |k: u32| compile(&format!("void main() {{ float x = {k}; }}"), s!("ps_5_1"));
        let cs_code = compile("[numthreads(64, 1, 1)] void main() {}", s!("cs_5_1"));
        // Vertex shader: frame and object constants as root views, four
        // root constants. Pixel shader: six textures through a table, frame
        // constants. As the title's (title root slots 0-3).
        let vs = preset(
            "vertex",
            HwStage::Vs,
            &vs_code,
            (0..4).map(|d| user(d, 3, RootType::Constants, d)).collect(),
            vec![0, 1],
            Vec::new(),
        );
        let ps: Vec<_> = (0..2)
            .map(|k| {
                preset(
                    "pixel",
                    HwStage::Ps,
                    &ps_code(k),
                    vec![user(0, 2, RootType::Table, 0)],
                    vec![0],
                    (0..6)
                        .map(|t| table_entry(ResourceType::Srv, 2, t))
                        .collect(),
                )
            })
            .collect();
        // Compute: a texture and an unordered buffer through a table, a
        // root constant buffer, two root constants.
        let cs = preset(
            "compute",
            HwStage::Cs,
            &cs_code,
            vec![
                user(0, 0, RootType::Table, 0),
                user(1, 2, RootType::Constants, 0),
                user(2, 2, RootType::Constants, 1),
            ],
            vec![1],
            vec![
                table_entry(ResourceType::Srv, 0, 0),
                table_entry(ResourceType::Uav, 0, 1),
            ],
        );
        let draw = [
            graphics_pipeline(&vs, &ps[0]),
            graphics_pipeline(&vs, &ps[1]),
        ];
        let dispatch = compute_pipeline(&cs);

        let ring = pool(RING_PAGES) as u64;
        buffer_at(ring, (RING_PAGES * crate::watch::PAGE) as u64);
        let index = pool(1) as u64;
        buffer_at(index, crate::watch::PAGE as u64);
        let out = pool(1) as u64;
        let out_buffer = buffer_at(out, crate::watch::PAGE as u64);
        let args_at = pool(1) as u64;
        let args = buffer_at(args_at, crate::watch::PAGE as u64);
        // SAFETY: the pool's first page: an indexed draw (36 indices, one
        // instance) and a dispatch (64 groups).
        unsafe {
            (args_at as *mut [u32; 5]).write([36, 1, 0, 0, 0]);
            ((args_at + 0x100) as *mut [u32; 3]).write([64, 1, 1]);
        }
        let memory = title_memory(8 << 16);
        let textures: Vec<This> = (0..8).map(|k| texture_at(memory + (k << 16))).collect();

        let heap_desc: [u32; 8] = [0, 4096, 1, 0, 0, 0, 0, 0];
        let mut heap = null;
        let heap_iid = iid(ID3D12DescriptorHeap::IID);
        assert_eq!(
            crate::descriptor::create_heap(null, heap_desc.as_ptr().cast(), &heap_iid, &mut heap),
            S_OK
        );
        let heap_start = crate::descriptor::heap(heap).start;
        let srv: [u32; 10] = [28, 4, 0x1688, 0, 0, 1, 0, 0, 0, 0];
        // Slots 0-7: the textures; 8: the compute output (raw).
        for (k, &t) in textures.iter().enumerate() {
            crate::descriptor::create_srv(null, t, &srv, heap_start + 32 * k as u64);
        }
        // `VICTORY_LIST_BENCH_PLACED=1`: the draws' six textures are views
        // the title places in its memory without a resource object (device
        // 0x230), as its streamed textures are.
        if placed_views() {
            let placed = title_memory(6 << 16);
            let words: [u32; 14] = [3, 0, 0, 0, 64, 0, 64, 1 | 1 << 16, 28, 1, 0, 0, 0, 0];
            for k in 0..6u64 {
                let at = placed + (k << 16);
                let view: [u32; 16] = [28, 4, 0x1688, 0, 0, 1, 0, 0, 0, 0, at as u32, (at >> 32) as u32, 0, 0, 0, 0];
                crate::descriptor::create_placed_srv(null, &words, &view, heap_start + 32 * k);
            }
        }
        let raw_uav: [u32; 10] = [39, 1, 0, 0, (crate::watch::PAGE / 4) as u32, 0, 0, 0, 1, 0];
        crate::descriptor::create_uav(null, out_buffer, null, &raw_uav, heap_start + 32 * 8);
        // Compute table at slot 7: the last texture, then the output.
        crate::descriptor::create_srv(null, textures[7], &srv, heap_start + 32 * 7);

        let signature = |kind: u32, stride: u32| {
            #[repr(C)]
            struct Desc {
                stride: u32,
                count: u32,
                arguments: *const [u32; 4],
                nodes: u32,
            }
            let argument = [kind, 0, 0, 0];
            let desc = Desc {
                stride,
                count: 1,
                arguments: &argument,
                nodes: 0,
            };
            let mut out = null;
            let made = crate::signature::create(
                null,
                (&raw const desc).cast(),
                null,
                &iid(ID3D12CommandSignature::IID),
                &mut out,
            );
            assert_eq!(made, S_OK);
            out
        };
        Scene {
            heap,
            heap_start,
            ring,
            index,
            out,
            args,
            args_at,
            textures,
            draw,
            dispatch,
            indexed_signature: signature(1, 20),
            dispatch_signature: signature(2, 12),
        }
    }
}

/// A title list and its allocator.
struct TitleList {
    list: This,
    allocator: This,
    newer: bool,
}

type Release = extern "system" fn(This) -> u32;

impl TitleList {
    fn new(pipeline: This) -> TitleList {
        let null = std::ptr::null();
        let (mut allocator, mut list) = (null, null);
        assert_eq!(
            crate::list::create_allocator(
                null,
                0,
                &iid(ID3D12CommandAllocator::IID),
                &mut allocator
            ),
            S_OK
        );
        assert_eq!(
            crate::list::create_list(
                null,
                0,
                0,
                allocator,
                pipeline,
                &iid(ID3D12GraphicsCommandList::IID),
                &mut list
            ),
            S_OK
        );
        TitleList {
            list,
            allocator,
            newer: crate::list::has_newer_table(list),
        }
    }

    fn at<F: Copy>(&self, older: usize) -> F {
        let offset = if self.newer && older >= 0x1d8 {
            older + 0x10
        } else {
            older
        };
        slot::<F>(self.list, offset)
    }

    fn close_and_reset(&self, pipeline: This) {
        assert_eq!(
            slot::<extern "system" fn(This) -> HRESULT>(self.list, 0x48)(self.list),
            S_OK
        );
        assert_eq!(
            slot::<extern "system" fn(This, This, This) -> HRESULT>(self.list, 0x50)(
                self.list,
                self.allocator,
                pipeline
            ),
            S_OK
        );
    }
}

impl Drop for TitleList {
    fn drop(&mut self) {
        slot::<Release>(self.list, 0x10)(self.list);
        slot::<Release>(self.allocator, 0x10)(self.allocator);
    }
}

#[repr(C)]
struct IndexView {
    address: u64,
    size: u32,
    format: u32,
}

#[repr(C)]
struct VertexView {
    address: u64,
    size: u32,
    stride: u32,
}

/// The title's method types, by the older layout's offsets.
type Heaps = extern "system" fn(This, u32, *const This);
type SetPipeline = extern "system" fn(This, This);
type Table = extern "system" fn(This, u32, u64);
type View = extern "system" fn(This, u32, u64);
type Constant = extern "system" fn(This, u32, u32, u32);
type Constants = extern "system" fn(This, u32, u32, *const u32, u32);
type IndexBuffer = extern "system" fn(This, *const IndexView);
type VertexBuffers = extern "system" fn(This, u32, u32, *const VertexView);
type Viewports = extern "system" fn(This, u32, *const D3D12_VIEWPORT);
type Scissors = extern "system" fn(This, u32, *const windows::Win32::Foundation::RECT);
type Topology = extern "system" fn(This, u32);
type DrawIndexed = extern "system" fn(This, u32, u32, u32, i32, u32);
type Dispatch = extern "system" fn(This, u32, u32, u32);
type Barrier = extern "system" fn(This, u32, *const [u64; 4]) -> u32;
type Indirect = extern "system" fn(This, This, u32, This, u64, This, u64);

/// A title list set up for draws: heaps, pipeline, topology, viewport,
/// tables and root views as a pass sets them.
fn graphics_list(scene: &Scene) -> TitleList {
    let l = TitleList::new(scene.draw[0]);
    set_up_graphics(&l, scene);
    l
}

fn set_up_graphics(l: &TitleList, scene: &Scene) {
    let list = l.list;
    l.at::<Heaps>(0x100)(list, 1, &scene.heap);
    l.at::<SetPipeline>(0xc8)(list, scene.draw[0]);
    l.at::<Topology>(0xa0)(list, 4);
    l.at::<Viewports>(0xa8)(
        list,
        1,
        &D3D12_VIEWPORT {
            Width: 1920.0,
            Height: 1080.0,
            MaxDepth: 1.0,
            ..Default::default()
        },
    );
    l.at::<Table>(0x120)(list, 2, scene.heap_start);
    l.at::<View>(0x150)(list, 0, scene.ring);
    l.at::<View>(0x150)(list, 1, scene.ring + 0x100);
    l.at::<Constants>(0x140)(list, 3, 4, [1u32, 2, 3, 4].as_ptr(), 0);
    l.at::<IndexBuffer>(0x178)(
        list,
        &IndexView {
            address: scene.index,
            size: 72,
            format: DXGI_FORMAT_R16_UINT.0 as u32,
        },
    );
}

fn compute_list(scene: &Scene) -> TitleList {
    let l = TitleList::new(scene.dispatch);
    set_up_compute(&l, scene);
    l
}

fn set_up_compute(l: &TitleList, scene: &Scene) {
    let list = l.list;
    l.at::<Heaps>(0x100)(list, 1, &scene.heap);
    l.at::<SetPipeline>(0xc8)(list, scene.dispatch);
    l.at::<Table>(0x118)(list, 0, scene.heap_start + 32 * 7);
    l.at::<View>(0x148)(list, 1, scene.ring);
    l.at::<Constants>(0x138)(list, 2, 2, [64u32, 1].as_ptr(), 0);
}

/// Nanoseconds a call of `call(list, i)`, over `calls` calls, the list
/// closed, reset and set up again every [`PER_LIST`] (not timed).
fn time(
    l: &TitleList,
    scene: &Scene,
    calls: usize,
    compute: bool,
    mut call: impl FnMut(This, usize),
) -> f64 {
    let pipeline = if compute {
        scene.dispatch
    } else {
        scene.draw[0]
    };
    let mut total = Duration::ZERO;
    let mut done = 0;
    while done < calls {
        let n = PER_LIST.min(calls - done);
        let started = Instant::now();
        for i in done..done + n {
            call(l.list, i);
        }
        total += started.elapsed();
        done += n;
        l.close_and_reset(pipeline);
        if compute {
            set_up_compute(l, scene);
        } else {
            set_up_graphics(l, scene);
        }
    }
    total.as_nanos() as f64 / calls as f64
}

/// The fastest of `runs` timings (the machine's noise only adds).
fn best(runs: usize, mut f: impl FnMut() -> f64) -> f64 {
    (0..runs).map(|_| f()).fold(f64::INFINITY, f64::min)
}

/// One draw as the title records it in a pass: object constants (a ring
/// position of their own), four root constants, the index buffer, then the
/// draw: four calls.
fn draw_sequence(l: &TitleList, scene: &Scene, list: This, i: usize) {
    let ring_bytes = (RING_PAGES * crate::watch::PAGE) as u64;
    l.at::<View>(0x150)(
        list,
        1,
        scene.ring + 0x100 + (i as u64 * 0x100) % (ring_bytes - 0x200),
    );
    l.at::<Constants>(0x140)(list, 3, 4, [i as u32, 2, 3, 4].as_ptr(), 0);
    l.at::<IndexBuffer>(0x178)(
        list,
        &IndexView {
            address: scene.index,
            size: 72,
            format: DXGI_FORMAT_R16_UINT.0 as u32,
        },
    );
    l.at::<DrawIndexed>(0x68)(list, 36, 1, 0, 0, 0);
}

#[test]
#[ignore = "timing; run alone on a quiet machine with --release --ignored --nocapture"]
fn list_bench() {
    let Some(host) = crate::host::host() else {
        eprintln!("list_bench skipped: no host D3D12 device");
        return;
    };
    // SAFETY: plain adapter queries on a live device.
    let adapter = unsafe {
        use windows::Win32::Graphics::Dxgi::{
            CreateDXGIFactory2, DXGI_CREATE_FACTORY_FLAGS, IDXGIAdapter1, IDXGIFactory4,
        };
        let factory: IDXGIFactory4 = CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0)).unwrap();
        let adapter: IDXGIAdapter1 = factory
            .EnumAdapterByLuid(host.device.GetAdapterLuid())
            .unwrap();
        adapter.GetDesc1().unwrap()
    };
    let name = String::from_utf16_lossy(&adapter.Description);
    let scene = Scene::new();
    let runs = 5;
    let calls = 20_000;
    if std::env::var_os("VICTORY_LIST_BENCH_SCALING").is_some() {
        scaling(&scene, runs, calls);
        return;
    }
    if std::env::var_os("VICTORY_LIST_BENCH_INDIRECT").is_some() {
        indirect_scaling(&scene, runs, calls / 4);
        return;
    }
    let row = |name: &str, ns: f64| {
        eprintln!("{name:<58} {ns:>9.0} ns");
    };
    eprintln!(
        "list_bench on {} ({calls} calls a row, best of {runs})",
        name.trim_end_matches('\0')
    );

    let g = graphics_list(&scene);
    let c = compute_list(&scene);
    let ring_bytes = (RING_PAGES * crate::watch::PAGE) as u64;
    row(
        "SetGraphicsRootConstantBufferView (0x150)",
        best(runs, || {
            time(&g, &scene, calls, false, |list, i| {
                g.at::<View>(0x150)(list, 1, scene.ring + (i as u64 * 0x100) % ring_bytes)
            })
        }),
    );
    row(
        "SetGraphicsRootDescriptorTable (0x120)",
        best(runs, || {
            time(&g, &scene, calls, false, |list, i| {
                g.at::<Table>(0x120)(list, 2, scene.heap_start + 32 * (i as u64 % 2))
            })
        }),
    );
    row(
        "SetGraphicsRoot32BitConstant (0x130)",
        best(runs, || {
            time(&g, &scene, calls, false, |list, i| {
                g.at::<Constant>(0x130)(list, 3, i as u32, 1)
            })
        }),
    );
    row(
        "SetGraphicsRoot32BitConstants, 4 dwords (0x140)",
        best(runs, || {
            time(&g, &scene, calls, false, |list, i| {
                g.at::<Constants>(0x140)(list, 3, 4, [i as u32, 2, 3, 4].as_ptr(), 0)
            })
        }),
    );
    row(
        "SetComputeRoot32BitConstants, 2 dwords (0x138)",
        best(runs, || {
            time(&c, &scene, calls, true, |list, i| {
                c.at::<Constants>(0x138)(list, 2, 2, [i as u32, 1].as_ptr(), 0)
            })
        }),
    );
    row(
        "IASetIndexBuffer (0x178)",
        best(runs, || {
            time(&g, &scene, calls, false, |list, _| {
                g.at::<IndexBuffer>(0x178)(
                    list,
                    &IndexView {
                        address: scene.index,
                        size: 72,
                        format: DXGI_FORMAT_R16_UINT.0 as u32,
                    },
                )
            })
        }),
    );
    row(
        "IASetVertexBuffers, 2 views (0x180)",
        best(runs, || {
            let views = [
                VertexView {
                    address: scene.ring,
                    size: 4096,
                    stride: 16,
                },
                VertexView {
                    address: scene.ring + 4096,
                    size: 4096,
                    stride: 8,
                },
            ];
            time(&g, &scene, calls, false, |list, _| {
                g.at::<VertexBuffers>(0x180)(list, 0, 2, views.as_ptr())
            })
        }),
    );
    row(
        "SetPipelineState (0xc8)",
        best(runs, || {
            time(&g, &scene, calls, false, |list, i| {
                g.at::<SetPipeline>(0xc8)(list, scene.draw[i % 2])
            })
        }),
    );
    row(
        "SetDescriptorHeaps, 1 heap (0x100)",
        best(runs, || {
            time(&g, &scene, calls, false, |list, _| {
                g.at::<Heaps>(0x100)(list, 1, &scene.heap)
            })
        }),
    );
    row(
        "RSSetViewports, 1 (0xa8)",
        best(runs, || {
            let v = D3D12_VIEWPORT {
                Width: 1920.0,
                Height: 1080.0,
                MaxDepth: 1.0,
                ..Default::default()
            };
            time(&g, &scene, calls, false, |list, _| {
                g.at::<Viewports>(0xa8)(list, 1, &v)
            })
        }),
    );
    row(
        "RSSetScissorRects, 1 (0xb0)",
        best(runs, || {
            let r = windows::Win32::Foundation::RECT {
                left: 0,
                top: 0,
                right: 1920,
                bottom: 1080,
            };
            time(&g, &scene, calls, false, |list, _| {
                g.at::<Scissors>(0xb0)(list, 1, &r)
            })
        }),
    );
    row(
        "ResourceBarrier, 1 texture transition (0xd0)",
        best(runs, || {
            time(&g, &scene, calls, false, |list, i| {
                let (before, after) = if i % 2 == 0 {
                    (0x40u64 | 0x80, 0x400u64)
                } else {
                    (0x400, 0x40 | 0x80)
                };
                let b = [
                    0u64,
                    scene.textures[0] as u64,
                    0xffff_ffff | before << 32,
                    after,
                ];
                g.at::<Barrier>(0xd0)(list, 1, &b);
            })
        }),
    );
    row(
        "ResourceBarrier, 1 buffer UAV (0xd0)",
        best(runs, || {
            time(&g, &scene, calls, false, |list, _| {
                let b = [2u64, scene.args as u64, 0, 0];
                g.at::<Barrier>(0xd0)(list, 1, &b);
            })
        }),
    );
    let drawn = crate::draw::drawn();
    row(
        "DrawIndexedInstanced, nothing changed (0x68)",
        best(runs, || {
            time(&g, &scene, calls, false, |list, _| {
                g.at::<DrawIndexed>(0x68)(list, 36, 1, 0, 0, 0)
            })
        }),
    );
    assert!(
        crate::draw::drawn() - drawn >= (runs * calls) as u64,
        "draws ran on the host"
    );
    let sequence = best(runs, || {
        time(&g, &scene, calls, false, |list, i| {
            draw_sequence(&g, &scene, list, i)
        })
    });
    row(
        "draw sequence (CBV, 4 constants, index buffer, draw): per draw",
        sequence,
    );
    row("  the same, per call", sequence / 4.0);
    row(
        "DrawIndexedInstanced after a pipeline change (0xc8 + 0x68)",
        best(runs, || {
            time(&g, &scene, calls, false, |list, i| {
                g.at::<SetPipeline>(0xc8)(list, scene.draw[i % 2]);
                g.at::<DrawIndexed>(0x68)(list, 36, 1, 0, 0, 0);
            })
        }),
    );
    row(
        "Dispatch, nothing changed (0x70)",
        best(runs, || {
            time(&c, &scene, calls, true, |list, _| {
                c.at::<Dispatch>(0x70)(list, 64, 1, 1)
            })
        }),
    );
    row(
        "Dispatch after 2 root constants (0x138 + 0x70)",
        best(runs, || {
            time(&c, &scene, calls, true, |list, i| {
                c.at::<Constants>(0x138)(list, 2, 2, [i as u32, 1].as_ptr(), 0);
                c.at::<Dispatch>(0x70)(list, 64, 1, 1);
            })
        }),
    );
    row(
        "ExecuteIndirect, 1 indexed draw (0x1d8)",
        best(runs, || {
            time(&g, &scene, calls / 4, false, |list, _| {
                g.at::<Indirect>(0x1d8)(
                    list,
                    scene.indexed_signature,
                    1,
                    scene.args,
                    0,
                    std::ptr::null(),
                    0,
                )
            })
        }),
    );
    row(
        "ExecuteIndirect, 1 dispatch (0x1d8)",
        best(runs, || {
            time(&c, &scene, calls / 4, true, |list, _| {
                c.at::<Indirect>(0x1d8)(
                    list,
                    scene.dispatch_signature,
                    1,
                    scene.args,
                    0x100,
                    std::ptr::null(),
                    0,
                )
            })
        }),
    );
    parts(&scene, &g);

    scaling(&scene, runs, calls);
    let _ = (scene.out, scene.args_at);
    drop(g);
    drop(c);
}

/// Whether the draws read placed views (`VICTORY_LIST_BENCH_PLACED=1`).
fn placed_views() -> bool {
    std::env::var_os("VICTORY_LIST_BENCH_PLACED").is_some()
}

/// The draw sequence on one thread and on five at once, each recording its
/// own list of draws, as the world's render threads do. Alone (no other
/// rows) with `VICTORY_LIST_BENCH_SCALING=1`.
fn scaling(scene: &Scene, runs: usize, calls: usize) {
    for threads in [1usize, 5] {
        let started = std::sync::Barrier::new(threads);
        let per_thread: Vec<f64> = std::thread::scope(|s| {
            let handles: Vec<_> = (0..threads)
                .map(|_| {
                    s.spawn(|| {
                        let l = graphics_list(scene);
                        started.wait();
                        best(runs, || {
                            time(&l, scene, calls, false, |list, i| {
                                draw_sequence(&l, scene, list, i)
                            })
                        })
                    })
                })
                .collect();
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        });
        let mean = per_thread.iter().sum::<f64>() / threads as f64;
        eprintln!(
            "draw sequence{} on {threads} thread(s) at once: {mean:.0} ns a draw, {:.0} ns a call (per thread)",
            if placed_views() { " (placed views)" } else { "" },
            mean / 4.0
        );
    }
}

/// The list's close as the title calls it, timed with the calls: work a
/// list defers to its end counts.
fn close_list(l: &TitleList) {
    assert_eq!(
        slot::<extern "system" fn(This) -> HRESULT>(l.list, 0x48)(l.list),
        S_OK
    );
}

fn reset_list(l: &TitleList, pipeline: This) {
    let reset = slot::<extern "system" fn(This, This, This) -> HRESULT>(l.list, 0x50);
    assert_eq!(reset(l.list, l.allocator, pipeline), S_OK);
}

/// ExecuteIndirect (one indexed draw, one dispatch) on one thread and on
/// five at once, each thread its own list, every [`PER_LIST`] calls the
/// list's close timed with them. Alone with `VICTORY_LIST_BENCH_INDIRECT=1`.
fn indirect_scaling(scene: &Scene, runs: usize, calls: usize) {
    for compute in [false, true] {
        for threads in [1usize, 5] {
            let started = std::sync::Barrier::new(threads);
            let per_thread: Vec<f64> = std::thread::scope(|s| {
                let handles: Vec<_> = (0..threads)
                    .map(|_| {
                        s.spawn(|| {
                            let (l, pipeline) = if compute {
                                (compute_list(scene), scene.dispatch)
                            } else {
                                (graphics_list(scene), scene.draw[0])
                            };
                            let (signature, offset) = if compute {
                                (scene.dispatch_signature, 0x100)
                            } else {
                                (scene.indexed_signature, 0)
                            };
                            started.wait();
                            best(runs, || {
                                let mut total = Duration::ZERO;
                                let mut done = 0;
                                while done < calls {
                                    let n = PER_LIST.min(calls - done);
                                    let at = Instant::now();
                                    for _ in 0..n {
                                        l.at::<Indirect>(0x1d8)(
                                            l.list,
                                            signature,
                                            1,
                                            scene.args,
                                            offset,
                                            std::ptr::null(),
                                            0,
                                        );
                                    }
                                    close_list(&l);
                                    total += at.elapsed();
                                    done += n;
                                    reset_list(&l, pipeline);
                                    if compute {
                                        set_up_compute(&l, scene);
                                    } else {
                                        set_up_graphics(&l, scene);
                                    }
                                }
                                total.as_nanos() as f64 / calls as f64
                            })
                        })
                    })
                    .collect();
                handles.into_iter().map(|h| h.join().unwrap()).collect()
            });
            let mean = per_thread.iter().sum::<f64>() / threads as f64;
            let what = if compute { "dispatch" } else { "indexed draw" };
            eprintln!(
                "ExecuteIndirect, 1 {what}, with its list's close, on {threads} thread(s) at once: {mean:.0} ns a call (per thread)"
            );
        }
    }
}

/// What a draw's work is made of: each piece of a table entry's and a root
/// view's work, timed alone (warm: after the draws above).
fn parts(scene: &Scene, g: &TitleList) {
    let n = 200_000;
    let each = |name: &str, mut f: Box<dyn FnMut() + '_>| {
        let ns = best(5, || {
            let started = Instant::now();
            for _ in 0..n {
                f();
            }
            started.elapsed().as_nanos() as f64 / n as f64
        });
        eprintln!("  part {name:<54} {ns:>7.1} ns");
    };
    let list = crate::list::host_list(g.list).unwrap();
    let t = scene.textures[0];
    let tex = crate::resource::host_resource(t).unwrap();
    let handle = scene.heap_start;
    let view = crate::descriptor::view(handle).unwrap();
    let host_heap = crate::descriptor::heap(scene.heap).host().unwrap();
    each(
        "descriptor::view (VIEWS lock, clone)",
        Box::new(|| {
            std::hint::black_box(crate::descriptor::view(std::hint::black_box(handle)));
        }),
    );
    each(
        "HostHeap::mirror, unchanged texture slot",
        Box::new(|| {
            std::hint::black_box(host_heap.mirror(0, handle, true));
        }),
    );
    each(
        "descriptor::host::placed_texture",
        Box::new(|| {
            std::hint::black_box(crate::descriptor::host::placed_texture(&view));
        }),
    );
    each(
        "descriptor::host::buffer_copy (texture view)",
        Box::new(|| {
            std::hint::black_box(crate::descriptor::host::buffer_copy(&view));
        }),
    );
    each(
        "resource::alias_of",
        Box::new(|| {
            std::hint::black_box(crate::resource::alias_of(std::hint::black_box(t)));
        }),
    );
    each(
        "resource::host_resource (+ drop)",
        Box::new(|| {
            std::hint::black_box(crate::resource::host_resource(std::hint::black_box(t)));
        }),
    );
    each(
        "resource::host_copy (+ drop)",
        Box::new(|| {
            std::hint::black_box(crate::resource::host_copy(std::hint::black_box(t)));
        }),
    );
    each(
        "texture::title_written (GetDesc)",
        Box::new(|| {
            std::hint::black_box(crate::texture::title_written(tex));
        }),
    );
    each(
        "texture::upload, second use in the list",
        Box::new(|| crate::texture::upload(list, t, tex)),
    );
    each(
        "list::ensure_state, state unchanged",
        Box::new(|| crate::list::ensure_state(list, t, 0x40 | 0x80)),
    );
    each(
        "shadow::find (BUFFERS lock, range)",
        Box::new(|| {
            std::hint::black_box(crate::shadow::find(std::hint::black_box(scene.ring), 1));
        }),
    );
    let shadow = crate::shadow::find(scene.ring, 1).unwrap();
    each(
        "Shadow::refresh, range already refreshed",
        Box::new(|| shadow.refresh(list, scene.ring, 256)),
    );
    each(
        "Instant::now",
        Box::new(|| {
            std::hint::black_box(Instant::now());
        }),
    );
}
