use std::ptr::null;

use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
use windows::Win32::Graphics::Direct3D::ID3DBlob;
use windows::Win32::Graphics::Direct3D12::*;
use windows::core::{Interface, s};

use super::*;
use crate::descriptor::{create_cbv, create_heap, heap};
use crate::host::{Host, HostList, host};
use crate::object::{S_OK, guid, slot};

const CBV_COPY_HLSL: &str = r#"
#define RS "DescriptorTable(CBV(b0)), UAV(u0)"
cbuffer Input : register(b0) { uint4 value; };
RWByteAddressBuffer Output : register(u0);
[RootSignature(RS)]
[numthreads(1, 1, 1)]
void main() {
    Output.Store(0, value.x);
}
"#;

struct TitleHeap(This);

impl TitleHeap {
    fn new(slots: u32) -> Self {
        let desc = [
            D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV.0 as u32,
            slots,
            D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE.0 as u32,
            0,
            0,
            0,
            0,
            0,
        ];
        let i = ID3D12DescriptorHeap::IID;
        let iid = guid(i.data1, i.data2, i.data3, i.data4);
        let mut out = null();
        assert_eq!(create_heap(null(), desc.as_ptr().cast(), &iid, &mut out), S_OK);
        Self(out)
    }
}

impl Drop for TitleHeap {
    fn drop(&mut self) {
        slot::<extern "system" fn(This) -> u32>(self.0, 0x10)(self.0);
    }
}

fn compile_cbv_copy(device: &ID3D12Device) -> (ID3D12RootSignature, ID3D12PipelineState) {
    let mut code: Option<ID3DBlob> = None;
    let mut errors: Option<ID3DBlob> = None;
    // SAFETY: static HLSL source and out pointers owned here.
    let compiled = unsafe {
        D3DCompile(
            CBV_COPY_HLSL.as_ptr().cast(),
            CBV_COPY_HLSL.len(),
            s!("cut_cbv_copy"),
            None,
            None,
            s!("main"),
            s!("cs_5_1"),
            0,
            0,
            &mut code,
            Some(&mut errors),
        )
    };
    if let Err(e) = compiled {
        let msg = errors
            .map(|b| {
                // SAFETY: error blob bytes.
                let bytes = unsafe {
                    std::slice::from_raw_parts(b.GetBufferPointer().cast::<u8>(), b.GetBufferSize())
                };
                String::from_utf8_lossy(bytes).into_owned()
            })
            .unwrap_or_default();
        panic!("D3DCompile failed ({e}): {msg}");
    }
    let blob = code.expect("compiled shader blob");
    // SAFETY: valid compiled DXBC blob with embedded root signature.
    let bytes = unsafe {
        std::slice::from_raw_parts(blob.GetBufferPointer().cast::<u8>(), blob.GetBufferSize())
    };
    // SAFETY: valid root signature and compute PSO description.
    unsafe {
        let root: ID3D12RootSignature = device.CreateRootSignature(0, bytes).expect("root signature");
        let desc = D3D12_COMPUTE_PIPELINE_STATE_DESC {
            pRootSignature: std::mem::transmute_copy(&root),
            CS: D3D12_SHADER_BYTECODE {
                pShaderBytecode: bytes.as_ptr().cast(),
                BytecodeLength: bytes.len(),
            },
            ..Default::default()
        };
        let pso: ID3D12PipelineState = device.CreateComputePipelineState(&desc).expect("compute PSO");
        (root, pso)
    }
}

fn record_cbv_dispatch(
    host: &Host,
    list: &HostList,
    root: &ID3D12RootSignature,
    pso: &ID3D12PipelineState,
    host_heap: &crate::descriptor::host::HostHeap,
    cbv_slot: u32,
) -> ID3D12Resource {
    let out_uav = host.uav_buffer(256).expect("uav output buffer");
    let readback = host.read_back_buffer(256).expect("readback buffer");
    let gpu_handle = host_heap.gpu_handle(cbv_slot);
    let uav_barrier = D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
        ..Default::default()
    };
    let to_copy = crate::host::transition(
        &out_uav,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        D3D12_RESOURCE_STATE_COPY_SOURCE,
    );
    // SAFETY: open command list with live resources, heap, root signature, and PSO.
    unsafe {
        list.list.SetDescriptorHeaps(&[Some(host_heap.heap.clone())]);
        list.list.SetComputeRootSignature(root);
        list.list.SetPipelineState(pso);
        list.list.SetComputeRootDescriptorTable(0, gpu_handle);
        list.list.SetComputeRootUnorderedAccessView(1, out_uav.GetGPUVirtualAddress());
        list.list.Dispatch(1, 1, 1);
        list.list.ResourceBarrier(&[uav_barrier, to_copy]);
        list.list.CopyBufferRegion(&readback, 0, &out_uav, 0, 4);
    }
    list.keep(&out_uav);
    list.keep(&readback);
    readback
}

fn read_u32(readback: &ID3D12Resource) -> u32 {
    let mut ptr = std::ptr::null_mut();
    let range = D3D12_RANGE { Begin: 0, End: 4 };
    // SAFETY: submission completed; buffer is at least 256 bytes.
    unsafe {
        readback.Map(0, Some(&range), Some(&mut ptr)).expect("map readback");
        let value = ptr.cast::<u32>().read_unaligned();
        readback.Unmap(0, None);
        value
    }
}

fn cbv_used() -> [UsedResource; 1] {
    [UsedResource {
        kind: ResourceType::Cbv,
        root: RootType::Table,
        slot: 0,
        table_offset_dwords: 0,
        lower_bound: 0,
        stride: 0,
        dwords: 4,
        user_data: 0,
    }]
}

/// A command list cut onto a new `HostList` (`State::replay`) must not reuse
/// `fresh_tables` CBV slots pointing into the previous `HostList`'s scratch
/// upload memory: once part 1 runs and recycles its scratch chunks, part 2
/// would otherwise read overwritten or stale constants on the GPU.
#[test]
fn cut_list_replayed_cbv_table_reads_updated_constants_after_part1_recycles() {
    let Some(host) = host() else {
        return;
    };
    let (root, pso) = compile_cbv_copy(&host.device);
    let title_heap = TitleHeap::new(64);
    let h = heap(title_heap.0);
    let host_heap = h.host().expect("host view heap");
    let table_handle = h.handle(0).expect("slot 0 handle");

    let mut cb_mem = Box::new([0u32; 64]);
    cb_mem[0] = 0x1111_2222;
    let cb_addr = cb_mem.as_ptr() as u64;
    let cb_desc = [cb_addr as u32, (cb_addr >> 32) as u32, 256, 0];
    create_cbv(null(), &cb_desc, table_handle);

    let plan = TablePlan::new(&cbv_used(), &[]);
    let mut state = State::default();
    state.set_heaps(&[title_heap.0]);
    state.set_table(true, 0, table_handle);

    // Part 1: before the cut.
    let list1 = host.new_list().expect("list1");
    let made1 = state.tables(&list1, &plan, true);
    assert_eq!(made1.len(), 1);
    let slot1 = made1[0].1 / 32;
    let readback1 = record_cbv_dispatch(host, &list1, &root, &pso, host_heap, slot1);
    list1.close();

    // Cut onto a fresh HostList (`list::cut` / `list::value_written` calls `State::replay`).
    let list2 = host.new_list().expect("list2");
    state.replay(&list2);
    let made2 = state.tables(&list2, &plan, true);
    assert_eq!(made2.len(), 1);
    let slot2 = made2[0].1 / 32;
    let readback2 = record_cbv_dispatch(host, &list2, &root, &pso, host_heap, slot2);
    list2.close();

    // Part 1 executes first and is recycled while part 2 waits on a memory value.
    host.execute(&[&list1]);
    let got1 = read_u32(&readback1);
    assert_eq!(got1, 0x1111_2222, "part 1 must read initial CBV value (got {got1:#010x})");
    list1.reset();

    // Title updates constant buffer memory before part 2 runs, and another command
    // list takes the recycled scratch chunk and overwrites it with different data.
    // SAFETY: exclusive test-owned constant buffer.
    unsafe { std::ptr::write_volatile(&mut cb_mem[0], 0x3333_4444) };
    let list_other = host.new_list().expect("list_other");
    let (_, scratch_ptr) = list_other.scratch(256).expect("other scratch");
    // SAFETY: 256 bytes of mapped upload scratch reserved for `list_other`.
    unsafe {
        std::slice::from_raw_parts_mut(scratch_ptr.cast::<u32>(), 64).fill(0xBAD0_F00D);
    }
    list_other.close();

    // Part 2 executes after its wait clears.
    host.execute(&[&list2]);
    let got2 = read_u32(&readback2);
    let _keep = cb_mem;
    assert_eq!(
        got2, 0x3333_4444,
        "part 2 must read updated constants from its own scratch upload, not recycled/stale part 1 scratch (got {got2:#010x}, slot1={slot1}, slot2={slot2})"
    );
    assert_ne!(slot2, slot1, "part 2 after a cut must allocate its own CBV dynamic slot");
    assert_eq!(
        state.dynamic.len(),
        2,
        "part 1's dynamic slot must stay live alongside part 2's until the state drops"
    );
}

/// Dropping part 1's `HostList` after it completes (for example when the idle
/// list pool is full or disabled) must not leave part 2's replayed CBV table
/// referencing part 1's freed scratch memory.
#[test]
fn cut_list_replayed_cbv_table_survives_dropped_part1() {
    let Some(host) = host() else {
        return;
    };
    let (root, pso) = compile_cbv_copy(&host.device);
    let title_heap = TitleHeap::new(64);
    let h = heap(title_heap.0);
    let host_heap = h.host().expect("host view heap");
    let table_handle = h.handle(0).expect("slot 0 handle");

    let mut cb_mem = Box::new([0u32; 64]);
    cb_mem[0] = 0xAAAA_BBBB;
    let cb_addr = cb_mem.as_ptr() as u64;
    let cb_desc = [cb_addr as u32, (cb_addr >> 32) as u32, 256, 0];
    create_cbv(null(), &cb_desc, table_handle);

    let plan = TablePlan::new(&cbv_used(), &[]);
    let mut state = State::default();
    state.set_heaps(&[title_heap.0]);
    state.set_table(true, 0, table_handle);

    let list1 = host.new_list().expect("list1");
    let made1 = state.tables(&list1, &plan, true);
    let slot1 = made1[0].1 / 32;
    let readback1 = record_cbv_dispatch(host, &list1, &root, &pso, host_heap, slot1);
    list1.close();

    let list2 = host.new_list().expect("list2");
    state.replay(&list2);
    let made2 = state.tables(&list2, &plan, true);
    let slot2 = made2[0].1 / 32;
    let readback2 = record_cbv_dispatch(host, &list2, &root, &pso, host_heap, slot2);
    list2.close();

    host.execute(&[&list1]);
    let got1 = read_u32(&readback1);
    assert_eq!(got1, 0xAAAA_BBBB, "part 1 must read initial CBV value (got {got1:#010x})");
    drop(list1);

    // SAFETY: exclusive test-owned constant buffer.
    unsafe { std::ptr::write_volatile(&mut cb_mem[0], 0x5566_7788) };
    host.execute(&[&list2]);
    let got2 = read_u32(&readback2);
    let _keep = cb_mem;
    assert_eq!(
        got2, 0x5566_7788,
        "part 2 must read updated constants (got {got2:#010x}, slot1={slot1}, slot2={slot2})"
    );
    assert_ne!(slot2, slot1);
}
