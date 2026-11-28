use super::{host, name, retire};
use std::sync::{Mutex, OnceLock};
use windows::Win32::Graphics::Direct3D12::ID3D12Resource;

const SLOT: u64 = 256;
const SLAB: u64 = 64 << 10;

type Slabs = (Vec<(ID3D12Resource, usize, u64)>, Vec<(usize, u64)>);
static SLABS: Mutex<Slabs> = Mutex::new((Vec::new(), Vec::new()));

fn enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("VICTORY_BUDGET_PACK").is_some_and(|v| v == "1"))
}

pub(super) enum Budget {
    Own(ID3D12Resource, usize),
    Packed(Slot),
}

pub(super) struct Slot {
    slab: usize,
    offset: u64,
    data: usize,
    gpu: u64,
}

struct Returned(usize, u64);

impl Drop for Returned {
    fn drop(&mut self) {
        SLABS.lock().unwrap_or_else(|e| e.into_inner()).1.push((self.0, self.1));
    }
}

impl Drop for Slot {
    fn drop(&mut self) {
        retire(Returned(self.slab, self.offset));
    }
}

impl Budget {
    pub(super) fn new(card_budget: bool) -> Option<Self> {
        Self::new_with(enabled(), card_budget)
    }

    fn new_with(pack: bool, card_budget: bool) -> Option<Self> {
        if !pack || card_budget {
            let (buffer, data) = host()?.shared_buffer(SLOT)?;
            return Some(Self::Own(buffer, data as usize));
        }
        let mut slabs = SLABS.lock().unwrap_or_else(|e| e.into_inner());
        if slabs.1.is_empty() {
            let (buffer, data) = host()?.shared_buffer(SLAB)?;
            name(&buffer, || "indirect budgets".into());
            // SAFETY: the buffer is live.
            let gpu = unsafe { buffer.GetGPUVirtualAddress() };
            let index = slabs.0.len();
            slabs.0.push((buffer, data as usize, gpu));
            slabs.1.extend((0..SLAB / SLOT).rev().map(|n| (index, n * SLOT)));
        }
        let (slab, offset) = slabs.1.pop()?;
        let (_, data, gpu) = slabs.0[slab];
        let data = data + offset as usize;
        // SAFETY: this slot is exclusive and was retired before it became free.
        unsafe { std::ptr::write_bytes(data as *mut u8, 0, SLOT as usize) };
        Some(Self::Packed(Slot { slab, offset, data, gpu: gpu + offset }))
    }

    pub(super) fn data(&self) -> usize {
        match self { Self::Own(_, data) => *data, Self::Packed(slot) => slot.data }
    }

    pub(super) fn gpu(&self) -> u64 {
        match self {
            // SAFETY: the buffer remains owned by this budget.
            Self::Own(buffer, _) => unsafe { buffer.GetGPUVirtualAddress() },
            Self::Packed(slot) => slot.gpu,
        }
    }

    pub(super) fn own_buffer(&self) -> Option<&ID3D12Resource> {
        match self { Self::Own(buffer, _) => Some(buffer), Self::Packed(_) => None }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packed_budgets_are_separate_zeroed_slots_and_card_budgets_are_owned() {
        if host().is_none() { return; }
        let first = Budget::new_with(true, false).expect("first budget");
        let second = Budget::new_with(true, false).expect("second budget");
        assert!(matches!(first, Budget::Packed(_)));
        assert!(matches!(second, Budget::Packed(_)));
        assert!(first.gpu().abs_diff(second.gpu()) >= SLOT);
        // SAFETY: both mapped slots are live and exclusively held by this test.
        unsafe {
            assert_eq!(*(first.data() as *const u32), 0);
            assert_eq!(*(second.data() as *const u32), 0);
            *(first.data() as *mut u32) = 42;
            assert_eq!(*(second.data() as *const u32), 0);
        }
        let card = Budget::new_with(true, true).expect("card budget backing");
        assert!(card.own_buffer().is_some());
        let disabled = Budget::new_with(false, false).expect("unpacked budget");
        assert!(disabled.own_buffer().is_some());
    }
}
