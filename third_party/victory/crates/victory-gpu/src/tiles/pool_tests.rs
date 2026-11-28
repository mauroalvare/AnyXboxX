use std::ptr::null_mut;
use std::time::{Duration, Instant};

use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_SAMPLE_DESC};

use super::tests::{placed, sub};
use super::*;
use crate::host::{Host, transition};

const POINTS: usize = 3;
const RGBA_BYTES: usize = 4;
const SAMPLE_STRIDE: u64 = D3D12_TEXTURE_DATA_PLACEMENT_ALIGNMENT as u64;
const SAMPLE_BYTES: u64 = 2 * POINTS as u64 * SAMPLE_STRIDE;
const HEAP_BYTES: u64 = HEAP_TILES as u64 * TILE;

struct Reservation {
    index: usize,
    runs: Vec<(u32, u32)>,
}

impl Reservation {
    fn new(device: &ID3D12Device, tiles: u32) -> Self {
        let (_, index, runs) = take(device, tiles, true).unwrap();
        Self { index, runs }
    }
}

impl Drop for Reservation {
    fn drop(&mut self) {
        give_back(self.index, &self.runs);
    }
}

struct Texture {
    resource: ID3D12Resource,
    tiled: Tiled,
    points: [(u32, u32); POINTS],
}

impl Texture {
    fn new(device: &ID3D12Device, width: u32, height: u32) -> Self {
        const LEVELS: u16 = 2;
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            Width: width.into(),
            Height: height,
            DepthOrArraySize: 1,
            MipLevels: LEVELS,
            Format: DXGI_FORMAT_R8G8B8A8_UNORM,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_64KB_UNDEFINED_SWIZZLE,
            ..Default::default()
        };
        let mut resource = None;
        // SAFETY: a plain reserved 2D texture with one layer and no clear value.
        unsafe {
            device
                .CreateReservedResource(&desc, D3D12_RESOURCE_STATE_COPY_DEST, None, &mut resource)
                .unwrap();
        }
        let resource = resource.unwrap();
        let tiled = Tiled::make(device, &resource, u32::from(LEVELS)).unwrap();
        Self {
            resource,
            tiled,
            points: [(0, 0), (width / 2, height / 2), (width - 1, height - 1)],
        }
    }

    fn map_first_mip(&self, tiles: u32) {
        let (changed, waiting) = self.tiled.uploaded(&self.resource, 1, true);
        assert!(changed && !waiting);
        assert_eq!(self.tiled.given_bytes(), u64::from(tiles) * TILE);
    }
}

struct Samples {
    upload: ID3D12Resource,
    upload_data: *mut u8,
    readback: ID3D12Resource,
}

impl Samples {
    fn new(host: &Host) -> Self {
        let (upload, upload_data) = host.upload_buffer(SAMPLE_BYTES).unwrap();
        let readback = host.read_back_buffer(SAMPLE_BYTES).unwrap();
        Self {
            upload,
            upload_data,
            readback,
        }
    }

    fn round_trip(&self, host: &Host, textures: &[Texture; 2], round: u8) {
        let list = host.new_list().unwrap();
        list.keep(&self.upload);
        list.keep(&self.readback);
        for (i, texture) in textures.iter().enumerate() {
            list.keep(&texture.resource);
            let to = sub(&texture.resource, 0);
            for (p, &(x, y)) in texture.points.iter().enumerate() {
                let footprint = sample_footprint(i, p);
                let rgba = sample(round, i, p);
                // SAFETY: one texel in a 512-byte slot of a mapped upload buffer.
                unsafe {
                    self.upload_data
                        .add(footprint.Offset as usize)
                        .copy_from_nonoverlapping(rgba.as_ptr(), rgba.len())
                };
                let from = placed(&self.upload, footprint);
                // SAFETY: the texture has tiles and both locations cover one texel.
                unsafe { list.list.CopyTextureRegion(&to, x, y, 0, &from, None) };
            }
        }
        // Write both textures before reading either, so aliasing between
        // their heap ranges would corrupt the markers in the first one.
        for (i, texture) in textures.iter().enumerate() {
            let from = sub(&texture.resource, 0);
            // SAFETY: the open list has recorded all writes to this texture.
            unsafe {
                list.list.ResourceBarrier(&[transition(
                    &texture.resource,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                    D3D12_RESOURCE_STATE_COPY_SOURCE,
                )]);
            }
            for (p, &(x, y)) in texture.points.iter().enumerate() {
                let to = placed(&self.readback, sample_footprint(i, p));
                let region = D3D12_BOX {
                    left: x,
                    top: y,
                    front: 0,
                    right: x + 1,
                    bottom: y + 1,
                    back: 1,
                };
                // SAFETY: the live source and destination are in their copy states.
                unsafe {
                    list.list
                        .CopyTextureRegion(&to, 0, 0, 0, &from, Some(&region))
                };
            }
        }
        list.close();
        host.execute(&[&list]);
        let mut data = null_mut();
        // SAFETY: the list finished writing the readback buffer.
        unsafe { self.readback.Map(0, None, Some(&mut data)).unwrap() };
        assert!(!data.is_null());
        for i in 0..textures.len() {
            for p in 0..POINTS {
                let at = sample_footprint(i, p).Offset as usize;
                // SAFETY: each copy wrote the four bytes at this offset.
                let rgba = unsafe { data.cast::<[u8; RGBA_BYTES]>().byte_add(at).read() };
                assert_eq!(
                    rgba,
                    sample(round, i, p),
                    "round {round}, texture {i}, sample {p}"
                );
            }
        }
        // SAFETY: all reads from the mapping are done.
        unsafe { self.readback.Unmap(0, None) };
    }
}

fn sample_footprint(image: usize, point: usize) -> D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
        Offset: (image * POINTS + point) as u64 * SAMPLE_STRIDE,
        Footprint: D3D12_SUBRESOURCE_FOOTPRINT {
            Format: DXGI_FORMAT_R8G8B8A8_UNORM,
            Width: 1,
            Height: 1,
            Depth: 1,
            RowPitch: D3D12_TEXTURE_DATA_PITCH_ALIGNMENT,
        },
    }
}

fn sample(round: u8, image: usize, point: usize) -> [u8; RGBA_BYTES] {
    [round, image as u8 * 127, point as u8 * 59, 255]
}

#[test]
#[ignore = "uses the process-wide tile pool; run by name in a fresh process"]
fn mixed_mip_sizes_reuse_two_heaps() {
    let Some(host) = host() else {
        eprintln!("Skipped the tile-pool workload because no D3D12 device was available");
        return;
    };
    let mut options = D3D12_FEATURE_DATA_D3D12_OPTIONS::default();
    // SAFETY: the device fills the options structure of the specified size.
    unsafe {
        host.device
            .CheckFeatureSupport(
                D3D12_FEATURE_D3D12_OPTIONS,
                (&raw mut options).cast(),
                size_of_val(&options) as u32,
            )
            .unwrap();
    }
    if options.TiledResourcesTier.0 < D3D12_TILED_RESOURCES_TIER_2.0 {
        eprintln!(
            "Skipped the tile-pool workload because the adapter lacks tier 2 tiled resources"
        );
        return;
    }
    assert_eq!(
        STATS[3].load(Ordering::Relaxed),
        0,
        "start the fixture with an empty tile pool"
    );

    const ROUNDS: u8 = 8;
    const SMALL: u32 = HEAP_TILES / 4;
    const LARGE: u32 = 3 * SMALL;
    // Heap 0 ends up with holes of 512 and 256 tiles. Heap 1 has one
    // 256-tile hole. Allocating SMALL in heap 0 would leave nowhere for LARGE.
    let _left = Reservation::new(&host.device, SMALL / 2);
    let middle = Reservation::new(&host.device, 2 * SMALL);
    let _right = Reservation::new(&host.device, SMALL / 2);
    let _other = Reservation::new(&host.device, LARGE);
    drop(middle);
    assert_eq!(STATS[3].load(Ordering::Relaxed), 2 * HEAP_BYTES);

    let samples = Samples::new(host);
    let returned_at_start = STATS[8].load(Ordering::Relaxed);
    let mut peak = 0;
    let mut mapping = Duration::ZERO;
    for round in 0..ROUNDS {
        let textures = [
            Texture::new(&host.device, 2048, 2048),
            Texture::new(&host.device, 4096, 3072),
        ];
        let started = Instant::now();
        textures[0].map_first_mip(SMALL);
        textures[1].map_first_mip(LARGE);
        mapping += started.elapsed();
        peak = peak.max(STATS[3].load(Ordering::Relaxed));
        samples.round_trip(host, &textures, round);
    }
    let returned = STATS[8].load(Ordering::Relaxed) - returned_at_start;
    let heaps_created = STATS[3].load(Ordering::Relaxed) / HEAP_BYTES + returned;
    eprintln!(
        "adapter={:?}, rounds={ROUNDS}, peak_heap_bytes={peak}, heaps_created={heaps_created}, heaps_returned={returned}, map_cpu_us={}",
        crate::host::adapter_summary(),
        mapping.as_micros()
    );
    assert_eq!(peak, 2 * HEAP_BYTES, "a large mip required a third heap");
    assert_eq!(heaps_created, 2, "streaming recreated a heap");
}
