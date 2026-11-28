//! Coverage and lighting provenance for an original two-ray scene.
//!
//! The phase fixture complements #472's movable-occluder test and #473's
//! translated node benchmark. It uses a synthetic scene record and compute
//! passes so the same cases can run on D3D12 devices with or without DXR.

#[cfg(windows)]
#[path = "rt_stage_coverage/d3d12.rs"]
mod d3d12;

const DIRECT: u32 = 8;
const SHADOW: u32 = 1;
const RAY_TRACED: u32 = 1;
const AMBIENT_FALLBACK: u32 = 2;

// The rays start at z = 0 and point along +z. They share the occluder's y plane.
const RAY_X: [i32; 2] = [-2, 2];
const RAY_END: i32 = 8;

#[derive(Clone, Copy)]
struct Occluder {
    x: [i32; 2],
    z: [i32; 2],
}

const LEFT: Occluder = Occluder {
    x: [-3, -1],
    z: [2, 3],
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Stage {
    Complete,
    MissingBuild,
    RejectedBinding,
    UnavailableResolve,
}

#[derive(Clone, Copy)]
struct Case {
    name: &'static str,
    occluder: Option<Occluder>,
    stage: Stage,
}

const CASES: [Case; 5] = [
    Case {
        name: "clear",
        occluder: None,
        stage: Stage::Complete,
    },
    Case {
        name: "occluded",
        occluder: Some(LEFT),
        stage: Stage::Complete,
    },
    // Keep this after a successful build with geometry in the same resource.
    Case {
        name: "missing build",
        occluder: Some(LEFT),
        stage: Stage::MissingBuild,
    },
    Case {
        name: "wrong binding",
        occluder: Some(LEFT),
        stage: Stage::RejectedBinding,
    },
    Case {
        name: "unavailable resolve",
        occluder: Some(LEFT),
        stage: Stage::UnavailableResolve,
    },
];

// A readback record shared with rt_stage_coverage/phases.hlsl. Each member is
// one dword or a pair of dwords. Completion counts come from the phase that
// produced them, and each pixel carries its source separately from its value.
#[repr(C)]
#[derive(Debug, PartialEq, Eq)]
struct Report {
    generation: u32,
    build: u32,
    nodes: u32,
    bind: u32,
    resolve: u32,
    hits: [u32; 2],
    output: u32,
    fallback: u32,
    light: [u32; 2],
    source: [u32; 2],
}

fn reference(case: Case, generation: u32) -> Report {
    let build = case.stage != Stage::MissingBuild;
    let bind = build && case.stage != Stage::RejectedBinding;
    let resolve = bind && case.stage != Stage::UnavailableResolve;
    let hits = RAY_X.map(|x| {
        u32::from(
            resolve
                && case.occluder.is_some_and(|o| {
                    (o.x[0]..=o.x[1]).contains(&x) && o.z[0].max(0) <= o.z[1].min(RAY_END)
                }),
        )
    });
    Report {
        generation,
        build: u32::from(build),
        nodes: u32::from(build && case.occluder.is_some()),
        bind: u32::from(bind),
        resolve: u32::from(resolve),
        hits,
        output: 1,
        fallback: u32::from(!resolve),
        light: hits.map(|hit| if hit == 1 { SHADOW } else { DIRECT }),
        source: [if resolve {
            RAY_TRACED
        } else {
            AMBIENT_FALLBACK
        }; 2],
    }
}

fn assert_report(actual: &Report, expected: &Report, case: &str) {
    assert_eq!(
        actual.generation, expected.generation,
        "{case}: frame generation"
    );
    assert_eq!(actual.build, expected.build, "{case}: build count");
    assert_eq!(actual.nodes, expected.nodes, "{case}: node count");
    assert_eq!(actual.bind, expected.bind, "{case}: bound scene count");
    assert_eq!(actual.resolve, expected.resolve, "{case}: resolve count");
    assert_eq!(actual.hits, expected.hits, "{case}: ray hits");
    assert_eq!(actual.output, expected.output, "{case}: output count");
    assert_eq!(actual.light, expected.light, "{case}: output values");
    assert_eq!(actual.fallback, expected.fallback, "{case}: fallback count");
    assert_eq!(
        actual.source, expected.source,
        "{case}: lighting provenance"
    );
}

#[test]
fn cpu_oracle_distinguishes_missing_work_from_a_miss() {
    let hand_checked = [
        // build, nodes, bind, resolve, hits, output, fallback, light, source
        (1, 0, 1, 1, [0, 0], 1, 0, [8, 8], [1, 1]),
        (1, 1, 1, 1, [1, 0], 1, 0, [1, 8], [1, 1]),
        (0, 0, 0, 0, [0, 0], 1, 1, [8, 8], [2, 2]),
        (1, 1, 0, 0, [0, 0], 1, 1, [8, 8], [2, 2]),
        (1, 1, 1, 0, [0, 0], 1, 1, [8, 8], [2, 2]),
    ];
    for (case, (build, nodes, bind, resolve, hits, output, fallback, light, source)) in
        CASES.into_iter().zip(hand_checked)
    {
        let expected = Report {
            generation: 7,
            build,
            nodes,
            bind,
            resolve,
            hits,
            output,
            fallback,
            light,
            source,
        };
        assert_report(&reference(case, 7), &expected, case.name);
    }
}
