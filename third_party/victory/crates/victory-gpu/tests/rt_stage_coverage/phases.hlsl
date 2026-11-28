// Original fixtures. The built node and frame report persist across command lists.
#define ROOT "RootConstants(num32BitConstants=10, b0), UAV(u0), UAV(u1)"
cbuffer Inputs : register(b0) {
    uint generation;
    uint node_count;
    int2 node_x;
    int2 node_z;
    int2 ray_x;
    int ray_end;
    uint unused;
};

struct Scene {
    uint generation;
    uint nodes;
    int2 x;
    int2 z;
};

struct Report {
    uint generation;
    uint build;
    uint nodes;
    uint bind;
    uint resolve;
    uint2 hits;
    uint output;
    uint fallback;
    uint2 light;
    uint2 source;
};

RWStructuredBuffer<Scene> scene : register(u0);
RWStructuredBuffer<Report> report : register(u1);

[RootSignature(ROOT)]
[numthreads(1, 1, 1)]
void reset_report() {
    // Only the spare scene is bound for this pass. The previous build stays live.
    scene[0] = (Scene)0;
    Report empty = (Report)0;
    empty.generation = generation;
    report[0] = empty;
}

[RootSignature(ROOT)]
[numthreads(1, 1, 1)]
void build_scene() {
    Scene built = { generation, node_count, node_x, node_z };
    scene[0] = built;
    report[0].build += 1;
    report[0].nodes = built.nodes;
}

[RootSignature(ROOT)]
[numthreads(1, 1, 1)]
void bind_scene() {
    // This reads the actual bound resource, including on a reused or wrong slot.
    if (scene[0].generation == generation) {
        report[0].bind += 1;
    }
}

[RootSignature(ROOT)]
[numthreads(1, 1, 1)]
void resolve_visibility() {
    if (report[0].bind == 0) {
        return;
    }
    Scene node = scene[0];
    if (node.nodes == 1 && node.z.y >= 0 && node.z.x <= ray_end) {
        report[0].hits.x = (node.x.x <= ray_x.x && ray_x.x <= node.x.y) ? 1 : 0;
        report[0].hits.y = (node.x.x <= ray_x.y && ray_x.y <= node.x.y) ? 1 : 0;
    }
    report[0].resolve += 1;
}

[RootSignature(ROOT)]
[numthreads(1, 1, 1)]
void publish_lighting() {
    if (report[0].resolve != 0) {
        const uint2 hits = report[0].hits;
        report[0].light = uint2(hits.x == 1 ? 1 : 8, hits.y == 1 ? 1 : 8);
        report[0].source = uint2(1, 1);
    } else {
        // An ambient approximation can have the same value as a ray-traced miss.
        report[0].light = uint2(8, 8);
        report[0].source = uint2(2, 2);
        report[0].fallback += 1;
    }
    report[0].output += 1;
}
