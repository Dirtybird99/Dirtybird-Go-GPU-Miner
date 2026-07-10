// Batched FNV-1a 64-bit hash over byte messages (one per invocation), matching
// github.com/segmentio/fasthash/fnv1a. Bytes are stored one per u32; per-message
// offset+len come from a meta buffer. Output is 2 u32 (low, high) per message.
// Composed after u64.wgsl.

@group(0) @binding(0) var<storage, read> fnv_msg: array<u32>;
@group(0) @binding(1) var<storage, read> fnv_meta: array<u32>;   // [off0,len0, ...]
@group(0) @binding(2) var<storage, read_write> fnv_out: array<u32>; // 2 u32/msg
struct FnvParams { count: u32 }
@group(0) @binding(3) var<uniform> fnv_params: FnvParams;

fn fnv1a64(off: u32, len: u32) -> vec2<u32> {
    var h = vec2<u32>(0x84222325u, 0xcbf29ce4u); // offset basis 0xcbf29ce484222325
    let prime = vec2<u32>(0x000001b3u, 0x00000100u); // 0x100000001b3
    for (var i: u32 = 0u; i < len; i = i + 1u) {
        h = u64_xor(h, vec2<u32>(fnv_msg[off + i] & 0xffu, 0u));
        h = u64_mul(h, prime);
    }
    return h;
}

@compute @workgroup_size(64)
fn fnv1a64_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= fnv_params.count) { return; }
    let h = fnv1a64(fnv_meta[i * 2u], fnv_meta[i * 2u + 1u]);
    fnv_out[i * 2u] = h.x;
    fnv_out[i * 2u + 1u] = h.y;
}
