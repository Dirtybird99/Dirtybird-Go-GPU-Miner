// Batched XXH64 (seed 0), matching github.com/cespare/xxhash Sum64, one per
// invocation. Byte range from a meta buffer [off,len]. Composed after u64.wgsl.

@group(0) @binding(0) var<storage, read> xx_msg: array<u32>;
@group(0) @binding(1) var<storage, read> xx_meta: array<u32>;    // [off0,len0, ...]
@group(0) @binding(2) var<storage, read_write> xx_out: array<u32>; // 2 u32/msg
struct XxParams { count: u32 }
@group(0) @binding(3) var<uniform> xx_params: XxParams;

// naga miscompiles module-scope const composites (reads as zero); use functions.
fn XXP1() -> vec2<u32> { return vec2<u32>(0x85ebca87u, 0x9e3779b1u); }
fn XXP2() -> vec2<u32> { return vec2<u32>(0x27d4eb4fu, 0xc2b2ae3du); }
fn XXP3() -> vec2<u32> { return vec2<u32>(0x9e3779f9u, 0x165667b1u); }
fn XXP4() -> vec2<u32> { return vec2<u32>(0xc2b2ae63u, 0x85ebca77u); }
fn XXP5() -> vec2<u32> { return vec2<u32>(0x165667c5u, 0x27d4eb2fu); }

fn xx_load64(off: u32, i: u32) -> vec2<u32> {
    let b = off + i;
    let lo = (xx_msg[b] & 0xffu) | ((xx_msg[b+1u] & 0xffu) << 8u) | ((xx_msg[b+2u] & 0xffu) << 16u) | ((xx_msg[b+3u] & 0xffu) << 24u);
    let hi = (xx_msg[b+4u] & 0xffu) | ((xx_msg[b+5u] & 0xffu) << 8u) | ((xx_msg[b+6u] & 0xffu) << 16u) | ((xx_msg[b+7u] & 0xffu) << 24u);
    return vec2<u32>(lo, hi);
}
fn xx_load32(off: u32, i: u32) -> u32 {
    let b = off + i;
    return (xx_msg[b] & 0xffu) | ((xx_msg[b+1u] & 0xffu) << 8u) | ((xx_msg[b+2u] & 0xffu) << 16u) | ((xx_msg[b+3u] & 0xffu) << 24u);
}

fn xx_round(acc: vec2<u32>, inp: vec2<u32>) -> vec2<u32> {
    var a = u64_add(acc, u64_mul(inp, XXP2()));
    a = u64_rotl(a, 31u);
    return u64_mul(a, XXP1());
}
fn xx_merge(acc: vec2<u32>, val: vec2<u32>) -> vec2<u32> {
    let v = xx_round(vec2<u32>(0u, 0u), val);
    return u64_add(u64_mul(u64_xor(acc, v), XXP1()), XXP4());
}

fn xxhash64(off: u32, len: u32) -> vec2<u32> {
    var h: vec2<u32>;
    var i: u32 = 0u;
    if (len >= 32u) {
        var v1 = u64_add(XXP1(), XXP2());
        var v2 = XXP2();
        var v3 = vec2<u32>(0u, 0u);
        var v4 = u64_add(vec2<u32>(~XXP1().x, ~XXP1().y), vec2<u32>(1u, 0u)); // -P1
        let limit = len - 32u;
        loop {
            v1 = xx_round(v1, xx_load64(off, i)); i = i + 8u;
            v2 = xx_round(v2, xx_load64(off, i)); i = i + 8u;
            v3 = xx_round(v3, xx_load64(off, i)); i = i + 8u;
            v4 = xx_round(v4, xx_load64(off, i)); i = i + 8u;
            if (i > limit) { break; }
        }
        h = u64_add(u64_add(u64_rotl(v1, 1u), u64_rotl(v2, 7u)), u64_add(u64_rotl(v3, 12u), u64_rotl(v4, 18u)));
        h = xx_merge(h, v1);
        h = xx_merge(h, v2);
        h = xx_merge(h, v3);
        h = xx_merge(h, v4);
    } else {
        h = XXP5();
    }
    h = u64_add(h, vec2<u32>(len, 0u));

    // remaining 8-byte blocks
    for (; i + 8u <= len; i = i + 8u) {
        let k = xx_round(vec2<u32>(0u, 0u), xx_load64(off, i));
        h = u64_xor(h, k);
        h = u64_add(u64_mul(u64_rotl(h, 27u), XXP1()), XXP4());
    }
    // remaining 4-byte block
    if (i + 4u <= len) {
        h = u64_xor(h, u64_mul(vec2<u32>(xx_load32(off, i), 0u), XXP1()));
        h = u64_add(u64_mul(u64_rotl(h, 23u), XXP2()), XXP3());
        i = i + 4u;
    }
    // remaining bytes
    for (; i < len; i = i + 1u) {
        h = u64_xor(h, u64_mul(vec2<u32>(xx_msg[off + i] & 0xffu, 0u), XXP5()));
        h = u64_mul(u64_rotl(h, 11u), XXP1());
    }
    // avalanche
    h = u64_xor(h, u64_shr(h, 33u));
    h = u64_mul(h, XXP2());
    h = u64_xor(h, u64_shr(h, 29u));
    h = u64_mul(h, XXP3());
    h = u64_xor(h, u64_shr(h, 32u));
    return h;
}

@compute @workgroup_size(64)
fn xxhash64_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= xx_params.count) { return; }
    let h = xxhash64(xx_meta[idx * 2u], xx_meta[idx * 2u + 1u]);
    xx_out[idx * 2u] = h.x;
    xx_out[idx * 2u + 1u] = h.y;
}
