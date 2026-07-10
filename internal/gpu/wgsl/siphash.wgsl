// Batched SipHash-2-4 (github.com/dchest/siphash: Hash(k0, k1, data)), one per
// invocation. Keys + byte range come from a meta buffer. Composed after u64.wgsl.
//
// meta layout per message: [k0.lo, k0.hi, k1.lo, k1.hi, off, len]
@group(0) @binding(0) var<storage, read> sip_msg: array<u32>;    // bytes, one per u32
@group(0) @binding(1) var<storage, read> sip_meta: array<u32>;
@group(0) @binding(2) var<storage, read_write> sip_out: array<u32>; // 2 u32/msg
struct SipParams { count: u32 }
@group(0) @binding(3) var<uniform> sip_params: SipParams;

var<private> v0: vec2<u32>;
var<private> v1: vec2<u32>;
var<private> v2: vec2<u32>;
var<private> v3: vec2<u32>;

fn sipround() {
    v0 = u64_add(v0, v1); v1 = u64_rotl(v1, 13u); v1 = u64_xor(v1, v0); v0 = u64_rotl(v0, 32u);
    v2 = u64_add(v2, v3); v3 = u64_rotl(v3, 16u); v3 = u64_xor(v3, v2);
    v0 = u64_add(v0, v3); v3 = u64_rotl(v3, 21u); v3 = u64_xor(v3, v0);
    v2 = u64_add(v2, v1); v1 = u64_rotl(v1, 17u); v1 = u64_xor(v1, v2); v2 = u64_rotl(v2, 32u);
}

// little-endian 8-byte load from the byte-per-u32 message at [off+i .. +8)
fn sip_load(off: u32, i: u32, avail: u32) -> vec2<u32> {
    var lo = 0u; var hi = 0u;
    for (var j: u32 = 0u; j < 4u; j = j + 1u) {
        if (i + j < avail) { lo = lo | ((sip_msg[off + i + j] & 0xffu) << (j * 8u)); }
    }
    for (var j: u32 = 0u; j < 4u; j = j + 1u) {
        if (i + 4u + j < avail) { hi = hi | ((sip_msg[off + i + 4u + j] & 0xffu) << (j * 8u)); }
    }
    return vec2<u32>(lo, hi);
}

@compute @workgroup_size(64)
fn siphash_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= sip_params.count) { return; }
    let m = idx * 6u;
    let k0 = vec2<u32>(sip_meta[m], sip_meta[m + 1u]);
    let k1 = vec2<u32>(sip_meta[m + 2u], sip_meta[m + 3u]);
    let off = sip_meta[m + 4u];
    let len = sip_meta[m + 5u];

    v0 = u64_xor(k0, vec2<u32>(0x70736575u, 0x736f6d65u));
    v1 = u64_xor(k1, vec2<u32>(0x6e646f6du, 0x646f7261u));
    v2 = u64_xor(k0, vec2<u32>(0x6e657261u, 0x6c796765u));
    v3 = u64_xor(k1, vec2<u32>(0x79746573u, 0x74656462u));

    let nblocks = len / 8u;
    for (var b: u32 = 0u; b < nblocks; b = b + 1u) {
        let mi = sip_load(off, b * 8u, len);
        v3 = u64_xor(v3, mi);
        sipround(); sipround();
        v0 = u64_xor(v0, mi);
    }
    // last block: remaining bytes + (len & 0xff) << 56
    var last = sip_load(off, nblocks * 8u, len);
    last.y = last.y | ((len & 0xffu) << 24u);
    v3 = u64_xor(v3, last);
    sipround(); sipround();
    v0 = u64_xor(v0, last);

    v2 = u64_xor(v2, vec2<u32>(0xffu, 0u));
    sipround(); sipround(); sipround(); sipround();
    let h = u64_xor(u64_xor(v0, v1), u64_xor(v2, v3));
    sip_out[idx * 2u] = h.x;
    sip_out[idx * 2u + 1u] = h.y;
}
