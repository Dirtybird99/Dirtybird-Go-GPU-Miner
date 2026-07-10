// RC4 matching astrobwt's prologue: NewCipher(step_3) then
// XORKeyStream(step_3, step_3) in place. Input = the 256-byte post-salsa
// state (also the key); output = the 256-byte post-rc4 state. One invocation
// per hash. The 256-entry state is var<private> (dynamic indexing of a
// var<private> array is compiled correctly by naga; a const array is not).

@group(0) @binding(0) var<storage, read> rc4_in: array<u32>;    // 256 bytes/hash, one per u32
@group(0) @binding(1) var<storage, read_write> rc4_out: array<u32>; // 256 bytes/hash
struct RC4Params { count: u32 }
@group(0) @binding(2) var<uniform> rc4_params: RC4Params;

var<private> rc4_s: array<u32, 256>;

@compute @workgroup_size(64)
fn rc4_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= rc4_params.count) { return; }
    let base = idx * 256u;

    // KSA: key is the 256-byte input, so key[i] == rc4_in[base + i].
    for (var i: u32 = 0u; i < 256u; i = i + 1u) {
        rc4_s[i] = i;
    }
    var j: u32 = 0u;
    for (var i: u32 = 0u; i < 256u; i = i + 1u) {
        j = (j + rc4_s[i] + (rc4_in[base + i] & 0xffu)) & 0xffu;
        let tmp = rc4_s[i];
        rc4_s[i] = rc4_s[j];
        rc4_s[j] = tmp;
    }

    // PRGA over the same 256 bytes, XOR in place.
    var pi: u32 = 0u;
    var pj: u32 = 0u;
    for (var n: u32 = 0u; n < 256u; n = n + 1u) {
        pi = (pi + 1u) & 0xffu;
        pj = (pj + rc4_s[pi]) & 0xffu;
        let tmp = rc4_s[pi];
        rc4_s[pi] = rc4_s[pj];
        rc4_s[pj] = tmp;
        let k = rc4_s[(rc4_s[pi] + rc4_s[pj]) & 0xffu];
        rc4_out[base + n] = (rc4_in[base + n] ^ k) & 0xffu;
    }
}
