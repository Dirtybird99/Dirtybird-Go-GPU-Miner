// Batched SHA-256: one invocation hashes one message. Messages are stored as
// bytes (one per u32) in a flat buffer; per-message offset+length come from a
// meta buffer. Output is 32 bytes/message, one byte per u32, big-endian digest
// order (matching crypto/sha256's Sum output).
//
// Message length here is bounded by SHA_MAX_BYTES; the AstroBWTv3 front-half
// input is a ~76-byte work blob, well within one or two blocks. This kernel is
// the reusable SHA-256 core; the final-hash stage feeds it the SA bytes.

const SHA_MAX_BYTES: u32 = 256u;      // max message length this kernel accepts
const SHA_MAX_BLOCKS: u32 = 5u;       // ceil((256+9)/64) = 5 blocks

// SHA_K is uploaded as a storage buffer (binding 0). naga in this gogpu/wgpu
// build miscompiles a dynamically-indexed module-scope `const` array (it reads
// as garbage), so the round constants come from a storage buffer, which is
// always safe to index dynamically. (Dynamic indexing of a var<private> array
// is fine — see rc4.wgsl and the sha_w schedule below; the hazard is `const`.)
@group(0) @binding(0) var<storage, read> sha_k: array<u32>;     // 64 round constants
@group(0) @binding(1) var<storage, read> sha_msg: array<u32>;   // bytes, one per u32
@group(0) @binding(2) var<storage, read> sha_meta: array<u32>;  // [off0,len0, off1,len1, ...]
@group(0) @binding(3) var<storage, read_write> sha_out: array<u32>; // 32 bytes/msg
struct ShaParams { count: u32 }
@group(0) @binding(4) var<uniform> sha_params: ShaParams;

fn sha_rotr(x: u32, n: u32) -> u32 { return (x >> n) | (x << (32u - n)); }

var<private> sha_w: array<u32, 64>;

// Digest of sha_msg[off .. off+len]; writes 32 bytes to sha_out at outByte.
fn sha256_bytes(off: u32, len: u32, outByte: u32) {
    var h0 = 0x6a09e667u; var h1 = 0xbb67ae85u; var h2 = 0x3c6ef372u; var h3 = 0xa54ff53au;
    var h4 = 0x510e527fu; var h5 = 0x9b05688cu; var h6 = 0x1f83d9abu; var h7 = 0x5be0cd19u;

    let bitLen = len * 8u;
    // total padded length in bytes: len + 1 (0x80) + k zeros + 8 (length), to a multiple of 64
    let padded = ((len + 8u) / 64u + 1u) * 64u;
    let nblocks = padded / 64u;

    for (var b: u32 = 0u; b < nblocks; b = b + 1u) {
        // build the 16 message words for this block from the byte view (with padding)
        for (var t: u32 = 0u; t < 16u; t = t + 1u) {
            var word = 0u;
            for (var j: u32 = 0u; j < 4u; j = j + 1u) {
                let pos = b * 64u + t * 4u + j;
                var byte = 0u;
                if (pos < len) {
                    byte = sha_msg[off + pos] & 0xffu;
                } else if (pos == len) {
                    byte = 0x80u;
                } else if (pos == padded - 4u) {
                    byte = (bitLen >> 24u) & 0xffu;
                } else if (pos == padded - 3u) {
                    byte = (bitLen >> 16u) & 0xffu;
                } else if (pos == padded - 2u) {
                    byte = (bitLen >> 8u) & 0xffu;
                } else if (pos == padded - 1u) {
                    byte = bitLen & 0xffu;
                }
                word = (word << 8u) | byte;
            }
            sha_w[t] = word;
        }
        for (var t: u32 = 16u; t < 64u; t = t + 1u) {
            let s0 = sha_rotr(sha_w[t-15u], 7u) ^ sha_rotr(sha_w[t-15u], 18u) ^ (sha_w[t-15u] >> 3u);
            let s1 = sha_rotr(sha_w[t-2u], 17u) ^ sha_rotr(sha_w[t-2u], 19u) ^ (sha_w[t-2u] >> 10u);
            sha_w[t] = sha_w[t-16u] + s0 + sha_w[t-7u] + s1;
        }

        var a = h0; var b2 = h1; var c = h2; var d = h3;
        var e = h4; var f = h5; var g = h6; var h = h7;
        for (var t: u32 = 0u; t < 64u; t = t + 1u) {
            let S1 = sha_rotr(e, 6u) ^ sha_rotr(e, 11u) ^ sha_rotr(e, 25u);
            let ch = (e & f) ^ ((~e) & g);
            let t1 = h + S1 + ch + sha_k[t] + sha_w[t];
            let S0 = sha_rotr(a, 2u) ^ sha_rotr(a, 13u) ^ sha_rotr(a, 22u);
            let maj = (a & b2) ^ (a & c) ^ (b2 & c);
            let t2 = S0 + maj;
            h = g; g = f; f = e; e = d + t1;
            d = c; c = b2; b2 = a; a = t1 + t2;
        }
        h0 = h0 + a; h1 = h1 + b2; h2 = h2 + c; h3 = h3 + d;
        h4 = h4 + e; h5 = h5 + f; h6 = h6 + g; h7 = h7 + h;
    }

    sha_store(outByte + 0u, h0);
    sha_store(outByte + 4u, h1);
    sha_store(outByte + 8u, h2);
    sha_store(outByte + 12u, h3);
    sha_store(outByte + 16u, h4);
    sha_store(outByte + 20u, h5);
    sha_store(outByte + 24u, h6);
    sha_store(outByte + 28u, h7);
}

fn sha_store(base: u32, v: u32) {
    sha_out[base + 0u] = (v >> 24u) & 0xffu;
    sha_out[base + 1u] = (v >> 16u) & 0xffu;
    sha_out[base + 2u] = (v >> 8u) & 0xffu;
    sha_out[base + 3u] = v & 0xffu;
}

@compute @workgroup_size(64)
fn sha256_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= sha_params.count) { return; }
    let off = sha_meta[i * 2u];
    let len = sha_meta[i * 2u + 1u];
    sha256_bytes(off, len, i * 32u);
}
