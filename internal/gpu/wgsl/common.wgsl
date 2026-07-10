// Shared conventions for all AstroBWTv3 kernels (see internal/gpu/shaders.go
// for the composition order):
//
//   - "bytes" live one per u32 element, always masked to 0..255.
//   - u64 values are vec2<u32> with .x = low word, .y = high word.
//   - step3 below is THE 256-byte wolf-loop state from pow.go; salsa20, rc4,
//     the 64-bit hashes and the op table all operate on it directly.
//   - every file prefixes its private helpers/vars (sha_, salsa_, rc4_,
//     u64_/xxh_/sip_/fnv_, op_) to keep the concatenated module collision-free.

var<private> step3: array<u32, 256>;

fn u8v(x: u32) -> u32 { return x & 0xffu; }

// rotate a byte left by r (r masked to 0..7)
fn rotl8(x: u32, r: u32) -> u32 {
    let k = r & 7u;
    return ((x << k) | (x >> (8u - k))) & 0xffu;
}

// reverse the 8 low bits
fn rev8(x: u32) -> u32 {
    return (reverseBits(x & 0xffu) >> 24u) & 0xffu;
}

// popcount of the 8 low bits
fn pop8(x: u32) -> u32 {
    return countOneBits(x & 0xffu);
}
