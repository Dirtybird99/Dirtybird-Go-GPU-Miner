// 64-bit unsigned integer emulation for WGSL (which has only u32). A u64 is a
// vec2<u32> with .x = low 32 bits, .y = high 32 bits. Used by the AstroBWTv3
// front-half's threshold hashes (fnv1a64, xxhash64, siphash-2-4).

fn u64_add(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    let lo = a.x + b.x;
    let carry = select(0u, 1u, lo < a.x);
    return vec2<u32>(lo, a.y + b.y + carry);
}

fn u64_xor(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    return vec2<u32>(a.x ^ b.x, a.y ^ b.y);
}

// 32x32 -> 64 unsigned multiply via 16-bit limbs.
fn mul32(a: u32, b: u32) -> vec2<u32> {
    let aL = a & 0xffffu; let aH = a >> 16u;
    let bL = b & 0xffffu; let bH = b >> 16u;
    let ll = aL * bL;
    let lh = aL * bH;
    let hl = aH * bL;
    let hh = aH * bH;
    let mid = (ll >> 16u) + (lh & 0xffffu) + (hl & 0xffffu);
    let lo = (ll & 0xffffu) | (mid << 16u);
    let hi = hh + (lh >> 16u) + (hl >> 16u) + (mid >> 16u);
    return vec2<u32>(lo, hi);
}

// 64x64 -> 64 multiply (result truncated to 64 bits).
fn u64_mul(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    let ll = mul32(a.x, b.x);
    let hi = ll.y + a.x * b.y + a.y * b.x;
    return vec2<u32>(ll.x, hi);
}

// rotate-left a 64-bit value by n (0 < n < 64).
fn u64_rotl(a: vec2<u32>, n: u32) -> vec2<u32> {
    if (n == 0u) { return a; }
    if (n < 32u) {
        let lo = (a.x << n) | (a.y >> (32u - n));
        let hi = (a.y << n) | (a.x >> (32u - n));
        return vec2<u32>(lo, hi);
    }
    if (n == 32u) { return vec2<u32>(a.y, a.x); }
    let m = n - 32u;
    let lo = (a.y << m) | (a.x >> (32u - m));
    let hi = (a.x << m) | (a.y >> (32u - m));
    return vec2<u32>(lo, hi);
}

fn u64_shl(a: vec2<u32>, n: u32) -> vec2<u32> {
    if (n == 0u) { return a; }
    if (n < 32u) { return vec2<u32>(a.x << n, (a.y << n) | (a.x >> (32u - n))); }
    return vec2<u32>(0u, a.x << (n - 32u));
}

fn u64_shr(a: vec2<u32>, n: u32) -> vec2<u32> {
    if (n == 0u) { return a; }
    if (n < 32u) { return vec2<u32>((a.x >> n) | (a.y << (32u - n)), a.y >> n); }
    return vec2<u32>(a.y >> (n - 32u), 0u);
}
