// AstroBWTv3 front half, one thread per nonce. Milestone-3a: the PROLOGUE
// (sha256(input) -> salsa20/20 -> rc4 -> fnv1a64) producing step_3 and the
// initial lhash, all chained in one kernel with step_3 in a private array.
// Composed after u64.wgsl. The wolf loop is added on top once this is verified.
//
// step_3 and the RC4 S-box live in private memory (one byte per u32). sha256
// and salsa are inlined (the standalone kernels have their own bindings and
// can't be concatenated).

@group(0) @binding(0) var<storage, read> fh_in: array<u32>;      // input blobs, bytes/u32
@group(0) @binding(1) var<storage, read> fh_meta: array<u32>;    // [off,len] per nonce
@group(0) @binding(2) var<storage, read> fh_shak: array<u32>;    // 64 sha256 round constants
@group(0) @binding(3) var<storage, read> fh_optable: array<u32>; // 256 x [count,op0..op3]
@group(0) @binding(4) var<storage, read_write> fh_out: array<u32>;   // data stream, bytes/u32, out_stride/nonce
@group(0) @binding(5) var<storage, read_write> fh_outlen: array<u32>; // data_len per nonce
struct FhParams { count: u32, out_stride: u32 }
@group(0) @binding(6) var<uniform> fh_p: FhParams;

var<private> fh_step3: array<u32, 256>;   // the wolf state (one byte per u32)
var<private> fh_rc4: array<u32, 256>;     // RC4 S-box
var<private> fh_rc4i: u32;
var<private> fh_rc4j: u32;
var<private> fh_w: array<u32, 64>;        // sha message schedule
var<private> fh_key: array<u32, 8>;       // sha digest / salsa key words

fn fh_rotr(x: u32, n: u32) -> u32 { return (x >> n) | (x << (32u - n)); }

// SHA-256 of fh_in[off..off+len] -> fh_key[0..7] (digest words, big-endian).
fn fh_sha256(off: u32, len: u32) {
    var h0 = 0x6a09e667u; var h1 = 0xbb67ae85u; var h2 = 0x3c6ef372u; var h3 = 0xa54ff53au;
    var h4 = 0x510e527fu; var h5 = 0x9b05688cu; var h6 = 0x1f83d9abu; var h7 = 0x5be0cd19u;
    let bitLen = len * 8u;
    let padded = ((len + 8u) / 64u + 1u) * 64u;
    let nblocks = padded / 64u;
    for (var b: u32 = 0u; b < nblocks; b = b + 1u) {
        for (var t: u32 = 0u; t < 16u; t = t + 1u) {
            var word = 0u;
            for (var j: u32 = 0u; j < 4u; j = j + 1u) {
                let pos = b * 64u + t * 4u + j;
                var byte = 0u;
                if (pos < len) { byte = fh_in[off + pos] & 0xffu; }
                else if (pos == len) { byte = 0x80u; }
                else if (pos == padded - 4u) { byte = (bitLen >> 24u) & 0xffu; }
                else if (pos == padded - 3u) { byte = (bitLen >> 16u) & 0xffu; }
                else if (pos == padded - 2u) { byte = (bitLen >> 8u) & 0xffu; }
                else if (pos == padded - 1u) { byte = bitLen & 0xffu; }
                word = (word << 8u) | byte;
            }
            fh_w[t] = word;
        }
        for (var t: u32 = 16u; t < 64u; t = t + 1u) {
            let s0 = fh_rotr(fh_w[t-15u], 7u) ^ fh_rotr(fh_w[t-15u], 18u) ^ (fh_w[t-15u] >> 3u);
            let s1 = fh_rotr(fh_w[t-2u], 17u) ^ fh_rotr(fh_w[t-2u], 19u) ^ (fh_w[t-2u] >> 10u);
            fh_w[t] = fh_w[t-16u] + s0 + fh_w[t-7u] + s1;
        }
        var a = h0; var b2 = h1; var c = h2; var d = h3;
        var e = h4; var f = h5; var g = h6; var hh = h7;
        for (var t: u32 = 0u; t < 64u; t = t + 1u) {
            let S1 = fh_rotr(e, 6u) ^ fh_rotr(e, 11u) ^ fh_rotr(e, 25u);
            let ch = (e & f) ^ ((~e) & g);
            let t1 = hh + S1 + ch + fh_shak[t] + fh_w[t];
            let S0 = fh_rotr(a, 2u) ^ fh_rotr(a, 13u) ^ fh_rotr(a, 22u);
            let maj = (a & b2) ^ (a & c) ^ (b2 & c);
            let t2 = S0 + maj;
            hh = g; g = f; f = e; e = d + t1; d = c; c = b2; b2 = a; a = t1 + t2;
        }
        h0 = h0 + a; h1 = h1 + b2; h2 = h2 + c; h3 = h3 + d;
        h4 = h4 + e; h5 = h5 + f; h6 = h6 + g; h7 = h7 + hh;
    }
    fh_key[0] = h0; fh_key[1] = h1; fh_key[2] = h2; fh_key[3] = h3;
    fh_key[4] = h4; fh_key[5] = h5; fh_key[6] = h6; fh_key[7] = h7;
}

fn fh_rotl32(x: u32, r: u32) -> u32 { return (x << r) | (x >> (32u - r)); }

// Salsa20/20 keystream (key = byteswap of the 8 sha digest words, counter 0),
// 4 blocks -> fh_step3[0..255].
fn fh_salsa() {
    // salsa key words = the 32 digest bytes read little-endian = byteswap(hi)
    var k: array<u32, 8>;
    for (var i: u32 = 0u; i < 8u; i = i + 1u) {
        let v = fh_key[i];
        k[i] = ((v & 0xffu) << 24u) | ((v & 0xff00u) << 8u) | ((v >> 8u) & 0xff00u) | ((v >> 24u) & 0xffu);
    }
    for (var blk: u32 = 0u; blk < 4u; blk = blk + 1u) {
        let j0 = 0x61707865u; let j1 = k[0]; let j2 = k[1]; let j3 = k[2];
        let j4 = k[3]; let j5 = 0x3320646eu; let j6 = 0u; let j7 = 0u;
        let j8 = blk; let j9 = 0u; let j10 = 0x79622d32u; let j11 = k[4];
        let j12 = k[5]; let j13 = k[6]; let j14 = k[7]; let j15 = 0x6b206574u;
        var x0=j0; var x1=j1; var x2=j2; var x3=j3; var x4=j4; var x5=j5; var x6=j6; var x7=j7;
        var x8=j8; var x9=j9; var x10=j10; var x11=j11; var x12=j12; var x13=j13; var x14=j14; var x15=j15;
        for (var r: u32 = 0u; r < 10u; r = r + 1u) {
            x4=x4^fh_rotl32(x0+x12,7u); x8=x8^fh_rotl32(x4+x0,9u); x12=x12^fh_rotl32(x8+x4,13u); x0=x0^fh_rotl32(x12+x8,18u);
            x9=x9^fh_rotl32(x5+x1,7u); x13=x13^fh_rotl32(x9+x5,9u); x1=x1^fh_rotl32(x13+x9,13u); x5=x5^fh_rotl32(x1+x13,18u);
            x14=x14^fh_rotl32(x10+x6,7u); x2=x2^fh_rotl32(x14+x10,9u); x6=x6^fh_rotl32(x2+x14,13u); x10=x10^fh_rotl32(x6+x2,18u);
            x3=x3^fh_rotl32(x15+x11,7u); x7=x7^fh_rotl32(x3+x15,9u); x11=x11^fh_rotl32(x7+x3,13u); x15=x15^fh_rotl32(x11+x7,18u);
            x1=x1^fh_rotl32(x0+x3,7u); x2=x2^fh_rotl32(x1+x0,9u); x3=x3^fh_rotl32(x2+x1,13u); x0=x0^fh_rotl32(x3+x2,18u);
            x6=x6^fh_rotl32(x5+x4,7u); x7=x7^fh_rotl32(x6+x5,9u); x4=x4^fh_rotl32(x7+x6,13u); x5=x5^fh_rotl32(x4+x7,18u);
            x11=x11^fh_rotl32(x10+x9,7u); x8=x8^fh_rotl32(x11+x10,9u); x9=x9^fh_rotl32(x8+x11,13u); x10=x10^fh_rotl32(x9+x8,18u);
            x12=x12^fh_rotl32(x15+x14,7u); x13=x13^fh_rotl32(x12+x15,9u); x14=x14^fh_rotl32(x13+x12,13u); x15=x15^fh_rotl32(x14+x13,18u);
        }
        fh_storeBlk(blk * 64u, x0+j0, x1+j1, x2+j2, x3+j3, x4+j4, x5+j5, x6+j6, x7+j7,
                    x8+j8, x9+j9, x10+j10, x11+j11, x12+j12, x13+j13, x14+j14, x15+j15);
    }
}

fn fh_put4(base: u32, v: u32) {
    fh_step3[base + 0u] = v & 0xffu;
    fh_step3[base + 1u] = (v >> 8u) & 0xffu;
    fh_step3[base + 2u] = (v >> 16u) & 0xffu;
    fh_step3[base + 3u] = (v >> 24u) & 0xffu;
}
fn fh_storeBlk(o: u32, w0:u32,w1:u32,w2:u32,w3:u32,w4:u32,w5:u32,w6:u32,w7:u32,w8:u32,w9:u32,w10:u32,w11:u32,w12:u32,w13:u32,w14:u32,w15:u32) {
    fh_put4(o+0u,w0); fh_put4(o+4u,w1); fh_put4(o+8u,w2); fh_put4(o+12u,w3);
    fh_put4(o+16u,w4); fh_put4(o+20u,w5); fh_put4(o+24u,w6); fh_put4(o+28u,w7);
    fh_put4(o+32u,w8); fh_put4(o+36u,w9); fh_put4(o+40u,w10); fh_put4(o+44u,w11);
    fh_put4(o+48u,w12); fh_put4(o+52u,w13); fh_put4(o+56u,w14); fh_put4(o+60u,w15);
}

// RC4 keyed by the current fh_step3 (all 256 bytes); resets i,j to 0.
fn fh_rc4_ksa() {
    for (var i: u32 = 0u; i < 256u; i = i + 1u) { fh_rc4[i] = i; }
    var j: u32 = 0u;
    for (var i: u32 = 0u; i < 256u; i = i + 1u) {
        j = (j + fh_rc4[i] + (fh_step3[i] & 0xffu)) & 0xffu;
        let tmp = fh_rc4[i]; fh_rc4[i] = fh_rc4[j]; fh_rc4[j] = tmp;
    }
    fh_rc4i = 0u; fh_rc4j = 0u;
}
// RC4 PRGA XOR over the first 256 bytes of fh_step3, advancing i,j.
fn fh_rc4_xor256() {
    for (var n: u32 = 0u; n < 256u; n = n + 1u) {
        fh_rc4i = (fh_rc4i + 1u) & 0xffu;
        fh_rc4j = (fh_rc4j + fh_rc4[fh_rc4i]) & 0xffu;
        let tmp = fh_rc4[fh_rc4i]; fh_rc4[fh_rc4i] = fh_rc4[fh_rc4j]; fh_rc4[fh_rc4j] = tmp;
        let ks = fh_rc4[(fh_rc4[fh_rc4i] + fh_rc4[fh_rc4j]) & 0xffu];
        fh_step3[n] = (fh_step3[n] ^ ks) & 0xffu;
    }
}

// fnv1a64 over fh_step3[0..len]
fn fh_fnv(len: u32) -> vec2<u32> {
    var h = vec2<u32>(0x84222325u, 0xcbf29ce4u);
    let prime = vec2<u32>(0x000001b3u, 0x00000100u);
    for (var i: u32 = 0u; i < len; i = i + 1u) {
        h = u64_xor(h, vec2<u32>(fh_step3[i] & 0xffu, 0u));
        h = u64_mul(h, prime);
    }
    return h;
}

// --- private-memory xxhash64 / siphash over fh_step3[0..len] ---
fn xp1() -> vec2<u32> { return vec2<u32>(0x85ebca87u, 0x9e3779b1u); }
fn xp2() -> vec2<u32> { return vec2<u32>(0x27d4eb4fu, 0xc2b2ae3du); }
fn xp3() -> vec2<u32> { return vec2<u32>(0x9e3779f9u, 0x165667b1u); }
fn xp4() -> vec2<u32> { return vec2<u32>(0xc2b2ae63u, 0x85ebca77u); }
fn xp5() -> vec2<u32> { return vec2<u32>(0x165667c5u, 0x27d4eb2fu); }

fn fh_ld64(i: u32) -> vec2<u32> {
    let lo = (fh_step3[i]&0xffu) | ((fh_step3[i+1u]&0xffu)<<8u) | ((fh_step3[i+2u]&0xffu)<<16u) | ((fh_step3[i+3u]&0xffu)<<24u);
    let hi = (fh_step3[i+4u]&0xffu) | ((fh_step3[i+5u]&0xffu)<<8u) | ((fh_step3[i+6u]&0xffu)<<16u) | ((fh_step3[i+7u]&0xffu)<<24u);
    return vec2<u32>(lo, hi);
}
fn fh_xround(acc: vec2<u32>, inp: vec2<u32>) -> vec2<u32> {
    var a = u64_add(acc, u64_mul(inp, xp2())); a = u64_rotl(a, 31u); return u64_mul(a, xp1());
}
fn fh_xmerge(acc: vec2<u32>, val: vec2<u32>) -> vec2<u32> {
    let v = fh_xround(vec2<u32>(0u,0u), val); return u64_add(u64_mul(u64_xor(acc, v), xp1()), xp4());
}
fn fh_xx(len: u32) -> vec2<u32> {
    var h: vec2<u32>; var i: u32 = 0u;
    if (len >= 32u) {
        var v1 = u64_add(xp1(), xp2()); var v2 = xp2(); var v3 = vec2<u32>(0u,0u);
        var v4 = u64_add(vec2<u32>(~xp1().x, ~xp1().y), vec2<u32>(1u,0u));
        let limit = len - 32u;
        loop {
            v1 = fh_xround(v1, fh_ld64(i)); i=i+8u;
            v2 = fh_xround(v2, fh_ld64(i)); i=i+8u;
            v3 = fh_xround(v3, fh_ld64(i)); i=i+8u;
            v4 = fh_xround(v4, fh_ld64(i)); i=i+8u;
            if (i > limit) { break; }
        }
        h = u64_add(u64_add(u64_rotl(v1,1u), u64_rotl(v2,7u)), u64_add(u64_rotl(v3,12u), u64_rotl(v4,18u)));
        h = fh_xmerge(h, v1); h = fh_xmerge(h, v2); h = fh_xmerge(h, v3); h = fh_xmerge(h, v4);
    } else { h = xp5(); }
    h = u64_add(h, vec2<u32>(len, 0u));
    for (; i + 8u <= len; i = i + 8u) {
        let k = fh_xround(vec2<u32>(0u,0u), fh_ld64(i));
        h = u64_xor(h, k); h = u64_add(u64_mul(u64_rotl(h,27u), xp1()), xp4());
    }
    if (i + 4u <= len) {
        let w = (fh_step3[i]&0xffu)|((fh_step3[i+1u]&0xffu)<<8u)|((fh_step3[i+2u]&0xffu)<<16u)|((fh_step3[i+3u]&0xffu)<<24u);
        h = u64_xor(h, u64_mul(vec2<u32>(w,0u), xp1()));
        h = u64_add(u64_mul(u64_rotl(h,23u), xp2()), xp3()); i=i+4u;
    }
    for (; i < len; i = i + 1u) {
        h = u64_xor(h, u64_mul(vec2<u32>(fh_step3[i]&0xffu,0u), xp5()));
        h = u64_mul(u64_rotl(h,11u), xp1());
    }
    h = u64_xor(h, u64_shr(h,33u)); h = u64_mul(h, xp2());
    h = u64_xor(h, u64_shr(h,29u)); h = u64_mul(h, xp3());
    h = u64_xor(h, u64_shr(h,32u));
    return h;
}

var<private> sv0: vec2<u32>; var<private> sv1: vec2<u32>;
var<private> sv2: vec2<u32>; var<private> sv3: vec2<u32>;
fn fh_sipround() {
    sv0=u64_add(sv0,sv1); sv1=u64_rotl(sv1,13u); sv1=u64_xor(sv1,sv0); sv0=u64_rotl(sv0,32u);
    sv2=u64_add(sv2,sv3); sv3=u64_rotl(sv3,16u); sv3=u64_xor(sv3,sv2);
    sv0=u64_add(sv0,sv3); sv3=u64_rotl(sv3,21u); sv3=u64_xor(sv3,sv0);
    sv2=u64_add(sv2,sv1); sv1=u64_rotl(sv1,17u); sv1=u64_xor(sv1,sv2); sv2=u64_rotl(sv2,32u);
}
fn fh_sipload(i: u32, avail: u32) -> vec2<u32> {
    var lo=0u; var hi=0u;
    for (var j: u32 = 0u; j < 4u; j = j+1u) { if (i+j<avail) { lo=lo|((fh_step3[i+j]&0xffu)<<(j*8u)); } }
    for (var j: u32 = 0u; j < 4u; j = j+1u) { if (i+4u+j<avail) { hi=hi|((fh_step3[i+4u+j]&0xffu)<<(j*8u)); } }
    return vec2<u32>(lo, hi);
}
fn fh_sip(k0: vec2<u32>, k1: vec2<u32>, len: u32) -> vec2<u32> {
    sv0=u64_xor(k0, vec2<u32>(0x70736575u,0x736f6d65u));
    sv1=u64_xor(k1, vec2<u32>(0x6e646f6du,0x646f7261u));
    sv2=u64_xor(k0, vec2<u32>(0x6e657261u,0x6c796765u));
    sv3=u64_xor(k1, vec2<u32>(0x79746573u,0x74656462u));
    let nb = len / 8u;
    for (var b: u32 = 0u; b < nb; b = b+1u) {
        let mi = fh_sipload(b*8u, len);
        sv3=u64_xor(sv3,mi); fh_sipround(); fh_sipround(); sv0=u64_xor(sv0,mi);
    }
    var last = fh_sipload(nb*8u, len);
    last.y = last.y | ((len & 0xffu) << 24u);
    sv3=u64_xor(sv3,last); fh_sipround(); fh_sipround(); sv0=u64_xor(sv0,last);
    sv2=u64_xor(sv2, vec2<u32>(0xffu,0u));
    fh_sipround(); fh_sipround(); fh_sipround(); fh_sipround();
    return u64_xor(u64_xor(sv0,sv1), u64_xor(sv2,sv3));
}

fn rotl8(x: u32, r: u32) -> u32 { let k = r & 7u; return ((x << k) | (x >> (8u - k))) & 0xffu; }
fn rev8(x: u32) -> u32 { return (reverseBits(x & 0xffu) >> 24u) & 0xffu; }

// apply case c's micro-op sequence to byte x (p = current step_3[pos2]).
fn fh_wolfApply(c: u32, x0: u32, p: u32) -> u32 {
    var x = x0 & 0xffu;
    let cnt = fh_optable[c * 5u];
    for (var k: u32 = 0u; k < cnt; k = k + 1u) {
        let code = fh_optable[c * 5u + 1u + k];
        let op = code >> 8u;
        let param = code & 0xffu;
        switch (op) {
            case 1u: { x = rotl8(x, param); }
            case 2u: { x = x ^ rotl8(x, param); }
            case 3u: { x = (x << (x & 3u)) & 0xffu; }
            case 4u: { x = x >> (x & 3u); }
            case 5u: { x = x ^ p; }
            case 6u: { x = x & p; }
            case 7u: { x = (~x) & 0xffu; }
            case 8u: { x = x ^ countOneBits(x & 0xffu); }
            case 9u: { x = rev8(x); }
            case 10u: { x = rotl8(x, x); }
            case 11u: { x = (x + x) & 0xffu; }
            case 12u: { x = (x * x) & 0xffu; }
            case 13u: { x = (x - (x ^ 97u)) & 0xffu; }
            default: {}
        }
    }
    return x & 0xffu;
}

@compute @workgroup_size(64)
fn fronthalf_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= fh_p.count) { return; }
    let off = fh_meta[idx * 2u];
    let len = fh_meta[idx * 2u + 1u];

    fh_sha256(off, len);
    fh_salsa();
    fh_rc4_ksa();
    fh_rc4_xor256();
    var lhash = fh_fnv(256u);
    var prevLhash = lhash;

    let outBase = idx * fh_p.out_stride;
    var tries: u32 = 0u;
    loop {
        tries = tries + 1u;
        let rs = u64_xor(u64_xor(prevLhash, lhash), vec2<u32>(tries, 0u));
        let op = rs.x & 0xffu;
        var pos1 = (rs.x >> 8u) & 0xffu;
        var pos2 = (rs.x >> 16u) & 0xffu;
        if (pos1 > pos2) { let t = pos1; pos1 = pos2; pos2 = t; }
        if (pos2 - pos1 > 32u) { pos2 = pos1 + ((pos2 - pos1) & 0x1fu); }

        if (op == 254u || op == 255u) { fh_rc4_ksa(); }
        for (var i: u32 = pos1; i < pos2; i = i + 1u) {
            fh_step3[i] = fh_wolfApply(op, fh_step3[i], fh_step3[pos2]);
            if (op == 0u) {
                let a = rev8(fh_step3[pos1]); let b = rev8(fh_step3[pos2]);
                fh_step3[pos2] = a; fh_step3[pos1] = b;
            }
            if (op == 253u) {
                prevLhash = u64_add(lhash, prevLhash);
                lhash = fh_xx(pos2);
            }
        }

        let diff = (fh_step3[pos1] - fh_step3[pos2]) & 0xffu;
        if (diff < 0x10u) { prevLhash = u64_add(lhash, prevLhash); lhash = fh_xx(pos2); }
        if (diff < 0x20u) { prevLhash = u64_add(lhash, prevLhash); lhash = fh_fnv(pos2); }
        if (diff < 0x30u) { prevLhash = u64_add(lhash, prevLhash); lhash = fh_sip(vec2<u32>(tries,0u), prevLhash, pos2); }
        if (diff <= 0x40u) { fh_rc4_xor256(); }

        fh_step3[255] = (fh_step3[255] ^ fh_step3[pos1] ^ fh_step3[pos2]) & 0xffu;
        // write the 256-byte chunk PACKED (4 bytes/u32, little-endian within the
        // word) — the exact layout the SA kernel's text input expects, so the
        // front-half output feeds the SA directly on-GPU with no readback.
        let wbase = outBase + (tries - 1u) * 64u;
        for (var w: u32 = 0u; w < 64u; w = w + 1u) {
            let b = w * 4u;
            fh_out[wbase + w] = (fh_step3[b] & 0xffu) | ((fh_step3[b+1u] & 0xffu) << 8u)
                | ((fh_step3[b+2u] & 0xffu) << 16u) | ((fh_step3[b+3u] & 0xffu) << 24u);
        }

        if (tries > 276u || (fh_step3[255] >= 0xf0u && tries > 260u)) { break; }
    }
    fh_outlen[idx] = (tries - 4u) * 256u + (((fh_step3[253] << 8u) | fh_step3[254]) & 0x3ffu);
}
