// Salsa20/20 keystream, matching golang.org/x/crypto/salsa20/salsa
// XORKeyStream(out, zeros[256], counter=0, key). One invocation produces the
// 256-byte (4-block) keystream for one 32-byte key. This is AstroBWTv3's
// step_3 right after the salsa stage.
//
// Layout of the 4x4 Salsa state (little-endian words):
//   0: "expa"        1..4: key[0:16]      5: "nd 3"
//   6..7: nonce (0)  8..9: block counter  10: "2-by" 11..14: key[16:32] 15: "te k"
// The 16-byte counter is all zero here, so words 6,7 = 0 and words 8,9 carry
// the per-block counter (little-endian, incremented once per 64-byte block).

@group(0) @binding(0) var<storage, read> salsa_keys: array<u32>;   // 8 u32 key words / invocation
@group(0) @binding(1) var<storage, read_write> salsa_out: array<u32>; // 256 bytes / invocation, one byte per u32
struct SalsaParams { count: u32 }
@group(0) @binding(2) var<uniform> salsa_params: SalsaParams;

fn salsa_rotl(x: u32, r: u32) -> u32 { return (x << r) | (x >> (32u - r)); }

// one 64-byte Salsa20/20 block for key words k0..k7 and 64-bit block counter,
// writing 64 bytes to salsa_out starting at outByte.
fn salsa_block(k0: u32, k1: u32, k2: u32, k3: u32, k4: u32, k5: u32, k6: u32, k7: u32,
               ctrLo: u32, ctrHi: u32, outByte: u32) {
    let j0 = 0x61707865u; let j1 = k0; let j2 = k1; let j3 = k2;
    let j4 = k3;          let j5 = 0x3320646eu; let j6 = 0u; let j7 = 0u;
    let j8 = ctrLo;       let j9 = ctrHi; let j10 = 0x79622d32u; let j11 = k4;
    let j12 = k5;         let j13 = k6; let j14 = k7; let j15 = 0x6b206574u;

    var x0 = j0; var x1 = j1; var x2 = j2; var x3 = j3;
    var x4 = j4; var x5 = j5; var x6 = j6; var x7 = j7;
    var x8 = j8; var x9 = j9; var x10 = j10; var x11 = j11;
    var x12 = j12; var x13 = j13; var x14 = j14; var x15 = j15;

    for (var i: u32 = 0u; i < 10u; i = i + 1u) {
        // column rounds
        x4  = x4  ^ salsa_rotl(x0  + x12, 7u);
        x8  = x8  ^ salsa_rotl(x4  + x0,  9u);
        x12 = x12 ^ salsa_rotl(x8  + x4,  13u);
        x0  = x0  ^ salsa_rotl(x12 + x8,  18u);
        x9  = x9  ^ salsa_rotl(x5  + x1,  7u);
        x13 = x13 ^ salsa_rotl(x9  + x5,  9u);
        x1  = x1  ^ salsa_rotl(x13 + x9,  13u);
        x5  = x5  ^ salsa_rotl(x1  + x13, 18u);
        x14 = x14 ^ salsa_rotl(x10 + x6,  7u);
        x2  = x2  ^ salsa_rotl(x14 + x10, 9u);
        x6  = x6  ^ salsa_rotl(x2  + x14, 13u);
        x10 = x10 ^ salsa_rotl(x6  + x2,  18u);
        x3  = x3  ^ salsa_rotl(x15 + x11, 7u);
        x7  = x7  ^ salsa_rotl(x3  + x15, 9u);
        x11 = x11 ^ salsa_rotl(x7  + x3,  13u);
        x15 = x15 ^ salsa_rotl(x11 + x7,  18u);
        // row rounds
        x1  = x1  ^ salsa_rotl(x0  + x3,  7u);
        x2  = x2  ^ salsa_rotl(x1  + x0,  9u);
        x3  = x3  ^ salsa_rotl(x2  + x1,  13u);
        x0  = x0  ^ salsa_rotl(x3  + x2,  18u);
        x6  = x6  ^ salsa_rotl(x5  + x4,  7u);
        x7  = x7  ^ salsa_rotl(x6  + x5,  9u);
        x4  = x4  ^ salsa_rotl(x7  + x6,  13u);
        x5  = x5  ^ salsa_rotl(x4  + x7,  18u);
        x11 = x11 ^ salsa_rotl(x10 + x9,  7u);
        x8  = x8  ^ salsa_rotl(x11 + x10, 9u);
        x9  = x9  ^ salsa_rotl(x8  + x11, 13u);
        x10 = x10 ^ salsa_rotl(x9  + x8,  18u);
        x12 = x12 ^ salsa_rotl(x15 + x14, 7u);
        x13 = x13 ^ salsa_rotl(x12 + x15, 9u);
        x14 = x14 ^ salsa_rotl(x13 + x12, 13u);
        x15 = x15 ^ salsa_rotl(x14 + x13, 18u);
    }

    salsa_store4(outByte +  0u, x0  + j0);
    salsa_store4(outByte +  4u, x1  + j1);
    salsa_store4(outByte +  8u, x2  + j2);
    salsa_store4(outByte + 12u, x3  + j3);
    salsa_store4(outByte + 16u, x4  + j4);
    salsa_store4(outByte + 20u, x5  + j5);
    salsa_store4(outByte + 24u, x6  + j6);
    salsa_store4(outByte + 28u, x7  + j7);
    salsa_store4(outByte + 32u, x8  + j8);
    salsa_store4(outByte + 36u, x9  + j9);
    salsa_store4(outByte + 40u, x10 + j10);
    salsa_store4(outByte + 44u, x11 + j11);
    salsa_store4(outByte + 48u, x12 + j12);
    salsa_store4(outByte + 52u, x13 + j13);
    salsa_store4(outByte + 56u, x14 + j14);
    salsa_store4(outByte + 60u, x15 + j15);
}

// store a Salsa word little-endian as 4 bytes (one per u32 out element)
fn salsa_store4(base: u32, v: u32) {
    salsa_out[base + 0u] = v & 0xffu;
    salsa_out[base + 1u] = (v >> 8u) & 0xffu;
    salsa_out[base + 2u] = (v >> 16u) & 0xffu;
    salsa_out[base + 3u] = (v >> 24u) & 0xffu;
}

@compute @workgroup_size(64)
fn salsa20_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= salsa_params.count) { return; }
    let kb = i * 8u;
    let k0 = salsa_keys[kb + 0u]; let k1 = salsa_keys[kb + 1u];
    let k2 = salsa_keys[kb + 2u]; let k3 = salsa_keys[kb + 3u];
    let k4 = salsa_keys[kb + 4u]; let k5 = salsa_keys[kb + 5u];
    let k6 = salsa_keys[kb + 6u]; let k7 = salsa_keys[kb + 7u];
    let outBase = i * 256u;
    for (var b: u32 = 0u; b < 4u; b = b + 1u) {
        salsa_block(k0, k1, k2, k3, k4, k5, k6, k7, b, 0u, outBase + b * 64u);
    }
}
