// SHA-256 over each little-endian u32 suffix-array row.
// Algorithm adapted from Dirtybird-CUDA-Miner (MIT), Copyright (c) 2026
// Dirtybird99: https://github.com/Dirtybird99/Dirtybird-CUDA-Miner
static const uint ROWS = 2048;
static const uint STRIDE = 70912;

[[vk::binding(0, 0)]] StructuredBuffer<uint> suffixes;
[[vk::binding(1, 0)]] StructuredBuffer<uint> lengths;
[[vk::binding(2, 0)]] StructuredBuffer<uint> roundK;
[[vk::binding(3, 0)]] RWStructuredBuffer<uint> digests;

uint rotr(uint x, uint n) { return (x >> n) | (x << (32 - n)); }

uint bswap(uint x)
{
    return ((x & 0x000000ff) << 24) |
           ((x & 0x0000ff00) << 8) |
           ((x & 0x00ff0000) >> 8) |
           ((x & 0xff000000) >> 24);
}

[shader("compute")]
[numthreads(32, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    uint row = tid.x;
    if (row >= ROWS) return;

    uint words = lengths[row];
    uint blocks = (words * 4 + 8) / 64 + 1;
    uint rowBase = row * STRIDE;
    uint h0 = 0x6a09e667, h1 = 0xbb67ae85, h2 = 0x3c6ef372, h3 = 0xa54ff53a;
    uint h4 = 0x510e527f, h5 = 0x9b05688c, h6 = 0x1f83d9ab, h7 = 0x5be0cd19;

    for (uint block = 0; block < blocks; ++block) {
        uint w[16];
        [unroll]
        for (uint t = 0; t < 16; ++t) {
            uint pos = block * 16 + t;
            uint word = 0;
            if (pos < words) word = bswap(suffixes[rowBase + pos]);
            else if (pos == words) word = 0x80000000;
            else if (block + 1 == blocks && t == 15) word = words * 32;
            w[t] = word;
        }

        uint a = h0, b = h1, c = h2, d = h3;
        uint e = h4, f = h5, g = h6, h = h7;
        [unroll]
        for (uint t = 0; t < 64; ++t) {
            uint slot = t & 15;
            if (t >= 16) {
                uint x = w[(t - 15) & 15];
                uint y = w[(t - 2) & 15];
                uint s0 = rotr(x, 7) ^ rotr(x, 18) ^ (x >> 3);
                uint s1 = rotr(y, 17) ^ rotr(y, 19) ^ (y >> 10);
                w[slot] = w[slot] + s0 + w[(t - 7) & 15] + s1;
            }
            uint s1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
            uint choice = (e & f) ^ ((~e) & g);
            uint temp1 = h + s1 + choice + roundK[t] + w[slot];
            uint s0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
            uint majority = (a & b) ^ (a & c) ^ (b & c);
            uint temp2 = s0 + majority;
            h = g; g = f; f = e; e = d + temp1;
            d = c; c = b; b = a; a = temp1 + temp2;
        }
        h0 += a; h1 += b; h2 += c; h3 += d;
        h4 += e; h5 += f; h6 += g; h7 += h;
    }

    uint out = row * 8;
    digests[out + 0] = bswap(h0); digests[out + 1] = bswap(h1);
    digests[out + 2] = bswap(h2); digests[out + 3] = bswap(h3);
    digests[out + 4] = bswap(h4); digests[out + 5] = bswap(h5);
    digests[out + 6] = bswap(h6); digests[out + 7] = bswap(h7);
}
