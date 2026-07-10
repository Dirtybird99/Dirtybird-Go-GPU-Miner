// Direct LCP refinement for equal three-byte prefix runs.
// Algorithm adapted from Dirtybird-CUDA-Miner (MIT), Copyright (c) 2026
// Dirtybird99: https://github.com/Dirtybird99/Dirtybird-CUDA-Miner
static const uint ROWS = 512;
static const uint STRIDE = 70912;
static const uint TEXT_WORDS = STRIDE / 4;
static const uint DEPTH = 3;
static const uint SENTINEL = 0xffffffff;
static const uint BIN_COUNT = 7;
static const uint DIRECT_ARGS = 16;

#ifndef MAXLEN
#define MAXLEN 32
#endif
#ifndef BIN
#define BIN 0
#endif

[[vk::binding(0, 0)]] RWStructuredBuffer<uint64_t> text;
[[vk::binding(1, 0)]] RWStructuredBuffer<uint> lengths;
[[vk::binding(2, 0)]] RWStructuredBuffer<uint> coarseKeys;
[[vk::binding(3, 0)]] RWStructuredBuffer<uint> suffixes;
[[vk::binding(4, 0)]] RWStructuredBuffer<uint2> descriptors;
[[vk::binding(5, 0)]] RWStructuredBuffer<uint> ids;
[[vk::binding(6, 0)]] RWStructuredBuffer<uint> control;

groupshared uint largeValues[MAXLEN];

uint load32(uint row, uint position)
{
    uint qword = row * (TEXT_WORDS / 2) + (position >> 3);
    uint shift = (position & 7) * 8;
    uint64_t lo = text[qword];
    if (shift == 0) return uint(lo);
    uint64_t value = lo >> shift;
    if (shift > 32) value |= text[qword + 1] << (64 - shift);
    return uint(value);
}

uint64_t load64(uint row, uint position)
{
    uint qword = row * (TEXT_WORDS / 2) + (position >> 3);
    uint shift = (position & 7) * 8;
    uint64_t lo = text[qword];
    if (shift == 0) return lo;
    return (lo >> shift) | (text[qword + 1] << (64 - shift));
}

int compareSuffix(uint row, uint n, uint a, uint b, uint from, out uint lcp)
{
    if (a == b) {
        lcp = n - a;
        return 0;
    }
    uint aa = a + from;
    uint bb = b + from;
    while (aa + 8 <= n && bb + 8 <= n) {
        uint64_t av = load64(row, aa);
        uint64_t bv = load64(row, bb);
        uint64_t difference = av ^ bv;
        if (difference != 0) {
            uint low = uint(difference);
            uint byteIndex = low != 0 ? uint(firstbitlow(low)) >> 3
                                      : 4 + (uint(firstbitlow(uint(difference >> 32))) >> 3);
            uint ca = uint(av >> (byteIndex * 8)) & 255;
            uint cb = uint(bv >> (byteIndex * 8)) & 255;
            lcp = aa + byteIndex - a;
            return ca < cb ? -1 : 1;
        }
        aa += 8;
        bb += 8;
    }
    while (aa + 4 <= n && bb + 4 <= n) {
        uint av = load32(row, aa);
        uint bv = load32(row, bb);
        uint difference = av ^ bv;
        if (difference != 0) {
            uint byteIndex = uint(firstbitlow(difference)) >> 3;
            uint ca = (av >> (byteIndex * 8)) & 255;
            uint cb = (bv >> (byteIndex * 8)) & 255;
            lcp = aa + byteIndex - a;
            return ca < cb ? -1 : 1;
        }
        aa += 4;
        bb += 4;
    }
    while (aa < n && bb < n) {
        uint ca = uint(text[row * (TEXT_WORDS / 2) + (aa >> 3)] >> ((aa & 7) * 8)) & 255;
        uint cb = uint(text[row * (TEXT_WORDS / 2) + (bb >> 3)] >> ((bb & 7) * 8)) & 255;
        if (ca != cb) {
            lcp = aa - a;
            return ca < cb ? -1 : 1;
        }
        aa++;
        bb++;
    }
    lcp = aa - a;
    if (aa >= n && bb >= n) {
        uint remainingA = n - a;
        uint remainingB = n - b;
        return remainingA < remainingB ? -1 : 1;
    }
    return aa >= n ? -1 : 1;
}

bool lessSuffix(uint row, uint n, uint a, uint b)
{
    if (a == SENTINEL) return false;
    if (b == SENTINEL) return true;
    uint ignored;
    return compareSuffix(row, n, a, b, DEPTH, ignored) < 0;
}

uint binOf(uint length)
{
    if (length == 2) return 0;
    if (length <= 8) return 1;
    if (length <= 16) return 2;
    if (length <= 32) return 3;
    if (length <= 128) return 4;
    if (length <= 512) return 5;
    return 6;
}

uint binCapacity(uint bin)
{
    uint minimums[7] = { 2, 3, 9, 17, 33, 129, 513 };
    return (ROWS * STRIDE) / minimums[bin] + 1;
}

uint binOffset(uint bin)
{
    uint offset = 0;
    [unroll]
    for (uint i = 0; i < 7; i++)
        if (i < bin) offset += binCapacity(i);
    return offset;
}

uint ballotBefore(uint4 ballot)
{
    uint lane = WaveGetLaneIndex();
    uint before = 0;
    [unroll]
    for (uint part = 0; part < 4; part++) {
        uint base = part * 32;
        if (lane >= base + 32)
            before += countbits(ballot[part]);
        else if (lane > base)
            before += countbits(ballot[part] & ((1u << (lane - base)) - 1));
    }
    return before;
}

uint ballotCount(uint4 ballot)
{
    return countbits(ballot.x) + countbits(ballot.y) + countbits(ballot.z) + countbits(ballot.w);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void DetectDirect(uint3 groupID : SV_GroupID, uint tid : SV_GroupThreadID)
{
    uint row = groupID.y;
    uint slot = groupID.x * 256 + tid;
    uint n = lengths[row];
    uint offset = row * STRIDE;
    bool emit = false;
    uint runLength = 0;
    uint myBin = 0;
    if (slot < n) {
        uint key = coarseKeys[offset + slot];
        bool start = slot == 0 || coarseKeys[offset + slot - 1] != key;
        if (start) {
            uint end = slot + 1;
            while (end < n && coarseKeys[offset + end] == key)
                end++;
            runLength = end - slot;
            emit = runLength > 1;
            if (runLength > 2048) {
                InterlockedOr(control[63], 1);
                emit = false;
            }
            if (emit) myBin = binOf(runLength);
        }
    }
    [unroll]
    for (uint bin = 0; bin < BIN_COUNT; bin++) {
        bool mine = emit && myBin == bin;
        uint4 ballot = WaveActiveBallot(mine);
        uint count = ballotCount(ballot);
        uint base = 0;
        if (WaveIsFirstLane() && count != 0)
            InterlockedAdd(control[bin], count, base);
        base = WaveReadLaneFirst(base);
        if (mine)
            descriptors[binOffset(bin) + base + ballotBefore(ballot)] = uint2(offset + slot, runLength);
    }
}

[shader("compute")]
[numthreads(1, 1, 1)]
void PrepareDirect()
{
    [unroll]
    for (uint bin = 0; bin < BIN_COUNT; bin++) {
        uint count = control[bin];
        uint arg = DIRECT_ARGS + bin * 3;
        control[arg] = bin < 4 ? (count + 127) / 128 : count;
        control[arg + 1] = 1;
        control[arg + 2] = 1;
    }
}

[shader("compute")]
[numthreads(128, 1, 1)]
void SortTiny(uint tid : SV_DispatchThreadID)
{
    uint count = control[BIN];
    if (tid >= count)
        return;
    uint2 descriptor = descriptors[binOffset(BIN) + tid];
    uint start = descriptor.x;
    uint length = descriptor.y;
    uint row = start / STRIDE;
    uint n = lengths[row];

    uint values[MAXLEN];
    uint lcps[MAXLEN];
    [unroll]
    for (uint i = 0; i < MAXLEN; i++) {
        if (i < length) values[i] = suffixes[start + i];
        lcps[i] = 0;
    }
    for (uint i = 1; i < length; i++) {
        uint key = values[i];
        uint h;
        int comparison = compareSuffix(row, n, key, values[i - 1], DEPTH, h);
        if (comparison > 0) {
            lcps[i] = h;
            continue;
        }
        int j = int(i) - 1;
        uint rightLCP = h;
        values[j + 1] = values[j];
        for (;;) {
            uint adjacent = lcps[j];
            if (j == 0)
                break;
            if (adjacent > h) {
                rightLCP = h;
                values[j] = values[j - 1];
                lcps[j + 1] = lcps[j];
                j--;
                continue;
            }
            if (adjacent < h) {
                rightLCP = adjacent;
                break;
            }
            uint nextLCP;
            int next = compareSuffix(row, n, key, values[j - 1], h, nextLCP);
            rightLCP = nextLCP;
            if (next > 0)
                break;
            h = nextLCP;
            values[j] = values[j - 1];
            lcps[j + 1] = lcps[j];
            j--;
        }
        values[j] = key;
        if (j > 0) lcps[j] = rightLCP;
        lcps[j + 1] = h;
    }
    [unroll]
    for (uint i = 0; i < MAXLEN; i++)
        if (i < length) suffixes[start + i] = values[i];
}

[shader("compute")]
[numthreads(128, 1, 1)]
void SortTinyGlobal(uint tid : SV_DispatchThreadID)
{
    uint count = control[BIN];
    if (tid >= count)
        return;
    uint2 descriptor = descriptors[binOffset(BIN) + tid];
    uint start = descriptor.x;
    uint length = descriptor.y;
    uint row = start / STRIDE;
    uint n = lengths[row];
    ids[start] = 0;
    for (uint i = 1; i < length; i++) {
        uint key = suffixes[start + i];
        uint h;
        int comparison = compareSuffix(row, n, key, suffixes[start + i - 1], DEPTH, h);
        if (comparison > 0) {
            ids[start + i] = h;
            continue;
        }
        int j = int(i) - 1;
        uint rightLCP = h;
        suffixes[start + j + 1] = suffixes[start + j];
        for (;;) {
            uint adjacent = ids[start + j];
            if (j == 0)
                break;
            if (adjacent > h) {
                rightLCP = h;
                suffixes[start + j] = suffixes[start + j - 1];
                ids[start + j + 1] = adjacent;
                j--;
                continue;
            }
            if (adjacent < h) {
                rightLCP = adjacent;
                break;
            }
            uint nextLCP;
            int next = compareSuffix(row, n, key, suffixes[start + j - 1], h, nextLCP);
            rightLCP = nextLCP;
            if (next > 0)
                break;
            h = nextLCP;
            suffixes[start + j] = suffixes[start + j - 1];
            ids[start + j + 1] = adjacent;
            j--;
        }
        suffixes[start + j] = key;
        if (j > 0) ids[start + j] = rightLCP;
        ids[start + j + 1] = h;
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void SortLarge(uint3 groupID : SV_GroupID, uint tid : SV_GroupThreadID)
{
    uint descriptorIndex = groupID.x;
    uint count = control[BIN];
    if (descriptorIndex >= count)
        return;
    uint2 descriptor = descriptors[binOffset(BIN) + descriptorIndex];
    uint start = descriptor.x;
    uint length = descriptor.y;
    uint row = start / STRIDE;
    uint n = lengths[row];
    uint size = 1;
    while (size < length) size <<= 1;
    for (uint i = tid; i < size; i += 256)
        largeValues[i] = i < length ? suffixes[start + i] : SENTINEL;
    GroupMemoryBarrierWithGroupSync();

    for (uint span = 2; span <= size; span <<= 1) {
        for (uint stride = span >> 1; stride != 0; stride >>= 1) {
            for (uint i = tid; i < size; i += 256) {
                uint partner = i ^ stride;
                if (partner > i) {
                    uint a = largeValues[i];
                    uint b = largeValues[partner];
                    bool ascending = (i & span) == 0;
                    bool swap = ascending ? lessSuffix(row, n, b, a) : lessSuffix(row, n, a, b);
                    if (swap) {
                        largeValues[i] = b;
                        largeValues[partner] = a;
                    }
                }
            }
            GroupMemoryBarrierWithGroupSync();
        }
    }
    for (uint i = tid; i < length; i += 256)
        suffixes[start + i] = largeValues[i];
}
