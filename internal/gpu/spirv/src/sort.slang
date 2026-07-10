// Fixed-size, stable radix sort for the 512-row AstroBWT GPU slab.
static const uint ROWS = 512;
static const uint ROW_LEN = 70912;
static const uint TILE_SIZE = 3840;
static const uint TILES_PER_ROW = 19;
static const uint THREADS = 256;
static const uint ITEMS_PER_THREAD = 15;
static const uint RADIX = 256;

[[vk::binding(0, 0)]] RWStructuredBuffer<uint> inputKeys;
[[vk::binding(1, 0)]] RWStructuredBuffer<uint> inputValues;
[[vk::binding(2, 0)]] RWStructuredBuffer<uint> outputKeys;
[[vk::binding(3, 0)]] RWStructuredBuffer<uint> outputValues;
[[vk::binding(4, 0)]] RWStructuredBuffer<uint> tileHist;
[[vk::binding(5, 0)]] RWStructuredBuffer<uint> tileOffsets;
[[vk::binding(6, 0)]] RWStructuredBuffer<uint64_t> text;
[[vk::binding(7, 0)]] RWStructuredBuffer<uint> lengths;

groupshared uint gData[4096];

uint flattenGroup(uint3 gid)
{
    return gid.x + gid.y * 65535;
}

uint loadByte(uint row, uint position)
{
    uint64_t word = text[row * (ROW_LEN / 8) + (position >> 3)];
    return uint(word >> ((position & 7) * 8)) & 255;
}

[shader("compute")]
[numthreads(THREADS, 1, 1)]
void init(uint3 gid : SV_GroupID, uint tid : SV_GroupThreadID)
{
    uint index = flattenGroup(gid) * THREADS + tid;
    if (index < ROWS * ROW_LEN)
    {
        uint row = index / ROW_LEN;
        uint position = index % ROW_LEN;
        uint n = lengths[row];
        if (position < n) {
            uint first = loadByte(row, position) + 1;
            uint second = position + 1 < n ? loadByte(row, position + 1) + 1 : 0;
            uint third = position + 2 < n ? loadByte(row, position + 2) + 1 : 0;
            inputKeys[index] = (first << 18) | (second << 9) | third;
            inputValues[index] = position;
        } else {
            inputKeys[index] = 0xffffffff;
            inputValues[index] = 0;
        }
    }
}

[shader("compute")]
[numthreads(THREADS, 1, 1)]
void histogram(uint3 gid : SV_GroupID, uint tid : SV_GroupThreadID)
{
    gData[tid] = 0;
    GroupMemoryBarrierWithGroupSync();

    uint tile = gid.x;
    uint row = tile / TILES_PER_ROW;
    uint tileInRow = tile % TILES_PER_ROW;
    uint begin = tileInRow * TILE_SIZE;
    uint validCount = min(TILE_SIZE, ROW_LEN - begin);
    uint rowBase = row * ROW_LEN + begin;
    for (uint i = tid; i < validCount; i += THREADS)
    {
        uint digit = (inputKeys[rowBase + i] >> RADIX_SHIFT) & 255;
        InterlockedAdd(gData[digit], 1);
    }
    GroupMemoryBarrierWithGroupSync();
    tileHist[tile * RADIX + tid] = gData[tid];
}

[shader("compute")]
[numthreads(THREADS, 1, 1)]
void scan(uint3 gid : SV_GroupID, uint digit : SV_GroupThreadID)
{
    uint row = gid.x;
    uint running = 0;
    for (uint tile = 0; tile < TILES_PER_ROW; ++tile)
    {
        uint index = ((row * TILES_PER_ROW + tile) * RADIX) + digit;
        uint count = tileHist[index];
        tileOffsets[index] = running;
        running += count;
    }

    gData[digit] = running;
    GroupMemoryBarrierWithGroupSync();

    uint base = 0;
    for (uint d = 0; d < digit; ++d)
        base += gData[d];

    for (uint tile = 0; tile < TILES_PER_ROW; ++tile)
    {
        uint index = ((row * TILES_PER_ROW + tile) * RADIX) + digit;
        tileOffsets[index] += base;
    }
}

uint firstPeer(uint4 peers)
{
    [unroll]
    for (uint part = 0; part < 4; ++part)
    {
        uint bit = firstbitlow(peers[part]);
        if (bit != 0xffffffff)
            return part * 32 + bit;
    }
    return 0;
}

void peerCounts(uint4 peers, out uint before, out uint total)
{
    uint lane = WaveGetLaneIndex();
    before = 0;
    total = 0;
    [unroll]
    for (uint part = 0; part < 4; ++part)
    {
        uint bits = peers[part];
        total += countbits(bits);
        uint partBase = part * 32;
        if (lane >= partBase + 32)
            before += countbits(bits);
        else if (lane > partBase)
            before += countbits(bits & ((1u << (lane - partBase)) - 1));
    }
}

[shader("compute")]
[numthreads(THREADS, 1, 1)]
void scatter(uint3 gid : SV_GroupID, uint tid : SV_GroupThreadID)
{
    uint waveWidth = WaveGetLaneCount();
    uint lane = WaveGetLaneIndex();
    uint wave = tid / waveWidth;
    uint waveCount = THREADS / waveWidth;
    uint waveHistSize = waveCount * RADIX;
    for (uint i = tid; i < waveHistSize; i += THREADS)
        gData[i] = 0;
    GroupMemoryBarrierWithGroupSync();

    uint tile = gid.x;
    uint row = tile / TILES_PER_ROW;
    uint tileInRow = tile % TILES_PER_ROW;
    uint begin = tileInRow * TILE_SIZE;
    uint validCount = min(TILE_SIZE, ROW_LEN - begin);
    uint rowBegin = row * ROW_LEN;
    uint tileBegin = rowBegin + begin;
    uint waveBegin = wave * ITEMS_PER_THREAD * waveWidth;

    uint keys[ITEMS_PER_THREAD];
    uint ranks[ITEMS_PER_THREAD];
    bool valid[ITEMS_PER_THREAD];

    [unroll]
    for (uint item = 0; item < ITEMS_PER_THREAD; ++item)
    {
        uint local = waveBegin + item * waveWidth + lane;
        valid[item] = local < validCount;
        keys[item] = valid[item] ? inputKeys[tileBegin + local] : 0xffffffff;

        uint4 peers = WaveActiveBallot(valid[item]);
        [unroll]
        for (uint bitIndex = 0; bitIndex < 8; ++bitIndex)
        {
            bool one = ((keys[item] >> (RADIX_SHIFT + bitIndex)) & 1) != 0;
            uint4 ones = WaveActiveBallot(valid[item] && one);
            peers &= one ? ones : ~ones;
        }

        uint before;
        uint total;
        peerCounts(peers, before, total);
        uint prior = 0;
        if (valid[item] && before == 0)
            InterlockedAdd(gData[wave * RADIX + ((keys[item] >> RADIX_SHIFT) & 255)], total, prior);
        ranks[item] = WaveReadLaneAt(prior, firstPeer(peers)) + before;
    }
    GroupMemoryBarrierWithGroupSync();

    uint digitCount = 0;
    for (uint w = 0; w < waveCount; ++w)
        digitCount += gData[w * RADIX + tid];
    uint digitPrefix = WavePrefixSum(digitCount);
    if (lane == waveWidth - 1)
        gData[TILE_SIZE + wave] = digitPrefix + digitCount;
    GroupMemoryBarrierWithGroupSync();

    if (wave == 0)
    {
        uint chunk = lane < waveCount ? gData[TILE_SIZE + lane] : 0;
        uint chunkPrefix = WavePrefixSum(chunk);
        if (lane < waveCount)
            gData[TILE_SIZE + lane] = chunkPrefix;
    }
    GroupMemoryBarrierWithGroupSync();
    uint chunkBase = gData[TILE_SIZE + wave];
    GroupMemoryBarrierWithGroupSync();
    gData[TILE_SIZE + tid] = chunkBase + digitPrefix;
    GroupMemoryBarrierWithGroupSync();

    [unroll]
    for (uint item = 0; item < ITEMS_PER_THREAD; ++item)
    {
        if (valid[item])
        {
            uint digit = (keys[item] >> RADIX_SHIFT) & 255;
            for (uint w = 0; w < wave; ++w)
                ranks[item] += gData[w * RADIX + digit];
            ranks[item] += gData[TILE_SIZE + digit];
        }
    }
    GroupMemoryBarrierWithGroupSync();

    [unroll]
    for (uint item = 0; item < ITEMS_PER_THREAD; ++item)
        if (valid[item])
            gData[ranks[item]] = keys[item];
    GroupMemoryBarrierWithGroupSync();

    uint sortedKeys[ITEMS_PER_THREAD];
    [unroll]
    for (uint item = 0; item < ITEMS_PER_THREAD; ++item)
    {
        uint local = tid + item * THREADS;
        if (local < validCount)
        {
            uint key = gData[local];
            sortedKeys[item] = key;
            uint digit = (key >> RADIX_SHIFT) & 255;
            uint destination = rowBegin + tileOffsets[tile * RADIX + digit]
                + local - gData[TILE_SIZE + digit];
            outputKeys[destination] = key;
        }
    }
    GroupMemoryBarrierWithGroupSync();

    [unroll]
    for (uint item = 0; item < ITEMS_PER_THREAD; ++item)
    {
        if (valid[item])
        {
            uint local = waveBegin + item * waveWidth + lane;
            gData[ranks[item]] = inputValues[tileBegin + local];
        }
    }
    GroupMemoryBarrierWithGroupSync();

    [unroll]
    for (uint item = 0; item < ITEMS_PER_THREAD; ++item)
    {
        uint local = tid + item * THREADS;
        if (local < validCount)
        {
            uint digit = (sortedKeys[item] >> RADIX_SHIFT) & 255;
            uint destination = rowBegin + tileOffsets[tile * RADIX + digit]
                + local - gData[TILE_SIZE + digit];
            outputValues[destination] = gData[local];
        }
    }
}
