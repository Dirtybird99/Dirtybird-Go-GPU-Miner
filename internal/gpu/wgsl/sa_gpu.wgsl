// GPU-resident suffix array via prefix doubling: the ENTIRE construction for a
// hash runs inside one workgroup with no host round-trips. Batched — workgroup
// g builds hash g's suffix array over its slab slice. This is the speedup that
// replaces the host-orchestrated per-round readback path in sa.go.
//
// Per hash, per doubling round h: stably sort the suffix indices by
// (rank[s], rank[s+h]) using two inline-key counting-radix sorts (LSD: rank2
// then rank1), then rebuild ranks from the new order. Ranks fit in <2^17, so
// each sort is 3 bytes / 3 digit passes. Bounded at ceil(log2 n) rounds.
//
// Buffers are laid out per-hash at stride STRIDE (= MAX_LENGTH index space);
// text is packed 4 bytes/u32 at stride STRIDE/4 words. All arrays for a hash
// live in global memory; workgroup memory holds only histograms/scan scratch.

const SG_WG: u32 = 256u;

// Binding order: read-only inputs first (text, lens), then read_write outputs
// (sa, sa2, rank, rank2), then the uniform — matching the KernelRun harness.
@group(0) @binding(0) var<storage, read> sg_text: array<u32>;      // packed 4 bytes/u32
@group(0) @binding(1) var<storage, read> sg_lens: array<u32>;      // n per hash
@group(0) @binding(2) var<storage, read_write> sg_sa: array<u32>;
@group(0) @binding(3) var<storage, read_write> sg_sa2: array<u32>;
@group(0) @binding(4) var<storage, read_write> sg_rank: array<u32>;
@group(0) @binding(5) var<storage, read_write> sg_rank2: array<u32>;
struct SGParams { num_hashes: u32, stride: u32 } // stride in index elements
@group(0) @binding(6) var<uniform> sg_p: SGParams;

var<workgroup> sg_hist: array<atomic<u32>, 256>;
var<workgroup> sg_base: array<u32, 256>;
var<workgroup> sg_run: array<u32, 256>;
var<workgroup> sg_tileDigit: array<u32, 256>;
var<workgroup> sg_tileHist: array<atomic<u32>, 256>;
var<workgroup> sg_n: u32;
var<workgroup> sg_off: u32;       // index base for this hash's sa/rank
var<workgroup> sg_toff: u32;      // word base for this hash's packed text
var<workgroup> sg_h: u32;
var<workgroup> sg_keySel: u32;    // 0 = rank2 (rank[s+h]+1), 1 = rank1 (rank[s])
var<workgroup> sg_done: u32;
var<workgroup> sg_scan: array<u32, 256>; // intra-tile scan buffer for rank rebuild
var<workgroup> sg_scanBase: u32;         // running rank across tiles
var<workgroup> sg_maxRank: u32;          // current max rank (drives adaptive pass count)
var<workgroup> sg_passes: u32;           // radix byte-passes needed this round

fn sg_byte(i: u32) -> u32 {
    let w = sg_text[sg_toff + (i >> 2u)];
    return (w >> ((i & 3u) * 8u)) & 0xffu;
}

// key of the suffix index s under the current sort selector.
fn sg_key(s: u32) -> u32 {
    if (sg_keySel == 1u) {
        return sg_rank[sg_off + s];
    }
    let sh = s + sg_h;
    if (sh < sg_n) { return sg_rank[sg_off + sh] + 1u; }
    return 0u;
}

fn sg_digit(key: u32, pass: u32) -> u32 {
    return (key >> (pass * 8u)) & 0xffu;
}

// Read/write the current slot buffer. Passes ping-pong between sg_sa and
// sg_sa2 (srcA true => read sg_sa / write sg_sa2), avoiding a per-pass copy.
fn sg_slotRead(srcA: bool, idx: u32) -> u32 {
    if (srcA) { return sg_sa[idx]; }
    return sg_sa2[idx];
}
fn sg_slotWrite(srcA: bool, idx: u32, v: u32) {
    if (srcA) { sg_sa2[idx] = v; } else { sg_sa[idx] = v; }
}

// One stable counting-radix sort of the suffix indices by sg_key(). Ping-pongs
// sg_sa <-> sg_sa2 across passes; a single normalize copy at the end leaves the
// result in sg_sa. Uses sg_passes byte-passes — only as many as the current key
// range (rank2 max = sg_maxRank+1) needs.
fn sg_sort(t: u32, nTiles: u32) {
    for (var pass: u32 = 0u; pass < sg_passes; pass = pass + 1u) {
        let srcA = (pass & 1u) == 0u;
        atomicStore(&sg_hist[t], 0u);
        workgroupBarrier();
        for (var tile: u32 = 0u; tile < nTiles; tile = tile + 1u) {
            let i = tile * SG_WG + t;
            if (i < sg_n) {
                atomicAdd(&sg_hist[sg_digit(sg_key(sg_slotRead(srcA, sg_off + i)), pass)], 1u);
            }
        }
        workgroupBarrier();

        if (t == 0u) {
            var acc: u32 = 0u;
            for (var d: u32 = 0u; d < 256u; d = d + 1u) {
                sg_base[d] = acc;
                acc = acc + atomicLoad(&sg_hist[d]);
                sg_run[d] = 0u;
            }
        }
        workgroupBarrier();

        for (var tile: u32 = 0u; tile < nTiles; tile = tile + 1u) {
            let i = tile * SG_WG + t;
            let active = i < sg_n;
            var s: u32 = 0u;
            var d: u32 = 0u;
            atomicStore(&sg_tileHist[t], 0u);
            sg_tileDigit[t] = 0xffffffffu;
            workgroupBarrier();

            if (active) {
                s = sg_slotRead(srcA, sg_off + i);
                d = sg_digit(sg_key(s), pass);
                sg_tileDigit[t] = d;
                atomicAdd(&sg_tileHist[d], 1u);
            }
            workgroupBarrier();

            if (active) {
                var rank: u32 = 0u;
                for (var tp: u32 = 0u; tp < t; tp = tp + 1u) {
                    if (sg_tileDigit[tp] == d) { rank = rank + 1u; }
                }
                sg_slotWrite(srcA, sg_off + sg_base[d] + sg_run[d] + rank, s);
            }
            workgroupBarrier();

            sg_run[t] = sg_run[t] + atomicLoad(&sg_tileHist[t]);
            workgroupBarrier();
        }
    }

    // If an odd number of passes ran, the sorted order is in sg_sa2; normalize
    // back to sg_sa with a single sweep.
    if ((sg_passes & 1u) == 1u) {
        for (var tile: u32 = 0u; tile < nTiles; tile = tile + 1u) {
            let i = tile * SG_WG + t;
            if (i < sg_n) { sg_sa[sg_off + i] = sg_sa2[sg_off + i]; }
        }
        workgroupBarrier();
    }
}

@compute @workgroup_size(256)
fn sa_batch_main(@builtin(workgroup_id) wid: vec3<u32>,
                 @builtin(local_invocation_id) lid: vec3<u32>) {
    let g = wid.x;
    if (g >= sg_p.num_hashes) { return; }
    let t = lid.x;

    if (t == 0u) {
        sg_n = sg_lens[g];
        sg_off = g * sg_p.stride;
        sg_toff = g * (sg_p.stride / 4u);
    }
    workgroupBarrier();
    let n = sg_n;
    let off = sg_off;
    if (n == 0u) { return; }
    let nTiles = (n + SG_WG - 1u) / SG_WG;

    // init sa[i]=i, rank[i]=first byte
    for (var tile: u32 = 0u; tile < nTiles; tile = tile + 1u) {
        let i = tile * SG_WG + t;
        if (i < n) {
            sg_sa[off + i] = i;
            sg_rank[off + i] = sg_byte(i);
        }
    }
    if (t == 0u) { sg_maxRank = 255u; } // initial ranks are byte values
    workgroupBarrier();

    for (var h: u32 = 1u; h < n; h = h * 2u) {
        if (t == 0u) {
            sg_h = h;
            // bytes needed to cover key range (rank+1 <= maxRank+1)
            let hi = sg_maxRank + 1u;
            var pb: u32 = 1u;
            if (hi >= 0x100u) { pb = 2u; }
            if (hi >= 0x10000u) { pb = 3u; }
            sg_passes = pb;
        }
        // sort by rank2 (low), then rank1 (high) — stable LSD over the pair
        if (t == 0u) { sg_keySel = 0u; }
        workgroupBarrier();
        sg_sort(t, nTiles);
        if (t == 0u) { sg_keySel = 1u; }
        workgroupBarrier();
        sg_sort(t, nTiles);

        // Rebuild ranks into sg_rank2 via a parallel tile-serial inclusive scan
        // of the "pair changed" flags. newrank[sa[k]] = 1-based group index.
        if (t == 0u) { sg_scanBase = 0u; }
        workgroupBarrier();
        for (var tile: u32 = 0u; tile < nTiles; tile = tile + 1u) {
            let k = tile * SG_WG + t;
            let active = k < n;
            var chg: u32 = 0u;
            var s: u32 = 0u;
            if (active) {
                s = sg_sa[off + k];
                let r1 = sg_rank[off + s];
                var r2: u32 = 0u;
                if (s + h < n) { r2 = sg_rank[off + s + h] + 1u; }
                if (k == 0u) {
                    chg = 1u;
                } else {
                    let sp = sg_sa[off + k - 1u];
                    let pr1 = sg_rank[off + sp];
                    var pr2: u32 = 0u;
                    if (sp + h < n) { pr2 = sg_rank[off + sp + h] + 1u; }
                    if (r1 != pr1 || r2 != pr2) { chg = 1u; }
                }
            }
            sg_scan[t] = chg;
            workgroupBarrier();

            // Hillis-Steele inclusive scan over the 256 lanes.
            for (var d: u32 = 1u; d < SG_WG; d = d * 2u) {
                var v = sg_scan[t];
                if (t >= d) { v = v + sg_scan[t - d]; }
                workgroupBarrier();
                sg_scan[t] = v;
                workgroupBarrier();
            }

            if (active) {
                sg_rank2[off + s] = sg_scanBase + sg_scan[t];
            }
            workgroupBarrier();
            if (t == 0u) { sg_scanBase = sg_scanBase + sg_scan[SG_WG - 1u]; }
            workgroupBarrier();
        }
        if (t == 0u) {
            sg_done = select(0u, 1u, sg_scanBase == n);
            sg_maxRank = sg_scanBase; // ranks are now 1..scanBase
        }
        workgroupBarrier();

        // copy new ranks back
        for (var tile: u32 = 0u; tile < nTiles; tile = tile + 1u) {
            let i = tile * SG_WG + t;
            if (i < n) { sg_rank[off + i] = sg_rank2[off + i]; }
        }
        workgroupBarrier();

        let done = workgroupUniformLoad(&sg_done);
        if (done == 1u) { break; }
    }
}
