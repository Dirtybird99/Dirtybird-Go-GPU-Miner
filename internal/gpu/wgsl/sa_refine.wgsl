// GPU-resident suffix array with ACTIVE-SET COMPACTION — the throughput lever
// over plain prefix doubling (sa_gpu.wgsl). After each doubling round, suffixes
// that have become unique (singletons) keep their final SA slot and are dropped
// from the active set; only the still-tied suffixes are re-sorted next round.
// The active set shrinks fast (measured ~72% tied at depth 4, collapsing past
// ~256 bytes), so total sort work is ~2-3n instead of prefix doubling's ~13n.
//
// One workgroup per hash. Ranks are POSITION ranks (see rf_rebuild): the
// full-n rank rebuild and active-list scan run ONCE to seed the loop, then
// each doubling round re-ranks and rebuilds the next active list segment-
// locally (rf_refineSeg), touching only the active values — never all n.
//
// Bindings: read-only text+lens, then rw sa, rank, rankTmp, av (dense active
// values / round-0 scratch), avTmp (sort scratch), active (slot-index list).

enable subgroups;

const RF_WG: u32 = 256u;

@group(0) @binding(0) var<storage, read> rf_text: array<u32>;   // packed 4 bytes/u32
@group(0) @binding(1) var<storage, read> rf_lens: array<u32>;
@group(0) @binding(2) var<storage, read_write> rf_sa: array<u32>;
@group(0) @binding(3) var<storage, read_write> rf_rank: array<u32>;
@group(0) @binding(4) var<storage, read_write> rf_rankTmp: array<u32>;
@group(0) @binding(5) var<storage, read_write> rf_av: array<u32>;
@group(0) @binding(6) var<storage, read_write> rf_avTmp: array<u32>;
@group(0) @binding(7) var<storage, read_write> rf_active: array<u32>;
@group(0) @binding(8) var<storage, read_write> rf_keyTmp: array<u32>;
// dbg_rounds: profiling knob. 0 = production (run the doubling loop to
// completion). Otherwise run exactly dbg_rounds-1 doubling rounds after the
// coarse sort, then stop — the SA is left partial/invalid, so this is for
// TIMING ONLY (per-round cost as the active set collapses), never correctness.
struct RFParams { num_hashes: u32, stride: u32, dbg_rounds: u32 }
@group(0) @binding(9) var<uniform> rf_p: RFParams;

var<workgroup> rf_hist: array<atomic<u32>, 256>;
var<workgroup> rf_base: array<u32, 256>;
var<workgroup> rf_run: array<u32, 256>;
var<workgroup> rf_tileHist: array<atomic<u32>, 256>;
// Per-subgroup per-digit counts for the cross-subgroup stable-rank correction.
// Up to 8 subgroups (256 / min subgroup size 32) x 256 digits.
var<workgroup> rf_sgHist: array<atomic<u32>, 2048>;
var<workgroup> rf_scan: array<u32, 256>;
var<workgroup> rf_scanBase: u32;
var<workgroup> rf_n: u32;
var<workgroup> rf_off: u32;
var<workgroup> rf_toff: u32;
var<workgroup> rf_h: u32;
var<workgroup> rf_keySel: u32;   // 0 = rank2, 1 = rank1, 2 = coarse lo, 3 = coarse hi, 4 = packed active key
var<workgroup> rf_cmpCoarse: u32; // rf_rebuild: 1 = group by the 4-byte coarse key
var<workgroup> rf_mode: u32;     // 0 = sort rf_sa (round 0), 1 = sort rf_av (refine)
var<workgroup> rf_count: u32;    // elements in the current sort (n or activeCount)
var<workgroup> rf_passes: u32;
var<workgroup> rf_maxRank: u32;
var<workgroup> rf_activeCount: u32;
var<workgroup> rf_groupCount: u32;
var<workgroup> rf_compactOK: u32;

fn rf_byte(i: u32) -> u32 {
    return (rf_text[rf_toff + (i >> 2u)] >> ((i & 3u) * 8u)) & 0xffu;
}

// Coarse depth-4 symbol: real bytes map to 1..256, past-the-end maps to 0 — the
// implicit sentinel that sorts below every byte, matching the SAIS oracle's
// shorter-suffix-first order. Each symbol needs 9 bits, so a 4-byte key is 36
// bits and travels as two 18-bit halves through a stable LSD sort (lo, then hi).
fn rf_sym(s: u32, d: u32) -> u32 {
    let p = s + d;
    if (p < rf_n) { return rf_byte(p) + 1u; }
    return 0u;
}
fn rf_coarseHi(s: u32) -> u32 { return (rf_sym(s, 0u) << 9u) | rf_sym(s, 1u); }
fn rf_coarseLo(s: u32) -> u32 { return (rf_sym(s, 2u) << 9u) | rf_sym(s, 3u); }

// Largest coarse half-key, (256<<9)|256 — forces rf_setPasses to 3 radix passes.
const RF_COARSE_MAX: u32 = 131328u;

fn rf_key(s: u32) -> u32 {
    if (rf_keySel == 2u) { return rf_coarseLo(s); }
    if (rf_keySel == 3u) { return rf_coarseHi(s); }
    if (rf_keySel == 1u) { return rf_rank[rf_off + s]; }
    let sh = s + rf_h;
    if (sh < rf_n) { return rf_rank[rf_off + sh] + 1u; }
    return 0u;
}

fn rf_digit(key: u32, pass: u32) -> u32 { return (key >> (pass * 8u)) & 0xffu; }

// popcount of subgroup-ballot mask `m` restricted to lanes with id < sgi.
fn rf_priorPop(m: vec4<u32>, sgi: u32) -> u32 {
    var c: u32 = 0u;
    if (sgi >= 32u) { c = c + countOneBits(m.x); } else { return countOneBits(m.x & ((1u << sgi) - 1u)); }
    if (sgi >= 64u) { c = c + countOneBits(m.y); } else { return c + countOneBits(m.y & ((1u << (sgi - 32u)) - 1u)); }
    if (sgi >= 96u) { c = c + countOneBits(m.z); } else { return c + countOneBits(m.z & ((1u << (sgi - 64u)) - 1u)); }
    return c + countOneBits(m.w & ((1u << (sgi - 96u)) - 1u));
}

// slot access for the current sort. mode 0: primary rf_sa, scratch rf_av.
// mode 1: primary rf_av, scratch rf_avTmp. srcA selects primary vs scratch.
fn rf_read(srcA: bool, idx: u32) -> u32 {
    if (rf_mode == 0u) {
        if (srcA) { return rf_sa[idx]; } return rf_av[idx];
    }
    if (srcA) { return rf_av[idx]; } return rf_avTmp[idx];
}
fn rf_write(srcA: bool, idx: u32, v: u32) {
    if (rf_mode == 0u) {
        if (srcA) { rf_av[idx] = v; } else { rf_sa[idx] = v; }
        return;
    }
    if (srcA) { rf_avTmp[idx] = v; } else { rf_av[idx] = v; }
}

// key scratch ping-pong, in lockstep with the value ping-pong: even passes
// read keys from rf_rankTmp (dead during sorts) and write rf_keyTmp; odd
// passes the reverse.
fn rf_keyRead(srcA: bool, idx: u32) -> u32 {
    if (srcA) { return rf_rankTmp[idx]; }
    return rf_keyTmp[idx];
}
fn rf_keyWrite(srcA: bool, idx: u32, v: u32) {
    if (srcA) { rf_keyTmp[idx] = v; } else { rf_rankTmp[idx] = v; }
}

// Stable counting-radix sort of the current slot array (rf_count elements) by
// rf_key(), starting on startA and ping-ponging primary<->scratch. Callers
// chain the low/high key sorts, whose equal pass counts always finish on A.
// Keys are materialized ONCE up front (the sort's only random rank[] reads)
// and travel with their values through every pass.
fn rf_sort(t: u32, sgi: u32, sgSize: u32, startA: bool) {
    let cnt = rf_count;
    let nTiles = (cnt + RF_WG - 1u) / RF_WG;
    let mySub = t / sgSize;         // this lane's subgroup index within the tile
    let numSub = RF_WG / sgSize;    // subgroups per 256-lane tile
    for (var tile: u32 = 0u; tile < nTiles; tile = tile + 1u) {
        let i = tile * RF_WG + t;
        if (i < cnt) {
            let idx = rf_off + i;
            let s = rf_read(startA, idx);
            var key = rf_key(s);
            if (rf_keySel == 4u) {
                let sh = s + rf_h;
                var r2 = 0u;
                if (sh < rf_n) { r2 = rf_rank[rf_off + sh] + 1u; }
                key = (rf_keyTmp[idx] << 17u) | r2;
            }
            if (startA) { rf_rankTmp[idx] = key; } else { rf_keyTmp[idx] = key; }
        }
    }
    workgroupBarrier();
    for (var pass: u32 = 0u; pass < rf_passes; pass = pass + 1u) {
        let srcA = startA == ((pass & 1u) == 0u);
        atomicStore(&rf_hist[t], 0u);
        workgroupBarrier();
        for (var tile: u32 = 0u; tile < nTiles; tile = tile + 1u) {
            let i = tile * RF_WG + t;
            if (i < cnt) {
                atomicAdd(&rf_hist[rf_digit(rf_keyRead(srcA, rf_off + i), pass)], 1u);
            }
        }
        workgroupBarrier();
        if (t == 0u) {
            var acc: u32 = 0u;
            for (var d: u32 = 0u; d < 256u; d = d + 1u) {
                rf_base[d] = acc;
                acc = acc + atomicLoad(&rf_hist[d]);
                rf_run[d] = 0u;
            }
        }
        workgroupBarrier();
        for (var tile: u32 = 0u; tile < nTiles; tile = tile + 1u) {
            let i = tile * RF_WG + t;
            let active = i < cnt;
            // zero per-tile histograms (tileHist[256] + sgHist[numSub*256])
            atomicStore(&rf_tileHist[t], 0u);
            for (var k: u32 = 0u; k < numSub; k = k + 1u) {
                atomicStore(&rf_sgHist[k * 256u + t], 0u);
            }
            var s: u32 = 0u;
            var key: u32 = 0u;
            var d: u32 = 0u;
            if (active) {
                s = rf_read(srcA, rf_off + i);
                key = rf_keyRead(srcA, rf_off + i);
                d = rf_digit(key, pass);
            }
            workgroupBarrier();
            if (active) {
                atomicAdd(&rf_tileHist[d], 1u);
                atomicAdd(&rf_sgHist[mySub * 256u + d], 1u);
            }
            workgroupBarrier();

            // within-subgroup stable rank via ballot bit-match on the 8-bit digit
            let am = subgroupBallot(active);
            var sm = am;
            for (var b: u32 = 0u; b < 8u; b = b + 1u) {
                let one = active && (((d >> b) & 1u) == 1u);
                let bb = subgroupBallot(one);
                if (((d >> b) & 1u) == 1u) { sm = sm & bb; } else { sm = sm & (am & ~bb); }
            }
            let rankInSub = rf_priorPop(sm, sgi);

            if (active) {
                // + same-digit count in earlier subgroups of this tile
                var earlier: u32 = 0u;
                for (var k: u32 = 0u; k < mySub; k = k + 1u) {
                    earlier = earlier + atomicLoad(&rf_sgHist[k * 256u + d]);
                }
                let dst = rf_off + rf_base[d] + rf_run[d] + earlier + rankInSub;
                rf_write(srcA, dst, s);
                rf_keyWrite(srcA, dst, key);
            }
            workgroupBarrier();
            rf_run[t] = rf_run[t] + atomicLoad(&rf_tileHist[t]);
            workgroupBarrier();
        }
    }
}

// pair of a suffix index under the current h (for rank rebuild / grouping)
fn rf_r1(s: u32) -> u32 { return rf_rank[rf_off + s]; }
fn rf_r2(s: u32) -> u32 {
    let sh = s + rf_h;
    if (sh < rf_n) { return rf_rank[rf_off + sh] + 1u; }
    return 0u;
}

fn rf_setPasses(t: u32) {
    if (t == 0u) {
        let hi = rf_maxRank + 1u;
        var pb: u32 = 1u;
        if (hi >= 0x100u) { pb = 2u; }
        if (hi >= 0x10000u) { pb = 3u; }
        if (hi >= 0x1000000u) { pb = 4u; }
        rf_passes = pb;
    }
}

// Full rank rebuild over rf_sa, emitting POSITION RANKS: every suffix in a tied
// group takes the SLOT INDEX of that group's first element as its rank. Position
// ranks are monotonic in sort order (so the two-key sort is unchanged) but, being
// absolute slot indices, let a tied segment be re-ranked from its own slots with
// no global renumbering — the property the segment-local refine relies on.
//
// Mechanism: the fast subgroup inclusive-add scan of "pair changed" flags still
// gives each slot its dense group id (1..G); a group's first slot writes its own
// index into rf_avTmp indexed by that id (a group-start table, free scratch here),
// then the copy-back maps each suffix's dense id through that table to its start
// slot. Sets rf_maxRank.
fn rf_rebuild(t: u32, sgi: u32, sgSize: u32) {
    let n = rf_n;
    let nTiles = (n + RF_WG - 1u) / RF_WG;
    let mySub = t / sgSize;
    let numSub = RF_WG / sgSize;
    // Uniform across the workgroup: last written by t==0 before a barrier.
    let cmpC = workgroupUniformLoad(&rf_cmpCoarse);
    if (t == 0u) { rf_scanBase = 0u; }
    workgroupBarrier();
    for (var tile: u32 = 0u; tile < nTiles; tile = tile + 1u) {
        let k = tile * RF_WG + t;
        let active = k < n;
        var chg: u32 = 0u;
        var s: u32 = 0u;
        if (active) {
            s = rf_sa[rf_off + k];
            if (k == 0u) {
                chg = 1u;
            } else {
                let sp = rf_sa[rf_off + k - 1u];
                if (cmpC == 1u) {
                    if (rf_coarseHi(s) != rf_coarseHi(sp) || rf_coarseLo(s) != rf_coarseLo(sp)) { chg = 1u; }
                } else if (rf_r1(s) != rf_r1(sp) || rf_r2(s) != rf_r2(sp)) {
                    chg = 1u;
                }
            }
        }
        let sgInc = subgroupExclusiveAdd(chg) + chg;
        if (sgi == sgSize - 1u) { rf_scan[mySub] = sgInc; }
        workgroupBarrier();
        var base: u32 = 0u;
        for (var q: u32 = 0u; q < mySub; q = q + 1u) { base = base + rf_scan[q]; }
        if (active) {
            let did = rf_scanBase + base + sgInc; // dense group id (1..G)
            rf_rankTmp[rf_off + s] = did;          // suffix -> its group id
            if (chg == 1u) { rf_avTmp[rf_off + did] = k; } // group id -> start slot
        }
        workgroupBarrier();
        if (t == 0u) {
            var tot: u32 = 0u;
            for (var q: u32 = 0u; q < numSub; q = q + 1u) { tot = tot + rf_scan[q]; }
            rf_scanBase = rf_scanBase + tot;
        }
        workgroupBarrier();
    }
    // Map each suffix's dense group id through the group-start table to its
    // position rank. rf_avTmp[did] was written by exactly the group that owns
    // did, and every stored id is a real group, so every read is populated.
    for (var tile: u32 = 0u; tile < nTiles; tile = tile + 1u) {
        let i = tile * RF_WG + t;
        if (i < n) { rf_rank[rf_off + i] = rf_avTmp[rf_off + rf_rankTmp[rf_off + i]]; }
    }
    workgroupBarrier();
    if (t == 0u) {
        rf_maxRank = n; // position ranks span [0, n); size the radix to n
    }
    workgroupBarrier();
}

// Build the active list: slots whose group has size > 1 (rank equals a
// neighbour's), compacted in increasing slot order via a tile-serial scan
// (subgroup-accelerated, same shape as rf_rebuild's).
fn rf_buildActive(t: u32, sgi: u32, sgSize: u32) {
    let n = rf_n;
    let nTiles = (n + RF_WG - 1u) / RF_WG;
    let mySub = t / sgSize;
    let numSub = RF_WG / sgSize;
    if (t == 0u) { rf_scanBase = 0u; }
    workgroupBarrier();
    for (var tile: u32 = 0u; tile < nTiles; tile = tile + 1u) {
        let k = tile * RF_WG + t;
        let active = k < n;
        var ns: u32 = 0u; // 1 if slot k is in a non-singleton group
        if (active) {
            let r = rf_rank[rf_off + rf_sa[rf_off + k]];
            var same = false;
            if (k > 0u && rf_rank[rf_off + rf_sa[rf_off + k - 1u]] == r) { same = true; }
            if (k + 1u < n && rf_rank[rf_off + rf_sa[rf_off + k + 1u]] == r) { same = true; }
            if (same) { ns = 1u; }
        }
        let sgInc = subgroupExclusiveAdd(ns) + ns;
        if (sgi == sgSize - 1u) { rf_scan[mySub] = sgInc; }
        workgroupBarrier();
        var base: u32 = 0u;
        for (var q: u32 = 0u; q < mySub; q = q + 1u) { base = base + rf_scan[q]; }
        if (active && ns == 1u) {
            // exclusive position = base + inclusive - 1
            rf_active[rf_off + rf_scanBase + base + sgInc - 1u] = k;
        }
        workgroupBarrier();
        if (t == 0u) {
            var tot: u32 = 0u;
            for (var q: u32 = 0u; q < numSub; q = q + 1u) { tot = tot + rf_scan[q]; }
            rf_scanBase = rf_scanBase + tot;
        }
        workgroupBarrier();
    }
    if (t == 0u) { rf_activeCount = rf_scanBase; }
    workgroupBarrier();
}

// Dense IDs for active rank1 groups. They are only the high half of the
// current round's composite key; global position ranks remain untouched.
fn rf_seedGroupIDs(t: u32, sgi: u32, sgSize: u32) {
    let ac = rf_activeCount;
    let nTiles = (ac + RF_WG - 1u) / RF_WG;
    let mySub = t / sgSize;
    let numSub = RF_WG / sgSize;
    if (t == 0u) { rf_scanBase = 0u; }
    workgroupBarrier();
    for (var tile: u32 = 0u; tile < nTiles; tile = tile + 1u) {
        let j = tile * RF_WG + t;
        let act = j < ac;
        var b = 0u;
        if (act) {
            let s = rf_sa[rf_off + rf_active[rf_off + j]];
            if (j == 0u) { b = 1u; }
            else {
                let p = rf_sa[rf_off + rf_active[rf_off + j - 1u]];
                if (rf_rank[rf_off + s] != rf_rank[rf_off + p]) { b = 1u; }
            }
        }
        let inc = subgroupExclusiveAdd(b) + b;
        if (sgi == sgSize - 1u) { rf_scan[mySub] = inc; }
        workgroupBarrier();
        var base = 0u;
        for (var q: u32 = 0u; q < mySub; q = q + 1u) { base = base + rf_scan[q]; }
        if (act) { rf_keyTmp[rf_off + j] = rf_scanBase + base + inc - 1u; }
        workgroupBarrier();
        if (t == 0u) {
            var total = 0u;
            for (var q: u32 = 0u; q < numSub; q = q + 1u) { total = total + rf_scan[q]; }
            rf_scanBase = rf_scanBase + total;
        }
        workgroupBarrier();
    }
    if (t == 0u) {
        rf_groupCount = rf_scanBase;
        rf_compactOK = 0u;
        if (rf_groupCount < 32767u && rf_n >= 1024u) { rf_compactOK = 1u; }
        rf_maxRank = (rf_groupCount << 17u) | rf_n;
    }
    workgroupBarrier();
    storageBarrier();
}

// Segment-local refine: the per-round replacement for rf_rebuild + rf_buildActive.
// With the active values freshly sorted and scattered into their slots, it
// recomputes position ranks and the NEXT active list touching only the ac active
// elements — never all n. That removes the two full-n sweeps that the profiler
// showed were ~40% of SA time.
//
// Because ranks are POSITION ranks, a suffix that becomes unique this round takes
// its own slot as its final rank and is simply left out of the next active list —
// never read or written again. Only suffixes still tied cost anything.
//
// Three scans over ac (all subgroup-accelerated, same shape as rf_rebuild):
//  1. boundary flags (old ranks, read before any rank is overwritten) + an
//     inclusive-add sub-run numbering; each sub-run's first element records its
//     slot (= the sub-run's new position rank) in a start-slot table.
//  2. map each element's suffix to its sub-run's start slot = its new rank.
//  3. compact the non-singleton elements (sub-run length > 1) into the next
//     active list, then copy it back over rf_active.
fn rf_refineSeg(t: u32, sgi: u32, sgSize: u32) {
    let ac = rf_activeCount;
    let nTiles = (ac + RF_WG - 1u) / RF_WG;
    let mySub = t / sgSize;
    let numSub = RF_WG / sgSize;

    // Pass 1: boundary flags + sub-run ids + start-slot table.
    if (t == 0u) { rf_scanBase = 0u; }
    workgroupBarrier();
    for (var tile: u32 = 0u; tile < nTiles; tile = tile + 1u) {
        let j = tile * RF_WG + t;
        let act = j < ac;
        var b: u32 = 0u;
        var slot: u32 = 0u;
        if (act) {
            slot = rf_active[rf_off + j];
            let s = rf_sa[rf_off + slot];
            if (j == 0u) {
                b = 1u;
            } else {
                let sp = rf_sa[rf_off + rf_active[rf_off + j - 1u]];
                if (rf_rank[rf_off + s] != rf_rank[rf_off + sp] || rf_r2(s) != rf_r2(sp)) { b = 1u; }
            }
        }
        let sgInc = subgroupExclusiveAdd(b) + b;
        if (sgi == sgSize - 1u) { rf_scan[mySub] = sgInc; }
        workgroupBarrier();
        var base: u32 = 0u;
        for (var q: u32 = 0u; q < mySub; q = q + 1u) { base = base + rf_scan[q]; }
        if (act) {
            let qid = rf_scanBase + base + sgInc; // sub-run id (1..Q)
            rf_rankTmp[rf_off + j] = qid;         // stash by element index
            rf_keyTmp[rf_off + j] = b;            // stash boundary flag
            if (b == 1u) { rf_avTmp[rf_off + qid] = slot; } // sub-run start slot
        }
        workgroupBarrier();
        if (t == 0u) {
            var tot: u32 = 0u;
            for (var q: u32 = 0u; q < numSub; q = q + 1u) { tot = tot + rf_scan[q]; }
            rf_scanBase = rf_scanBase + tot;
        }
        workgroupBarrier();
    }
    if (t == 0u) {
        rf_groupCount = rf_scanBase;
        rf_compactOK = 0u;
        if (rf_groupCount < 32767u && rf_n >= 1024u) { rf_compactOK = 1u; }
        rf_maxRank = (rf_groupCount << 17u) | rf_n;
    }
    workgroupBarrier();
    // Pass 2: new rank = sub-run start slot. All old-rank reads (pass 1) are done.
    for (var tile: u32 = 0u; tile < nTiles; tile = tile + 1u) {
        let j = tile * RF_WG + t;
        if (j < ac) {
            let qid = rf_rankTmp[rf_off + j];
            let s = rf_sa[rf_off + rf_active[rf_off + j]];
            rf_rank[rf_off + s] = rf_avTmp[rf_off + qid];
        }
    }
    workgroupBarrier();
    // Pass 3: compact non-singleton elements into the next active list (rf_av).
    if (t == 0u) { rf_scanBase = 0u; }
    workgroupBarrier();
    for (var tile: u32 = 0u; tile < nTiles; tile = tile + 1u) {
        let j = tile * RF_WG + t;
        let act = j < ac;
        var ns: u32 = 0u;
        var slot: u32 = 0u;
        var qid: u32 = 0u;
        if (act) {
            slot = rf_active[rf_off + j];
            qid = rf_rankTmp[rf_off + j];
            let b = rf_keyTmp[rf_off + j];
            var bnext: u32 = 1u; // past the end counts as a boundary
            if (j + 1u < ac) { bnext = rf_keyTmp[rf_off + j + 1u]; }
            // non-singleton iff it shares its sub-run with a neighbour: it is a
            // singleton only if it starts a run AND the next element starts one.
            if (!(b == 1u && bnext == 1u)) { ns = 1u; }
        }
        let sgInc = subgroupExclusiveAdd(ns) + ns;
        if (sgi == sgSize - 1u) { rf_scan[mySub] = sgInc; }
        workgroupBarrier();
        var base: u32 = 0u;
        for (var q: u32 = 0u; q < mySub; q = q + 1u) { base = base + rf_scan[q]; }
        if (act && ns == 1u) {
            let dst = rf_scanBase + base + sgInc - 1u;
            rf_av[rf_off + dst] = slot;
            rf_keyTmp[rf_off + dst] = qid - 1u;
        }
        workgroupBarrier();
        if (t == 0u) {
            var tot: u32 = 0u;
            for (var q: u32 = 0u; q < numSub; q = q + 1u) { tot = tot + rf_scan[q]; }
            rf_scanBase = rf_scanBase + tot;
        }
        workgroupBarrier();
    }
    let nac = rf_scanBase;
    // Pass 4: install the next active list.
    let nacTiles = (nac + RF_WG - 1u) / RF_WG;
    for (var tile: u32 = 0u; tile < nacTiles; tile = tile + 1u) {
        let j = tile * RF_WG + t;
        if (j < nac) { rf_active[rf_off + j] = rf_av[rf_off + j]; }
    }
    workgroupBarrier();
    if (t == 0u) {
        rf_activeCount = nac;
    }
    workgroupBarrier();
    storageBarrier();
}

@compute @workgroup_size(256)
fn sa_refine_main(@builtin(workgroup_id) wid: vec3<u32>,
                  @builtin(local_invocation_id) lid: vec3<u32>,
                  @builtin(subgroup_invocation_id) sgi: u32,
                  @builtin(subgroup_size) sgSize: u32) {
    let g = wid.x;
    if (g >= rf_p.num_hashes) { return; }
    let t = lid.x;
    if (t == 0u) {
        rf_n = rf_lens[g];
        rf_off = g * rf_p.stride;
        rf_toff = g * (rf_p.stride / 4u);
    }
    workgroupBarrier();
    let n = rf_n;
    let off = rf_off;
    if (n == 0u) { return; }
    let nTiles = (n + RF_WG - 1u) / RF_WG;

    for (var tile: u32 = 0u; tile < nTiles; tile = tile + 1u) {
        let i = tile * RF_WG + t;
        if (i < n) { rf_sa[off + i] = i; }
    }
    // Coarse round: one stable LSD sort straight to depth 4 (lo half, then hi
    // half), replacing the h=1 and h=2 doubling rounds and the full-n
    // rebuild + buildActive sweeps between them. Ranks are then seeded from the
    // depth-4 buckets, so the doubling loop below resumes at h=4.
    //
    // Every suffix whose 4-byte window runs off the end (s > n-4) carries a 0
    // symbol there, so it separates from every other suffix in this one sort —
    // depth-4 buckets of size > 1 contain only suffixes with s <= n-4.
    if (t == 0u) { rf_mode = 0u; rf_count = n; rf_maxRank = RF_COARSE_MAX; }
    rf_setPasses(t);
    let coarseSecondA = (workgroupUniformLoad(&rf_passes) & 1u) == 0u;
    if (t == 0u) { rf_keySel = 2u; }
    workgroupBarrier();
    rf_sort(t, sgi, sgSize, true);
    if (t == 0u) { rf_keySel = 3u; }
    workgroupBarrier();
    rf_sort(t, sgi, sgSize, coarseSecondA);
    if (t == 0u) { rf_cmpCoarse = 1u; }
    workgroupBarrier();
    rf_rebuild(t, sgi, sgSize);
    if (t == 0u) { rf_cmpCoarse = 0u; }
    workgroupBarrier();

    // Seed the active list once with a full-n scan; every later round derives
    // the next active list segment-locally inside rf_refineSeg (over ac, not n).
    rf_buildActive(t, sgi, sgSize);
    rf_seedGroupIDs(t, sgi, sgSize);

    var h: u32 = 4u;
    var roundIdx: u32 = 0u;
    loop {
        // Profiling cap (dbg_rounds != 0): stop after a fixed round count so the
        // host can time the loop truncated to k rounds. Production passes 0.
        if (rf_p.dbg_rounds != 0u && roundIdx + 1u >= rf_p.dbg_rounds) { break; }
        let ac = workgroupUniformLoad(&rf_activeCount);
        if (ac == 0u) { break; }
        if (h >= n) { break; }

        if (t == 0u) { rf_h = h; }
        workgroupBarrier();

        // gather active values into rf_av[off + j]
        let acTiles = (ac + RF_WG - 1u) / RF_WG;
        for (var tile: u32 = 0u; tile < acTiles; tile = tile + 1u) {
            let j = tile * RF_WG + t;
            if (j < ac) { rf_av[off + j] = rf_sa[off + rf_active[off + j]]; }
        }
        workgroupBarrier();

        // Pack dense active-group/rank2 keys when they fit; preserve the
        // original two-key path for short/sentinel-heavy edge cases.
        if (t == 0u) { rf_mode = 1u; rf_count = ac; }
        let packed = workgroupUniformLoad(&rf_compactOK);
        if (packed != 0u) {
            rf_setPasses(t);
            let packedPasses = workgroupUniformLoad(&rf_passes);
            if (t == 0u) { rf_keySel = 4u; }
            workgroupBarrier();
            rf_sort(t, sgi, sgSize, true);
        } else {
            if (t == 0u) { rf_maxRank = n; }
            workgroupBarrier();
            rf_setPasses(t);
            let refineSecondA = (workgroupUniformLoad(&rf_passes) & 1u) == 0u;
            if (t == 0u) { rf_keySel = 0u; }
            workgroupBarrier();
            rf_sort(t, sgi, sgSize, true);
            if (t == 0u) { rf_keySel = 1u; }
            workgroupBarrier();
            rf_sort(t, sgi, sgSize, refineSecondA);
        }

        // scatter refined values back to their slots
        for (var tile: u32 = 0u; tile < acTiles; tile = tile + 1u) {
            let j = tile * RF_WG + t;
            if (j < ac) { rf_sa[off + rf_active[off + j]] = rf_av[off + j]; }
        }
        workgroupBarrier();

        // segment-local rank rebuild + next active list (over ac, not n)
        rf_refineSeg(t, sgi, sgSize);
        h = h * 2u;
        roundIdx = roundIdx + 1u;
    }
}
