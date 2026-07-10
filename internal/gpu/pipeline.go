package gpu

import (
	"crypto/sha256"
	"encoding/binary"
	"fmt"
	"runtime"
	"sync"

	"github.com/Dirtybird99/Dirtybird-Go-GPU-Miner/internal/astrobwt"
)

// Pipeline computes DERO AstroBWTv3 with GPU suffix sorting. The portable path
// uses CPU front/final stages around a WGSL SA. The WGSL fused path moves the
// front half to the GPU. The NVIDIA/Vulkan stream instead builds Wolf data in
// Go, then runs native SPIR-V SA + final SHA and reads back only digests. Every
// path is byte-for-byte equal to astrobwt.Sum.
type Pipeline struct {
	ctx *Ctx
}

func NewPipeline(ctx *Ctx) *Pipeline { return &Pipeline{ctx: ctx} }

// HashBatch hashes many inputs, building all suffix arrays in one batched GPU
// dispatch. Front and back halves run on the CPU oracle. This is the portable
// fallback path.
func (p *Pipeline) HashBatch(inputs [][]byte) ([][32]byte, error) {
	datas := make([][]byte, len(inputs))
	dls := make([]int, len(inputs))
	// Front half is embarrassingly parallel across nonces — spread it over all
	// cores (it's ~35% of batch time when run single-threaded).
	parallelFor(len(inputs), func(i int) {
		d, dl := astrobwt.OracleStream(inputs[i])
		datas[i] = d
		dls[i] = int(dl)
	})
	sas, err := p.ctx.SuffixArrayBatch(datas)
	if err != nil {
		return nil, err
	}
	out := make([][32]byte, len(inputs))
	parallelFor(len(inputs), func(i int) {
		out[i] = finalHash(sas[i], dls[i])
	})
	return out, nil
}

// parallelFor runs fn(0..n-1) across GOMAXPROCS workers.
func parallelFor(n int, fn func(i int)) {
	workers := runtime.GOMAXPROCS(0)
	if workers > n {
		workers = n
	}
	if workers <= 1 {
		for i := 0; i < n; i++ {
			fn(i)
		}
		return
	}
	var wg sync.WaitGroup
	wg.Add(workers)
	idx := make(chan int, n)
	for i := 0; i < n; i++ {
		idx <- i
	}
	close(idx)
	for w := 0; w < workers; w++ {
		go func() {
			defer wg.Done()
			for i := range idx {
				fn(i)
			}
		}()
	}
	wg.Wait()
}

// HashBatchFused computes the batch entirely on the GPU — front half AND
// suffix array chained in one submission with no data-stream readback — then
// the final SHA on the CPU. This is the fully-GPU throughput path; the CPU is
// left only the final ~9% SHA over the read-back SA. Byte-identical to
// astrobwt.Sum.
func (p *Pipeline) HashBatchFused(inputs [][]byte) ([][32]byte, error) {
	if len(inputs) == 0 {
		return nil, nil
	}
	c := p.ctx
	if !c.fused.fits(c, len(inputs)) {
		if c.fused != nil {
			c.fused.release()
			c.fused = nil
		}
		fs, err := c.buildFusedSession(len(inputs))
		if err != nil {
			return nil, err
		}
		c.fused = fs
	}
	sas, lens, err := c.fused.run(inputs)
	if err != nil {
		return nil, err
	}
	return finalHashBatch(sas, lens), nil
}

// finalHashBatch runs the final SHA over each hash's SA across all cores.
func finalHashBatch(sas [][]int32, lens []uint32) [][32]byte {
	out := make([][32]byte, len(sas))
	parallelFor(len(sas), func(i int) {
		out[i] = finalHash(sas[i], int(lens[i]))
	})
	return out
}

// HashBatchFusedStream hashes a stream of batches through the selected fast
// path. WGSL ping-pongs two sessions so batch k+1 GPU compute overlaps batch
// k readback + CPU SHA. Native SPIR-V groups four 512-row sorts under one
// 2048-row GPU SHA while Go prepares the next Wolf buffer. Returns one digest
// slice per input batch and is byte-identical to the CPU oracle.
func (p *Pipeline) HashBatchFusedStream(batches [][][]byte) ([][][32]byte, error) {
	c := p.ctx
	if len(batches) == 0 {
		return nil, nil
	}
	maxN := 0
	for i, b := range batches {
		if len(b) == 0 {
			return nil, fmt.Errorf("fused stream batch %d is empty", i)
		}
		if len(b) > maxN {
			maxN = len(b)
		}
	}
	if c.useSPIRV {
		streamer, err := p.NewFusedStreamer(maxN)
		if err != nil {
			return nil, err
		}
		out := make([][][32]byte, len(batches))
		for i, batch := range batches {
			got, err := streamer.Submit(batch)
			if err != nil {
				return nil, err
			}
			if i > 0 {
				out[i-1] = got
			}
		}
		last, err := streamer.Drain()
		if err != nil {
			return nil, err
		}
		out[len(out)-1] = last
		return out, nil
	}
	for i := range c.fusedPipe {
		if !c.fusedPipe[i].fits(c, maxN) {
			if c.fusedPipe[i] != nil {
				c.fusedPipe[i].release()
				c.fusedPipe[i] = nil
			}
			fs, err := c.buildFusedSession(maxN)
			if err != nil {
				return nil, err
			}
			c.fusedPipe[i] = fs
		}
	}

	// Strict ping-pong (no skipped indices) keeps the invariant that a session
	// is reused only two batches later, after its prior occupant was collected.
	out := make([][][32]byte, len(batches))
	prev := -1 // index of the batch whose compute is in flight, awaiting collect
	for i, b := range batches {
		if err := c.fusedPipe[i%2].submitCompute(b); err != nil {
			return nil, err
		}
		if prev >= 0 {
			sas, lens, err := c.fusedPipe[prev%2].collect()
			if err != nil {
				return nil, err
			}
			out[prev] = finalHashBatch(sas, lens) // overlaps batch i's GPU compute
		}
		prev = i
	}
	if prev >= 0 {
		sas, lens, err := c.fusedPipe[prev%2].collect()
		if err != nil {
			return nil, err
		}
		out[prev] = finalHashBatch(sas, lens)
	}
	return out, nil
}

// FusedStreamer drives the two ping-ponging fused sessions for a CONTINUOUS
// stream of batches (the live miner). Unlike HashBatchFusedStream (which takes
// a fixed batch list), it exposes a stateful step: Submit issues the current
// batch's GPU compute and returns the PREVIOUS batch's finished hashes, so the
// caller's CPU work on those (verify + submit) overlaps the current compute.
type FusedStreamer struct {
	c    *Ctx
	cur  int  // session index for the NEXT Submit
	have bool // a batch is in flight awaiting collect
	fast *spirvFastStreamer
}

// NewFusedStreamer sizes both pipeline sessions for up to maxBatch hashes.
// Requires the subgroup fused path (caller must have verified subgroup size).
func (p *Pipeline) NewFusedStreamer(maxBatch int) (*FusedStreamer, error) {
	c := p.ctx
	// SelfTest uses the one-shot session. The live streamer needs two separate
	// sessions, so release it before allocating another two fixed 512-row slabs.
	if c.fused != nil {
		c.fused.release()
		c.fused = nil
	}
	if c.useSPIRV {
		if maxBatch > spirvStreamRows {
			return nil, fmt.Errorf("SPIR-V streamer supports at most %d hashes, got %d", spirvStreamRows, maxBatch)
		}
		for i := range c.fusedPipe {
			if c.fusedPipe[i] != nil {
				c.fusedPipe[i].release()
				c.fusedPipe[i] = nil
			}
		}
		if c.spirvFast == nil {
			var err error
			c.spirvFast, err = buildSPIRVFastStreamer(c)
			if err != nil {
				return nil, err
			}
		}
		if c.spirvFast.have {
			return nil, fmt.Errorf("SPIR-V streamer already has a batch in flight")
		}
		return &FusedStreamer{c: c, fast: c.spirvFast}, nil
	}
	if c.spirvFast != nil {
		c.spirvFast.release()
		c.spirvFast = nil
	}
	for i := range c.fusedPipe {
		if !c.fusedPipe[i].fits(c, maxBatch) {
			if c.fusedPipe[i] != nil {
				c.fusedPipe[i].release()
				c.fusedPipe[i] = nil
			}
			fs, err := c.buildFusedSession(maxBatch)
			if err != nil {
				return nil, err
			}
			c.fusedPipe[i] = fs
		}
	}
	return &FusedStreamer{c: c}, nil
}

// Submit issues GPU compute for `inputs` and returns the final hashes of the
// PREVIOUSLY submitted batch (nil on the first call). The returned batch's
// compute is already done; its CPU post-processing by the caller overlaps
// `inputs`' GPU compute. `inputs` is copied into GPU buffers before return, so
// the caller may reuse the slice after Submit — but must keep the buffers
// behind the PREVIOUS Submit's inputs alive until it finishes with the
// returned hashes (the caller double-buffers its own per-batch state).
func (fs *FusedStreamer) Submit(inputs [][]byte) ([][32]byte, error) {
	if len(inputs) == 0 {
		return nil, fmt.Errorf("fused streamer batch is empty")
	}
	if fs.fast != nil {
		return fs.fast.Submit(inputs)
	}
	if err := fs.c.fusedPipe[fs.cur].submitCompute(inputs); err != nil {
		return nil, err
	}
	var out [][32]byte
	if fs.have {
		sas, lens, err := fs.c.fusedPipe[1-fs.cur].collect()
		if err != nil {
			return nil, err
		}
		out = finalHashBatch(sas, lens)
	}
	fs.have = true
	fs.cur = 1 - fs.cur
	return out, nil
}

// Drain returns the last in-flight batch's hashes (nil if none), completing
// the pipeline. Call once after the last Submit.
func (fs *FusedStreamer) Drain() ([][32]byte, error) {
	if fs.fast != nil {
		return fs.fast.Drain()
	}
	if !fs.have {
		return nil, nil
	}
	sas, lens, err := fs.c.fusedPipe[1-fs.cur].collect()
	if err != nil {
		return nil, err
	}
	fs.have = false
	return finalHashBatch(sas, lens), nil
}

// shaBufPool recycles the per-hash SHA input buffers. The streaming miner
// calls finalHash 256 times per batch at ~3 batches/s; fresh ~142KB buffers
// each call put >100MB/s of garbage on the heap, and the resulting GC pauses
// stall the CPU side of the double-buffered stream (the GPU goes idle).
var shaBufPool = sync.Pool{New: func() interface{} { b := make([]byte, saStride*4); return &b }}

// finalHash computes SHA-256 over the SA as little-endian int32 bytes (pow.go:28).
func finalHash(sa []int32, dl int) [32]byte {
	bp := shaBufPool.Get().(*[]byte)
	buf := (*bp)[:dl*4]
	for i := 0; i < dl; i++ {
		binary.LittleEndian.PutUint32(buf[i*4:], uint32(sa[i]))
	}
	sum := sha256.Sum256(buf)
	shaBufPool.Put(bp)
	return sum
}

// KAT is the consensus known-answer test: AstroBWTv3("a").
var KATInput = []byte("a")
var KATHash = [32]byte{
	0x54, 0xe2, 0x32, 0x4d, 0xda, 0xcc, 0x3f, 0x03, 0x83, 0x50, 0x1a, 0x9e, 0x57, 0x60, 0xf8, 0x5d,
	0x63, 0xe9, 0xbc, 0x67, 0x05, 0xe9, 0x12, 0x4c, 0xa7, 0xae, 0xf8, 0x90, 0x16, 0xab, 0x81, 0xea,
}

// SelfTest verifies the pipeline THE MINER WILL ACTUALLY RUN: the path is
// derived from the context exactly as main derives its hash function. SPIR-V
// exercises the native Vulkan sorter, refine exercises sa_refine.wgsl, and the
// portable path exercises the CPU front half + sa_gpu.wgsl. Two checks:
//
//  1. The consensus KAT pow("a") through the full selected path.
//  2. The adversarial SA corpus (sa_vectors.go) through the active SA kernel —
//     sentinel-vs-zero, oversize buckets, n%4 padding: the corners where a
//     per-backend naga miscompile would hide, which a single tiny KAT input
//     cannot reach.
//
// A failed check returns (false, err) with err naming what mismatched. This
// gate is an optimization (don't mine 100% waste on a miscomputing device),
// not a consensus guard — the per-share CPU re-hash in processBatch is what
// makes a bad submission impossible — so callers may offer an override.
func (p *Pipeline) SelfTest() (bool, error) {
	hash := p.HashBatch
	if p.ctx.useRefine || p.ctx.useSPIRV {
		hash = p.HashBatchFused
	}

	// Consensus KAT through the selected path, in a small padded batch (the
	// kernels are batch-shaped; index 0 is the KAT input).
	inputs := [][]byte{KATInput}
	for i := 1; i < 8; i++ {
		inputs = append(inputs, []byte("padding"))
	}
	got, err := hash(inputs)
	if err != nil {
		return false, fmt.Errorf("KAT dispatch: %w", err)
	}
	if got[0] != KATHash {
		return false, fmt.Errorf("KAT mismatch: pow(%q) = %x, want %x", KATInput, got[0], KATHash)
	}

	// Adversarial SA corpus in one dispatch through the active SA kernel. Slow
	// cases are skipped: their O(n^3) reference is a test-suite cost, not a
	// startup cost.
	var datas [][]byte
	var names []string
	for _, tc := range SyntheticSACases() {
		if tc.Slow {
			continue
		}
		datas = append(datas, tc.Data)
		names = append(names, tc.Name)
	}
	var sas [][]int32
	if p.ctx.useSPIRV {
		sas, err = p.ctx.fused.runSPIRVData(datas)
	} else {
		sas, err = p.ctx.SuffixArrayBatch(datas)
	}
	if err != nil {
		return false, fmt.Errorf("synthetic SA dispatch: %w", err)
	}
	for i := range datas {
		if !equalI32(sas[i], saisRef(datas[i])) {
			return false, fmt.Errorf("synthetic SA case %q (n=%d) mismatched the reference", names[i], len(datas[i]))
		}
	}

	// The native mining stream deliberately repartitions the one-shot stages:
	// Go builds Wolf rows and SPIR-V performs final SHA over a grouped archive.
	// Exercise that exact composition at startup as well as its constituent SA
	// kernel above. A live degradation re-probe can happen while the current
	// stream group is in flight, in which case the constituent KAT remains the
	// safe non-invasive check.
	if p.ctx.useSPIRV && (p.ctx.spirvFast == nil || !p.ctx.spirvFast.have) {
		batches, err := p.HashBatchFusedStream([][][]byte{inputs})
		if err != nil {
			return false, fmt.Errorf("native stream KAT dispatch: %w", err)
		}
		if len(batches) != 1 || len(batches[0]) != len(inputs) || batches[0][0] != KATHash {
			return false, fmt.Errorf("native stream KAT mismatch: pow(%q) was not reproduced", KATInput)
		}
	}
	return true, nil
}
