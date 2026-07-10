package gpu

import (
	"math/rand"
	"testing"

	"github.com/Dirtybird99/Dirtybird-Go-GPU-Miner/internal/astrobwt"
)

// TestFusedStreamVsOracle verifies the double-buffered streaming path
// (HashBatchFusedStream) is byte-identical to astrobwt.Sum across MULTIPLE
// batches — exercising the ping-pong session reuse, the non-draining collect,
// and the zero-copy readback that stays mapped across the CPU SHA.
func TestFusedStreamVsOracle(t *testing.T) {
	c := testCtx(t)
	p := NewPipeline(c)
	rng := rand.New(rand.NewSource(4070))
	const (
		batch    = 256
		nBatches = 5 // odd count so the drain path (final collect) is exercised
	)
	batches := make([][][]byte, nBatches)
	for b := range batches {
		inputs := make([][]byte, batch)
		for i := range inputs {
			blob := make([]byte, 76)
			rng.Read(blob)
			inputs[i] = blob
		}
		batches[b] = inputs
	}

	got, err := p.HashBatchFusedStream(batches)
	if err != nil {
		t.Fatalf("HashBatchFusedStream: %v", err)
	}
	if len(got) != nBatches {
		t.Fatalf("got %d batches, want %d", len(got), nBatches)
	}
	bad := 0
	for b := range batches {
		if len(got[b]) != batch {
			t.Fatalf("batch %d: got %d hashes, want %d", b, len(got[b]), batch)
		}
		for i, in := range batches[b] {
			if want := astrobwt.Sum(in); got[b][i] != want {
				bad++
				if bad <= 4 {
					t.Errorf("batch %d hash %d:\n got  %x\n want %x", b, i, got[b][i], want)
				}
			}
		}
	}
	if bad != 0 {
		t.Fatalf("%d/%d streamed hashes wrong", bad, nBatches*batch)
	}
}

// TestFusedStreamer512VsOracle drives the FusedStreamer at the LIVE default
// batch size (512), which no other test covers — the fused/streaming tests all
// run at 256, and the big-batch test drives the non-fused SuffixArrayBatch path.
// The fused session at 512 has plumbing 256 cannot exercise: ~139 MiB SA/staging
// buffers, full-row readback copies, and 512-workgroup dispatches. It runs on its
// OWN device (two ping-pong sessions are ~2.3 GB and would collide with the
// shared test context's sessions — the BUG-VK-001 map-pressure pattern) and is
// skipped in -short. Requires the subgroup fused path.
func TestFusedStreamer512VsOracle(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping 512-batch fused streamer test in -short")
	}
	c, err := NewCtx(PowerHigh)
	if err != nil {
		t.Skipf("no GPU device: %v", err)
	}
	defer c.Close()
	if c.SubgroupSize() < 32 {
		t.Skip("fused path needs subgroup size >= 32")
	}
	c.UseRefine(true)
	p := NewPipeline(c)
	rng := rand.New(rand.NewSource(512512))
	const (
		batch    = 512
		nBatches = 4
	)
	allInputs := make([][][]byte, nBatches)
	for b := range allInputs {
		in := make([][]byte, batch)
		for i := range in {
			blob := make([]byte, 76)
			rng.Read(blob)
			in[i] = blob
		}
		allInputs[b] = in
	}
	streamer, err := p.NewFusedStreamer(batch)
	if err != nil {
		t.Fatalf("NewFusedStreamer(512): %v", err)
	}
	got := make([][][32]byte, nBatches)
	prevIdx := -1
	for b := 0; b < nBatches; b++ {
		res, err := streamer.Submit(allInputs[b])
		if err != nil {
			t.Fatalf("Submit %d: %v", b, err)
		}
		if b > 0 {
			got[prevIdx] = res
		}
		prevIdx = b
	}
	res, err := streamer.Drain()
	if err != nil {
		t.Fatalf("Drain: %v", err)
	}
	got[prevIdx] = res
	bad := 0
	for b := range allInputs {
		for i, in := range allInputs[b] {
			if want := astrobwt.Sum(in); got[b][i] != want {
				bad++
			}
		}
	}
	if bad != 0 {
		t.Fatalf("%d/%d streamed hashes wrong at batch 512", bad, nBatches*batch)
	}
}

// TestFusedStreamerVsOracle drives the stateful FusedStreamer exactly as the
// live miner's mineLoopStream does — Submit returns the PREVIOUS batch, then a
// final Drain — with the caller double-buffering its own per-batch inputs. It
// checks that results are associated with the correct batch (no off-by-one in
// the ping-pong) and are byte-identical to astrobwt.Sum.
func TestFusedStreamerVsOracle(t *testing.T) {
	c := testCtx(t)
	p := NewPipeline(c)
	rng := rand.New(rand.NewSource(99))
	const (
		batch    = 256
		nBatches = 6
	)
	// Distinct, reproducible inputs per batch so a misassociated result is caught.
	mkBatch := func() [][]byte {
		in := make([][]byte, batch)
		for i := range in {
			blob := make([]byte, 76)
			rng.Read(blob)
			in[i] = blob
		}
		return in
	}
	allInputs := make([][][]byte, nBatches)
	for b := range allInputs {
		allInputs[b] = mkBatch()
	}

	streamer, err := p.NewFusedStreamer(batch)
	if err != nil {
		t.Fatalf("NewFusedStreamer: %v", err)
	}

	got := make([][][32]byte, nBatches)
	// Two input buffers, mirroring mineLoopStream's blob double-buffering: the
	// previous batch's inputs must stay alive until its results come back.
	prevIdx := -1
	for b := 0; b < nBatches; b++ {
		res, err := streamer.Submit(allInputs[b])
		if err != nil {
			t.Fatalf("Submit %d: %v", b, err)
		}
		if b == 0 {
			if res != nil {
				t.Fatalf("first Submit returned results, want nil")
			}
		} else {
			got[prevIdx] = res
		}
		prevIdx = b
	}
	res, err := streamer.Drain()
	if err != nil {
		t.Fatalf("Drain: %v", err)
	}
	got[prevIdx] = res

	bad := 0
	for b := range allInputs {
		if len(got[b]) != batch {
			t.Fatalf("batch %d: got %d hashes, want %d", b, len(got[b]), batch)
		}
		for i, in := range allInputs[b] {
			if want := astrobwt.Sum(in); got[b][i] != want {
				bad++
				if bad <= 4 {
					t.Errorf("batch %d hash %d:\n got  %x\n want %x", b, i, got[b][i], want)
				}
			}
		}
	}
	if bad != 0 {
		t.Fatalf("%d/%d streamer hashes wrong", bad, nBatches*batch)
	}
}
