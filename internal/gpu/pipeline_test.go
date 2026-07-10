package gpu

import (
	"math/rand"
	"testing"

	"github.com/Dirtybird99/Dirtybird-Go-GPU-Miner/internal/astrobwt"
)

// TestPipelineKAT verifies the end-to-end GPU-assisted hash reproduces the
// AstroBWTv3 consensus known-answer test.
func TestPipelineKAT(t *testing.T) {
	c := testCtx(t)
	p := NewPipeline(c)
	ok, err := p.SelfTest()
	if err != nil || !ok {
		t.Fatalf("SelfTest: ok=%v err=%v", ok, err) // err names the failing check
	}
}

// TestPipelineVsOracle checks the composed pipeline hash matches astrobwt.Sum
// byte-for-byte over random block-sized inputs, one blob per dispatch (the
// single-hash shape, distinct from the batch-32 test below).
func TestPipelineVsOracle(t *testing.T) {
	c := testCtx(t)
	p := NewPipeline(c)
	rng := rand.New(rand.NewSource(76))
	for i := 0; i < 8; i++ {
		blob := make([]byte, 76)
		rng.Read(blob)
		got, err := p.HashBatch([][]byte{blob})
		if err != nil {
			t.Fatalf("hash %d: %v", i, err)
		}
		want := astrobwt.Sum(blob)
		if got[0] != want {
			t.Fatalf("hash %d mismatch:\n got  %x\n want %x", i, got[0], want)
		}
	}
}

// TestPipelineBatchVsOracle checks the batched throughput path matches
// astrobwt.Sum for every input in the batch.
func TestPipelineBatchVsOracle(t *testing.T) {
	c := testCtx(t)
	p := NewPipeline(c)
	rng := rand.New(rand.NewSource(77))
	inputs := make([][]byte, 32)
	for i := range inputs {
		blob := make([]byte, 76)
		rng.Read(blob)
		inputs[i] = blob
	}
	got, err := p.HashBatch(inputs)
	if err != nil {
		t.Fatalf("HashBatch: %v", err)
	}
	for i, in := range inputs {
		if want := astrobwt.Sum(in); got[i] != want {
			t.Fatalf("batch hash %d mismatch:\n got  %x\n want %x", i, got[i], want)
		}
	}
}
