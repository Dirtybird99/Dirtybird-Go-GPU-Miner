package gpu

import (
	"math/rand"
	"testing"

	"github.com/Dirtybird99/Dirtybird-Go-GPU-Miner/internal/astrobwt"
)

// TestFusedVsOracle verifies the fully-GPU path (front half + SA fused, no
// data-stream readback) produces AstroBWTv3 hashes byte-identical to
// astrobwt.Sum over a live-size batch.
func TestFusedVsOracle(t *testing.T) {
	c := testCtx(t)
	p := NewPipeline(c)
	rng := rand.New(rand.NewSource(4070))
	const batch = 256
	inputs := make([][]byte, batch)
	for i := range inputs {
		b := make([]byte, 76)
		rng.Read(b)
		inputs[i] = b
	}
	got, err := p.HashBatchFused(inputs)
	if err != nil {
		t.Fatalf("HashBatchFused: %v", err)
	}
	bad := 0
	for i, in := range inputs {
		if want := astrobwt.Sum(in); got[i] != want {
			bad++
			if bad <= 4 {
				t.Errorf("hash %d:\n got  %x\n want %x", i, got[i], want)
			}
		}
	}
	if bad != 0 {
		t.Fatalf("%d/%d fused hashes wrong", bad, batch)
	}
}

// TestFusedKAT verifies the fused GPU path reproduces the consensus KAT.
func TestFusedKAT(t *testing.T) {
	c := testCtx(t)
	p := NewPipeline(c)
	// pad the batch; hash index 0 is the KAT input "a".
	inputs := [][]byte{[]byte("a")}
	for i := 1; i < 8; i++ {
		inputs = append(inputs, []byte("padding"))
	}
	got, err := p.HashBatchFused(inputs)
	if err != nil {
		t.Fatalf("HashBatchFused: %v", err)
	}
	if got[0] != KATHash {
		t.Fatalf("KAT: got %x want %x", got[0], KATHash)
	}
}
