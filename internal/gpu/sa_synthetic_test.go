package gpu

import (
	"fmt"
	"math/rand"
	"testing"
)

// TestSARefineSynthetic exercises the adversarial inputs for the coarse depth-4
// sort that random real buffers never produce: suffixes whose 4-byte window runs
// off the end of the text (where the implicit sentinel must sort below every
// byte, including a literal 0x00), buckets far wider than a workgroup, and texts
// whose length is not a multiple of the 4-byte packing word.
func TestSARefineSynthetic(t *testing.T) {
	c := testCtx(t)
	c.UseRefine(true)
	defer c.UseRefine(false)

	// The corpus lives in sa_vectors.go, shared with the runtime startup gate
	// (Pipeline.SelfTest) so the gate checks exactly these corners.
	for _, tc := range SyntheticSACases() {
		t.Run(tc.Name, func(t *testing.T) {
			if tc.Slow && testing.Short() {
				t.Skip("slow O(n^3) reference case skipped in -short")
			}
			got, err := c.SuffixArrayBatch([][]byte{tc.Data})
			if err != nil {
				t.Fatalf("SuffixArrayBatch: %v", err)
			}
			if want := saisRef(tc.Data); !equalI32(got[0], want) {
				first := -1
				for k := range got[0] {
					if k >= len(want) || got[0][k] != want[k] {
						first = k
						break
					}
				}
				t.Fatalf("n=%d: SA mismatch at slot %d", len(tc.Data), first)
			}
		})
	}
}

// TestSARefineBigBatch guards the dispatch-sizing bug class that silently
// dropped work in the CUDA miner above batch ~1536: a per-hash workgroup count
// that overflows a grid limit corrupts SAs while a batch-256 gate stays green.
//
// It runs on its OWN device rather than the shared test Ctx: SuffixArrayBatch
// maps its whole readback staging buffer (slab x 283 KB rows, whatever the text
// length) in one call, so on the shared device the map fails (BUG-VK-001)
// whenever the fused tests' sessions already hold host memory — a resource
// collision that has nothing to do with the SA and would make this test flaky.
func TestSARefineBigBatch(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping in -short")
	}
	c, err := NewCtx(PowerHigh)
	if err != nil {
		t.Skipf("no GPU device: %v", err)
	}
	defer c.Close()
	c.UseRefine(true)
	rng := rand.New(rand.NewSource(940940))
	for _, batch := range []int{512, 940} {
		t.Run(fmt.Sprintf("batch%d", batch), func(t *testing.T) {
			datas := make([][]byte, batch)
			for i := range datas {
				// Short synthetic texts keep VRAM sane while still filling the
				// dispatch: correctness here is about workgroup indexing.
				n := 600 + rng.Intn(400)
				d := make([]byte, n)
				for j := range d {
					d[j] = byte(rng.Intn(4)) // small alphabet => deep ties
				}
				datas[i] = d
			}
			got, err := c.SuffixArrayBatch(datas)
			if err != nil {
				t.Fatalf("SuffixArrayBatch(%d): %v", batch, err)
			}
			bad := 0
			for i := range datas {
				if !equalI32(got[i], saisRef(datas[i])) {
					bad++
				}
			}
			if bad != 0 {
				t.Fatalf("%d/%d SAs wrong at batch %d", bad, batch, batch)
			}
		})
	}
}
