package gpu

import (
	"math/rand"
	"testing"

	"github.com/Dirtybird99/Dirtybird-Go-GPU-Miner/internal/astrobwt"
)

// TestSARefineBatch256 is the optimize-loop correctness gate at the live batch
// size: 256 real AstroBWTv3 buffers built in one dispatch by the refine kernel,
// every SA byte-identical to the sais oracle. Encodes the batch-256-vs-live
// discipline — a gate that tested smaller than production once hid a
// 40%-corruption bug.
func TestSARefineBatch256(t *testing.T) {
	c := testCtx(t)
	c.UseRefine(true)
	defer c.UseRefine(false)
	rng := rand.New(rand.NewSource(256256))
	const batch = 256
	datas := make([][]byte, batch)
	oracles := make([][]int32, batch)
	for i := 0; i < batch; i++ {
		blob := make([]byte, 76)
		rng.Read(blob)
		data, sa, dl := astrobwt.OracleSA(blob)
		d := make([]byte, dl)
		copy(d, data[:dl])
		datas[i] = d
		oracles[i] = sa
	}
	got, err := c.SuffixArrayBatch(datas)
	if err != nil {
		t.Fatalf("SuffixArrayBatch(refine, 256): %v", err)
	}
	bad := 0
	for i := range datas {
		if !equalI32(got[i], oracles[i]) {
			bad++
			if bad <= 3 {
				first := -1
				for k := range got[i] {
					if k >= len(oracles[i]) || got[i][k] != oracles[i][k] {
						first = k
						break
					}
				}
				t.Errorf("hash %d (n=%d): SA mismatch at %d", i, len(datas[i]), first)
			}
		}
	}
	if bad != 0 {
		t.Fatalf("%d/%d SAs wrong at batch 256", bad, batch)
	}
}
