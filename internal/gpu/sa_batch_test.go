package gpu

import (
	"math/rand"
	"testing"

	"github.com/Dirtybird99/Dirtybird-Go-GPU-Miner/internal/astrobwt"
)

// TestSuffixArrayBatchSmall checks the fully-GPU-resident batched SA on small
// mixed strings (including LCP pathologies) in one dispatch.
func TestSuffixArrayBatchSmall(t *testing.T) {
	c := testCtx(t)
	datas := [][]byte{
		[]byte("banana"),
		[]byte("mississippi"),
		[]byte("aaaaaaaa"),
		[]byte("abababababab"),
		[]byte("the quick brown fox"),
		{0, 1, 0, 1, 2, 0, 1},
	}
	got, err := c.SuffixArrayBatch(datas)
	if err != nil {
		t.Fatalf("SuffixArrayBatch: %v", err)
	}
	for i, d := range datas {
		want := saisRef(d)
		if !equalI32(got[i], want) {
			t.Fatalf("%q:\n got  %v\n want %v", d, got[i], want)
		}
	}
}

// TestSuffixArrayBatchLive builds a batch of real AstroBWTv3 buffers in ONE
// dispatch and checks every SA byte-for-byte against the sais oracle.
func TestSuffixArrayBatchLive(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping live-buffer batched SA test in -short")
	}
	c := testCtx(t)
	rng := rand.New(rand.NewSource(4070))
	const batch = 16
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
		t.Fatalf("SuffixArrayBatch: %v", err)
	}
	for i := range datas {
		if !equalI32(got[i], oracles[i]) {
			first := -1
			for k := range got[i] {
				if k >= len(oracles[i]) || got[i][k] != oracles[i][k] {
					first = k
					break
				}
			}
			t.Fatalf("hash %d (n=%d): SA mismatch at %d (got %d want %d)",
				i, len(datas[i]), first, got[i][first], oracles[i][first])
		}
	}
}
