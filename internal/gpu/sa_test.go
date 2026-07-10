package gpu

import (
	"math/rand"
	"testing"

	"github.com/Dirtybird99/Dirtybird-Go-GPU-Miner/internal/astrobwt"
)

// TestSuffixArraySmall checks the GPU prefix-doubling SA against sais on small
// hand-picked strings, including the repetition pathologies AstroBWTv3 data
// hits (long LCPs).
func TestSuffixArraySmall(t *testing.T) {
	c := testCtx(t)
	cases := [][]byte{
		[]byte("banana"),
		[]byte("mississippi"),
		[]byte("aaaaaaaa"),          // maximal-LCP stress
		[]byte("abababababab"),      // period-2
		[]byte{0, 0, 1, 0, 0, 1, 0}, // zeros + sentinel-ish
		[]byte("the quick brown fox jumps over the lazy dog"),
	}
	for _, text := range cases {
		got, err := c.SuffixArrayBatch([][]byte{text})
		if err != nil {
			t.Fatalf("%q: %v", text, err)
		}
		want := saisRef(text)
		if !equalI32(got[0], want) {
			t.Fatalf("%q:\n got  %v\n want %v", text, got[0], want)
		}
	}
}

// TestSuffixArrayLive builds SAs for real AstroBWTv3 buffers and checks them
// byte-for-byte against the sais oracle used by the CPU miner.
func TestSuffixArrayLive(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping live-buffer SA test in -short")
	}
	c := testCtx(t)
	rng := rand.New(rand.NewSource(4070))
	for h := 0; h < 4; h++ {
		blob := make([]byte, 76)
		rng.Read(blob)
		data, oracleSA, dl := astrobwt.OracleSA(blob)
		gotB, err := c.SuffixArrayBatch([][]byte{data[:dl]})
		if err != nil {
			t.Fatalf("hash %d: %v", h, err)
		}
		got := gotB[0]
		if !equalI32(got, oracleSA) {
			// find first divergence for a useful message
			first := -1
			for i := range got {
				if i >= len(oracleSA) || got[i] != oracleSA[i] {
					first = i
					break
				}
			}
			t.Fatalf("hash %d (n=%d): SA mismatch at index %d (got %d want %d)",
				h, dl, first, got[first], oracleSA[first])
		}
	}
}

// saisRef and equalI32 live in sa_vectors.go (compiled, not test-only), shared
// with the runtime startup gate.
