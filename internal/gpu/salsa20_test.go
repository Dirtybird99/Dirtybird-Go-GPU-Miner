package gpu

import (
	"crypto/sha256"
	"encoding/binary"
	"math/rand"
	"testing"

	"github.com/Dirtybird99/Dirtybird-Go-GPU-Miner/internal/astrobwt"
)

// keyWords packs a 32-byte key into 8 little-endian u32 words (Salsa20 layout).
func keyWords(key [32]byte) []uint32 {
	w := make([]uint32, 8)
	for i := range w {
		w[i] = binary.LittleEndian.Uint32(key[i*4:])
	}
	return w
}

// salsaBatch runs the WGSL Salsa20 kernel over keys, returning 256 keystream
// bytes each.
func salsaBatch(t *testing.T, c *Ctx, keys [][32]byte) [][256]byte {
	t.Helper()

	kw := make([]uint32, 0, len(keys)*8)
	for _, k := range keys {
		kw = append(kw, keyWords(k)...)
	}
	n := uint32(len(keys))
	out, err := c.Run(KernelRun{
		WGSL:    Salsa20WGSL,
		Entry:   "salsa20_main",
		Inputs:  [][]byte{U32s(kw...)},
		Outputs: []int{len(keys) * 256 * 4},
		Uniform: U32s(n),
		Groups:  [3]uint32{(n + 63) / 64, 1, 1},
	})
	if err != nil {
		t.Fatalf("Run: %v", err)
	}

	raw := u32sAsBytes(out[0])
	res := make([][256]byte, len(keys))
	for i := range keys {
		copy(res[i][:], raw[i*256:(i+1)*256])
	}
	return res
}

// TestSalsa20Prologue checks the GPU Salsa20 keystream against the AstroBWTv3
// oracle's post-salsa state, over a live-size batch of real derived keys.
func TestSalsa20Prologue(t *testing.T) {
	c := testCtx(t)

	const batch = 4096
	rng := rand.New(rand.NewSource(20))
	keys := make([][32]byte, batch)
	oracle := make([][256]byte, batch)
	for i := range keys {
		blob := make([]byte, 76) // DERO-ish work blob size
		rng.Read(blob)
		// keys[i] is what the front-half actually feeds Salsa: sha256(blob).
		keys[i] = sha256.Sum256(blob)
		_, afterSalsa, _, _ := astrobwt.OraclePrologue(blob)
		oracle[i] = afterSalsa
	}

	got := salsaBatch(t, c, keys)
	bad := 0
	for i := range keys {
		if got[i] != oracle[i] {
			if bad < 3 {
				t.Errorf("key %d:\n got  %x\n want %x", i, got[i][:32], oracle[i][:32])
			}
			bad++
		}
	}
	if bad != 0 {
		t.Fatalf("%d/%d salsa keystreams wrong at batch %d", bad, batch, batch)
	}
}
