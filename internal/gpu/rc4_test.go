package gpu

import (
	"crypto/sha256"
	"math/rand"
	"testing"

	"github.com/Dirtybird99/Dirtybird-Go-GPU-Miner/internal/astrobwt"
)

// TestRC4Prologue checks the GPU RC4 stage against the oracle's post-rc4
// state. Input to the kernel is the post-salsa state (which is both the RC4
// key and the plaintext, matching astrobwt's in-place XORKeyStream).
func TestRC4Prologue(t *testing.T) {
	c := testCtx(t)

	const batch = 4096
	rng := rand.New(rand.NewSource(44))
	in := make([]byte, 0, batch*256*4)
	oracle := make([][256]byte, batch)
	for i := 0; i < batch; i++ {
		blob := make([]byte, 76)
		rng.Read(blob)
		_ = sha256.Sum256(blob)
		_, afterSalsa, afterRC4, _ := astrobwt.OraclePrologue(blob)
		in = append(in, bytesAsU32(afterSalsa[:])...)
		oracle[i] = afterRC4
	}

	out, err := c.Run(KernelRun{
		WGSL:    RC4WGSL,
		Entry:   "rc4_main",
		Inputs:  [][]byte{in},
		Outputs: []int{batch * 256 * 4},
		Uniform: U32s(batch),
		Groups:  [3]uint32{(batch + 63) / 64, 1, 1},
	})
	if err != nil {
		t.Fatalf("Run: %v", err)
	}

	raw := u32sAsBytes(out[0])
	bad := 0
	for i := 0; i < batch; i++ {
		var got [256]byte
		copy(got[:], raw[i*256:(i+1)*256])
		if got != oracle[i] {
			if bad < 3 {
				t.Errorf("hash %d:\n got  %x\n want %x", i, got[:32], oracle[i][:32])
			}
			bad++
		}
	}
	if bad != 0 {
		t.Fatalf("%d/%d rc4 outputs wrong at batch %d", bad, batch, batch)
	}
}
