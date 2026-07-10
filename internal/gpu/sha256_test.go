package gpu

import (
	"bytes"
	"crypto/sha256"
	"encoding/binary"
	"math/rand"
	"testing"
)

// bytesAsU32 expands each byte to its own little-endian u32 word, the layout
// the WGSL kernels use for byte arrays (WGSL has no u8 type).
func bytesAsU32(b []byte) []byte {
	out := make([]byte, len(b)*4)
	for i, v := range b {
		binary.LittleEndian.PutUint32(out[i*4:], uint32(v))
	}
	return out
}

// u32sAsBytes takes the low byte of each u32 word.
func u32sAsBytes(b []byte) []byte {
	out := make([]byte, len(b)/4)
	for i := range out {
		out[i] = byte(binary.LittleEndian.Uint32(b[i*4:]))
	}
	return out
}

// shaBatch runs the WGSL SHA-256 kernel over msgs, returning one digest each.
func shaBatch(t *testing.T, c *Ctx, msgs [][]byte) [][32]byte {
	t.Helper()

	var flat []byte
	meta := make([]uint32, 0, len(msgs)*2)
	for _, m := range msgs {
		meta = append(meta, uint32(len(flat)), uint32(len(m)))
		flat = append(flat, m...)
	}
	if len(flat) == 0 {
		flat = []byte{0} // storage buffers must be non-empty
	}

	n := uint32(len(msgs))
	out, err := c.Run(KernelRun{
		WGSL:    CommonWGSL + SHA256WGSL,
		Entry:   "sha256_main",
		Inputs:  [][]byte{U32s(sha256K[:]...), bytesAsU32(flat), U32s(meta...)},
		Outputs: []int{len(msgs) * 32 * 4},
		Uniform: U32s(n),
		Groups:  [3]uint32{(n + 63) / 64, 1, 1},
	})
	if err != nil {
		t.Fatalf("Run: %v", err)
	}

	digests := make([][32]byte, len(msgs))
	raw := u32sAsBytes(out[0])
	for i := range msgs {
		copy(digests[i][:], raw[i*32:(i+1)*32])
	}
	return digests
}

// TestSHA256KAT checks the standard vectors plus AstroBWTv3's own KAT input.
func TestSHA256KAT(t *testing.T) {
	c := testCtx(t)

	msgs := [][]byte{
		[]byte(""),
		[]byte("a"),
		[]byte("abc"),
		[]byte("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"),
		bytes.Repeat([]byte("x"), 55), // one block, max without a second
		bytes.Repeat([]byte("x"), 56), // forces a second block
		bytes.Repeat([]byte("x"), 63),
		bytes.Repeat([]byte("x"), 64), // exact block boundary
		bytes.Repeat([]byte("x"), 65),
		bytes.Repeat([]byte("z"), 119),
		bytes.Repeat([]byte("z"), 120),
		bytes.Repeat([]byte("z"), 128),
		bytes.Repeat([]byte("q"), 255),
		bytes.Repeat([]byte("q"), 256), // SHA_MAX_BYTES
	}

	got := shaBatch(t, c, msgs)
	for i, m := range msgs {
		want := sha256.Sum256(m)
		if got[i] != want {
			t.Errorf("msg %d (len %d):\n got  %x\n want %x", i, len(m), got[i], want)
		}
	}
}

// TestSHA256Batch hashes a randomized batch at a live-ish batch size, with
// lengths biased toward the padding boundaries where SHA-256 goes wrong.
func TestSHA256Batch(t *testing.T) {
	c := testCtx(t)

	const batch = 4096
	rng := rand.New(rand.NewSource(4070))
	msgs := make([][]byte, batch)
	boundaries := []int{0, 1, 54, 55, 56, 63, 64, 65, 111, 112, 119, 120, 127, 128, 191, 192, 247, 248, 255, 256}
	for i := range msgs {
		var n int
		if i%3 == 0 {
			n = boundaries[rng.Intn(len(boundaries))]
		} else {
			n = rng.Intn(257)
		}
		m := make([]byte, n)
		rng.Read(m)
		msgs[i] = m
	}

	got := shaBatch(t, c, msgs)
	bad := 0
	for i, m := range msgs {
		if want := sha256.Sum256(m); got[i] != want {
			if bad < 5 {
				t.Errorf("msg %d (len %d):\n got  %x\n want %x", i, len(m), got[i], want)
			}
			bad++
		}
	}
	if bad != 0 {
		t.Fatalf("%d/%d digests wrong at batch %d", bad, batch, batch)
	}
}
