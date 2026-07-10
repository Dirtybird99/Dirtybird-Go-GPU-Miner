package gpu

import (
	"encoding/binary"
	"math/rand"
	"testing"

	"github.com/cespare/xxhash"
	"github.com/dchest/siphash"
	"github.com/segmentio/fasthash/fnv1a"
)

// TestFNV1a64 verifies the WGSL u64-emulated FNV-1a matches the Go reference
// (the same lib the AstroBWTv3 front-half uses) over a batch of varied lengths.
func TestFNV1a64(t *testing.T) {
	c := testCtx(t)
	rng := rand.New(rand.NewSource(64))
	const batch = 2048
	msgs := make([][]byte, batch)
	var flat []byte
	meta := make([]uint32, 0, batch*2)
	for i := range msgs {
		n := rng.Intn(257) // 0..256, the step_3[:pos2] range
		m := make([]byte, n)
		rng.Read(m)
		msgs[i] = m
		meta = append(meta, uint32(len(flat)), uint32(n))
		flat = append(flat, m...)
	}
	if len(flat) == 0 {
		flat = []byte{0}
	}

	out, err := c.Run(KernelRun{
		WGSL:    U64WGSL + Fnv1a64WGSL,
		Entry:   "fnv1a64_main",
		Inputs:  [][]byte{bytesAsU32(flat), U32s(meta...)},
		Outputs: []int{batch * 2 * 4},
		Uniform: U32s(batch),
		Groups:  [3]uint32{(batch + 63) / 64, 1, 1},
	})
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	bad := 0
	for i, m := range msgs {
		got := uint64(binary.LittleEndian.Uint32(out[0][i*8:])) | uint64(binary.LittleEndian.Uint32(out[0][i*8+4:]))<<32
		want := fnv1a.HashBytes64(m)
		if got != want {
			bad++
			if bad <= 4 {
				t.Errorf("msg %d (len %d): got %016x want %016x", i, len(m), got, want)
			}
		}
	}
	if bad != 0 {
		t.Fatalf("%d/%d fnv1a64 wrong", bad, batch)
	}
}

// TestSipHash verifies WGSL SipHash-2-4 matches dchest/siphash over random
// keys and lengths (the AstroBWTv3 threshold hash: siphash.Hash(tries, prev, data)).
func TestSipHash(t *testing.T) {
	c := testCtx(t)
	rng := rand.New(rand.NewSource(24))
	const batch = 2048
	type msg struct {
		k0, k1 uint64
		data   []byte
	}
	msgs := make([]msg, batch)
	var flat []byte
	meta := make([]uint32, 0, batch*6)
	for i := range msgs {
		n := rng.Intn(257)
		d := make([]byte, n)
		rng.Read(d)
		k0, k1 := rng.Uint64(), rng.Uint64()
		msgs[i] = msg{k0, k1, d}
		meta = append(meta, uint32(k0), uint32(k0>>32), uint32(k1), uint32(k1>>32), uint32(len(flat)), uint32(n))
		flat = append(flat, d...)
	}
	if len(flat) == 0 {
		flat = []byte{0}
	}
	out, err := c.Run(KernelRun{
		WGSL:    U64WGSL + SipHashWGSL,
		Entry:   "siphash_main",
		Inputs:  [][]byte{bytesAsU32(flat), U32s(meta...)},
		Outputs: []int{batch * 2 * 4},
		Uniform: U32s(batch),
		Groups:  [3]uint32{(batch + 63) / 64, 1, 1},
	})
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	bad := 0
	for i, m := range msgs {
		got := uint64(binary.LittleEndian.Uint32(out[0][i*8:])) | uint64(binary.LittleEndian.Uint32(out[0][i*8+4:]))<<32
		want := siphash.Hash(m.k0, m.k1, m.data)
		if got != want {
			bad++
			if bad <= 4 {
				t.Errorf("msg %d (len %d): got %016x want %016x", i, len(m.data), got, want)
			}
		}
	}
	if bad != 0 {
		t.Fatalf("%d/%d siphash wrong", bad, batch)
	}
}

// TestXXHash64 verifies WGSL XXH64 matches cespare/xxhash over random lengths
// (the AstroBWTv3 threshold hash: xxhash.Sum64(step_3[:pos2])).
func TestXXHash64(t *testing.T) {
	c := testCtx(t)
	rng := rand.New(rand.NewSource(99))
	const batch = 4096
	msgs := make([][]byte, batch)
	var flat []byte
	meta := make([]uint32, 0, batch*2)
	// include every length 0..256 explicitly plus random, to hit all paths
	for i := range msgs {
		var n int
		if i <= 256 {
			n = i
		} else {
			n = rng.Intn(257)
		}
		m := make([]byte, n)
		rng.Read(m)
		msgs[i] = m
		meta = append(meta, uint32(len(flat)), uint32(n))
		flat = append(flat, m...)
	}
	if len(flat) == 0 {
		flat = []byte{0}
	}
	out, err := c.Run(KernelRun{
		WGSL:    U64WGSL + XXHash64WGSL,
		Entry:   "xxhash64_main",
		Inputs:  [][]byte{bytesAsU32(flat), U32s(meta...)},
		Outputs: []int{batch * 2 * 4},
		Uniform: U32s(batch),
		Groups:  [3]uint32{(batch + 63) / 64, 1, 1},
	})
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	bad := 0
	for i, m := range msgs {
		got := uint64(binary.LittleEndian.Uint32(out[0][i*8:])) | uint64(binary.LittleEndian.Uint32(out[0][i*8+4:]))<<32
		want := xxhash.Sum64(m)
		if got != want {
			bad++
			if bad <= 6 {
				t.Errorf("msg %d (len %d): got %016x want %016x", i, len(m), got, want)
			}
		}
	}
	if bad != 0 {
		t.Fatalf("%d/%d xxhash64 wrong", bad, batch)
	}
}
