package gpu

import "fmt"

// saStride is the per-hash index space in the batched SA buffers. AstroBWTv3
// data never exceeds MAX_LENGTH = 70912 bytes, so one row holds any hash.
const saStride = 70912

// SADebugRounds, when nonzero, caps the SA refine doubling loop at
// SADebugRounds-1 rounds after the coarse sort — the kernel then stops with a
// PARTIAL, invalid SA. It exists only for profiling per-round cost (cmd/sabench
// -maxrounds). Production leaves it 0 (run the loop to completion). It is read
// at dispatch time on the single mining goroutine, so it needs no
// synchronization; never set it while mining.
var SADebugRounds uint32

// SuffixArrayBatch builds the suffix arrays of many hashes in a single GPU
// dispatch — one workgroup per hash, the whole prefix-doubling loop resident on
// the GPU (no per-round readback). datas[i] must be at most saStride bytes.
// Returns one []int32 SA per input, each of length len(datas[i]).
func (c *Ctx) SuffixArrayBatch(datas [][]byte) ([][]int32, error) {
	s := len(datas)
	if s == 0 {
		return nil, nil
	}
	// Per-binding cap: one SA row buffer is s*saStride*4 bytes and is bound as a
	// storage buffer, so MaxStorageBindingSize — not MaxBufferSize — is the limit
	// that decides whether it can bind. They diverge by orders of magnitude on
	// some backends (DX12 grants a 128 MiB storage binding while advertising a
	// 128 GB buffer size), so checking the wrong one lets a slab the device will
	// refuse sail through to CreateBindGroup. NOTE this bounds one buffer, not
	// total VRAM — a session allocates ~8 such buffers (~2.3 MB/hash), so a small
	// GPU may still OOM well before this limit.
	if lim := c.MaxStorageBindingSize; lim > 0 && uint64(s)*saStride*4 > lim {
		return nil, fmt.Errorf("slab too large: %d hashes exceed the %d MiB per-binding limit (~%d max); also mind total VRAM (~2.3 MB/hash)",
			s, lim>>20, lim/(saStride*4))
	}

	lens := make([]uint32, s)
	for i, d := range datas {
		if len(d) > saStride {
			return nil, fmt.Errorf("hash %d: len %d exceeds stride %d", i, len(d), saStride)
		}
		lens[i] = uint32(len(d))
	}

	// Reuse a persistent buffer/pipeline session sized for this slab + kernel;
	// rebuild only when either changes. This avoids re-allocating ~430 MB of
	// GPU buffers per dispatch.
	if c.saSess == nil || c.saSess.slab < s || c.saSess.refine != c.useRefine {
		if c.saSess != nil {
			c.saSess.release()
			c.saSess = nil
		}
		sess, err := c.buildSASession(s, c.useRefine)
		if err != nil {
			return nil, err
		}
		c.saSess = sess
	}

	saBytes, err := c.saSess.run(datas, lens, s)
	if err != nil {
		return nil, err
	}
	return decodeSA(saBytes, lens), nil
}
