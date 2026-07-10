package gpu

import (
	"bytes"
	"encoding/binary"
	"math/rand"
	"testing"

	"github.com/Dirtybird99/Dirtybird-Go-GPU-Miner/internal/astrobwt"
)

const fhStride = 70912 / 4 // packed words per nonce (4 bytes/u32); == saStride/4

// TestFrontHalfFull verifies the full GPU front half (prologue + wolf loop) —
// one thread per nonce — reproduces the AstroBWTv3 data stream + data_len
// byte-for-byte vs astrobwt.WolfStreamTable (== OracleStream).
func TestFrontHalfFull(t *testing.T) {
	c := testCtx(t)
	rng := rand.New(rand.NewSource(7))
	const batch = 256
	var flat []byte
	meta := make([]uint32, 0, batch*2)
	wantData := make([][]byte, batch)
	wantLen := make([]uint32, batch)
	for i := 0; i < batch; i++ {
		blob := make([]byte, 76)
		rng.Read(blob)
		meta = append(meta, uint32(len(flat)), uint32(len(blob)))
		flat = append(flat, blob...)
		d, dl := astrobwt.WolfStreamTable(blob)
		wantData[i] = d
		wantLen[i] = dl
	}
	optable := astrobwt.WolfOpTable()

	out, err := c.Run(KernelRun{
		WGSL:    U64WGSL + FrontHalfWGSL,
		Entry:   "fronthalf_main",
		Inputs:  [][]byte{bytesAsU32(flat), U32s(meta...), U32s(sha256K[:]...), U32s(optable...)},
		Outputs: []int{batch * fhStride * 4, batch * 4}, // dataOut (packed words), outLen
		Uniform: U32s(batch, fhStride),
		Groups:  [3]uint32{(batch + 63) / 64, 1, 1},
	})
	if err != nil {
		t.Fatalf("Run: %v", err)
	}

	// out[0] is packed 4-bytes/u32; unpack per nonce to compare the data stream.
	badLen, badData := 0, 0
	rowBytes := fhStride * 4
	for i := 0; i < batch; i++ {
		gotLen := binary.LittleEndian.Uint32(out[1][i*4:])
		if gotLen != wantLen[i] {
			badLen++
			if badLen <= 3 {
				t.Errorf("nonce %d: data_len got %d want %d", i, gotLen, wantLen[i])
			}
			continue
		}
		got := out[0][i*rowBytes : i*rowBytes+int(gotLen)] // packed bytes == the stream, little-endian
		if !bytes.Equal(got, wantData[i]) {
			badData++
			if badData <= 3 {
				first := -1
				for k := range wantData[i] {
					if got[k] != wantData[i][k] {
						first = k
						break
					}
				}
				t.Errorf("nonce %d: data mismatch at byte %d (len %d)", i, first, gotLen)
			}
		}
	}
	if badLen != 0 || badData != 0 {
		t.Fatalf("front-half mismatch: %d/%d data_len, %d/%d data", badLen, batch, badData, batch)
	}
}
