package astrobwt

// This file exists only in the GPU miner: it exposes the intermediate
// AstroBWTv3 stages so GPU kernels can be parity-tested stage by stage.
// Nothing here touches the hash computation.

import (
	"github.com/segmentio/fasthash/fnv1a"
	"golang.org/x/crypto/salsa20/salsa"
)

// OraclePrologue reproduces the AstroBWTv3 prologue exactly as
// astroBWTv3Stream does, exposing the intermediate 256-byte state so the GPU
// front-half kernels can be parity-tested stage by stage.
//
//	afterSalsa = Salsa20/20(key=sha256(input), counter=0) keystream (256 B)
//	afterRC4   = RC4(key=afterSalsa) applied to afterSalsa in place
//	lhash      = fnv1a64(afterRC4)
func OraclePrologue(input []byte) (shaKey [32]byte, afterSalsa, afterRC4 [256]byte, lhash uint64) {
	shaKey = sha256Fallback(input)

	var counter [16]byte
	var step3 [256]byte
	salsa.XORKeyStream(step3[:], step3[:], &counter, &shaKey)
	afterSalsa = step3

	rc4 := NewCipher(step3[:])
	rc4.XORKeyStream(step3[:], step3[:])
	afterRC4 = step3

	lhash = fnv1a.HashBytes64(afterRC4[:])
	return
}

// OracleStream runs the front half (prologue + wolf loop) and returns copies
// of the data stream the suffix array is built over, plus its length.
func OracleStream(input []byte) (data []byte, dataLen uint32) {
	scratch := NewScratchData()
	scratch.useV114 = false
	dataLen = astroBWTv3Stream(input, scratch)
	data = make([]byte, dataLen)
	copy(data, scratch.data[:dataLen])
	return data, dataLen
}

// OracleSA runs everything up to and including the suffix array and returns
// copies of the data stream and the SA (reference sais path, not v114).
func OracleSA(input []byte) (data []byte, sa []int32, dataLen uint32) {
	scratch := NewScratchData()
	scratch.useV114 = false
	dataLen = astroBWTv3Stream(input, scratch)
	data = make([]byte, dataLen)
	copy(data, scratch.data[:dataLen])
	sa = make([]int32, dataLen)
	copy(sa, scratch.sa[:dataLen])
	return data, sa, dataLen
}
