package astrobwt

// Table-driven reproduction of the AstroBWTv3 wolf loop, used to VERIFY the
// micro-op table emitted by cmd/genwolf reproduces the reference bit-exact
// (see wolftable_test.go). The WGSL front-half kernel is a direct translation
// of this: a switch over ~13 micro-op ids driven by wolfOps, instead of 256
// literal cases. The op ids MUST match cmd/genwolf.

import (
	"io"
	"math/bits"

	"github.com/cespare/xxhash"
	"github.com/dchest/siphash"
	"github.com/segmentio/fasthash/fnv1a"
	"golang.org/x/crypto/salsa20/salsa"
)

const (
	opRotlK    = 1
	opXorRotlK = 2
	opShl      = 3
	opShr      = 4
	opXorP     = 5
	opAndP     = 6
	opNot      = 7
	opXorPop   = 8
	opRev      = 9
	opRotlX    = 10
	opAdd      = 11
	opMul      = 12
	opSub97    = 13
)

// wolfApplyByte applies case c's micro-op sequence to byte x, with p = the
// current step_3[pos2].
func wolfApplyByte(c uint8, x, p byte) byte {
	for _, code := range wolfOps[c] {
		op := code >> 8
		param := int(code & 0xff)
		switch op {
		case opRotlK:
			x = bits.RotateLeft8(x, param)
		case opXorRotlK:
			x = x ^ bits.RotateLeft8(x, param)
		case opShl:
			x = x << (x & 3)
		case opShr:
			x = x >> (x & 3)
		case opXorP:
			x = x ^ p
		case opAndP:
			x = x & p
		case opNot:
			x = ^x
		case opXorPop:
			x = x ^ byte(bits.OnesCount8(x))
		case opRev:
			x = bits.Reverse8(x)
		case opRotlX:
			x = bits.RotateLeft8(x, int(x))
		case opAdd:
			x = x + x
		case opMul:
			x = x * x
		case opSub97:
			x = x - (x ^ 97)
		}
	}
	return x
}

// WolfStreamInto reproduces astroBWTv3Stream's data stream + length in out.
// out must have room for the longest possible stream.
func WolfStreamInto(input, out []byte) (dataLen uint32, err error) {
	if len(out) < int(MAX_LENGTH) {
		return 0, io.ErrShortBuffer
	}

	var step3 [256]byte
	var counter [16]byte
	shaKey := sha256Fallback(input)
	salsa.XORKeyStream(step3[:], step3[:], &counter, &shaKey)
	rc4 := NewCipher(step3[:])
	rc4.XORKeyStream(step3[:], step3[:])
	lhash := fnv1a.HashBytes64(step3[:])
	prevLhash := lhash

	tries := uint64(0)
	for {
		tries++
		rs := prevLhash ^ lhash ^ tries
		op := byte(rs)
		pos1 := byte(rs >> 8)
		pos2 := byte(rs >> 16)
		if pos1 > pos2 {
			pos1, pos2 = pos2, pos1
		}
		if pos2-pos1 > 32 {
			pos2 = pos1 + (pos2-pos1)&0x1f
		}

		if op == 254 || op == 255 {
			rc4 = NewCipher(step3[:]) // case 254/255 rekey (before the byte loop)
		}
		for i := pos1; i < pos2; i++ {
			step3[i] = wolfApplyByte(op, step3[i], step3[pos2])
			if op == 0 { // case 0's in-loop reverse-swap of pos1/pos2
				step3[pos2], step3[pos1] = bits.Reverse8(step3[pos1]), bits.Reverse8(step3[pos2])
			}
			if op == 253 { // case 253's per-byte xxhash
				prevLhash = lhash + prevLhash
				lhash = xxhash.Sum64(step3[:pos2])
			}
		}

		if step3[pos1]-step3[pos2] < 0x10 {
			prevLhash = lhash + prevLhash
			lhash = xxhash.Sum64(step3[:pos2])
		}
		if step3[pos1]-step3[pos2] < 0x20 {
			prevLhash = lhash + prevLhash
			lhash = fnv1a.HashBytes64(step3[:pos2])
		}
		if step3[pos1]-step3[pos2] < 0x30 {
			prevLhash = lhash + prevLhash
			lhash = siphash.Hash(tries, prevLhash, step3[:pos2])
		}
		if step3[pos1]-step3[pos2] <= 0x40 {
			rc4.XORKeyStream(step3[:], step3[:])
		}

		step3[255] = step3[255] ^ step3[pos1] ^ step3[pos2]
		copy(out[(tries-1)*256:], step3[:])

		if tries > 260+16 || (step3[255] >= 0xf0 && tries > 260) {
			break
		}
	}

	return uint32((tries-4)*256 + (uint64(step3[253])<<8|uint64(step3[254]))&0x3ff), nil
}

// WolfStreamTable reproduces astroBWTv3Stream's data stream + length using the
// generated op table instead of the 256-case switch. Everything else (prologue,
// nonce mixing, threshold hashes, RC4, break, data_len) mirrors pow.go exactly.
func WolfStreamTable(input []byte) (data []byte, dataLen uint32) {
	out := make([]byte, MAX_LENGTH+64)
	dataLen, _ = WolfStreamInto(input, out)
	return out[:dataLen], dataLen
}

// WolfOpTable returns the micro-op table flattened for GPU upload: 256 cases,
// 5 u32 each — [count, op0, op1, op2, op3] (codes are op<<8|param, 0 padded).
func WolfOpTable() []uint32 {
	t := make([]uint32, 256*5)
	for c := 0; c < 256; c++ {
		ops := wolfOps[c]
		t[c*5] = uint32(len(ops))
		for k, code := range ops {
			t[c*5+1+k] = uint32(code)
		}
	}
	return t
}
