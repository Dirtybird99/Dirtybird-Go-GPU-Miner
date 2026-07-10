package astrobwt

import (
	"bytes"
	"errors"
	"io"
	"math/rand"
	"testing"
)

// TestWolfTableMatchesOracle proves the generated micro-op table + table-driven
// wolf loop reproduce the reference stream (OracleStream) byte-for-byte. If any
// table entry or special-case handling is wrong, the data/length diverges.
// This validates the table before it is translated to WGSL.
func TestWolfTableMatchesOracle(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	into := make([]byte, MAX_LENGTH)
	for i := 0; i < 2000; i++ {
		blob := make([]byte, 76)
		rng.Read(blob)
		wantData, wantLen := OracleStream(blob)
		gotData, gotLen := WolfStreamTable(blob)
		if gotLen != wantLen {
			t.Fatalf("case %d: data_len got %d want %d", i, gotLen, wantLen)
		}
		if !bytes.Equal(gotData, wantData) {
			first := -1
			for k := range wantData {
				if gotData[k] != wantData[k] {
					first = k
					break
				}
			}
			t.Fatalf("case %d: data mismatch at byte %d (len %d)", i, first, wantLen)
		}
		intoLen, err := WolfStreamInto(blob, into)
		if err != nil {
			t.Fatalf("case %d: WolfStreamInto: %v", i, err)
		}
		if intoLen != wantLen || !bytes.Equal(into[:intoLen], wantData) {
			t.Fatalf("case %d: WolfStreamInto mismatch: lens got %d want %d", i, intoLen, wantLen)
		}
	}
}

// TestWolfTableKAT checks the table path also reproduces the pow("a") stream.
func TestWolfTableKAT(t *testing.T) {
	wantData, wantLen := OracleStream([]byte("a"))
	gotData, gotLen := WolfStreamTable([]byte("a"))
	if gotLen != wantLen || !bytes.Equal(gotData, wantData) {
		t.Fatalf("KAT stream mismatch: lens got %d want %d", gotLen, wantLen)
	}
	into := make([]byte, MAX_LENGTH)
	intoLen, err := WolfStreamInto([]byte("a"), into)
	if err != nil || intoLen != wantLen || !bytes.Equal(into[:intoLen], wantData) {
		t.Fatalf("WolfStreamInto KAT mismatch: len %d, want %d, err %v", intoLen, wantLen, err)
	}
}

func TestWolfStreamIntoRejectsShortBuffer(t *testing.T) {
	if n, err := WolfStreamInto([]byte("a"), make([]byte, MAX_LENGTH-1)); n != 0 || !errors.Is(err, io.ErrShortBuffer) {
		t.Fatalf("got len %d, err %v; want len 0, io.ErrShortBuffer", n, err)
	}
}
