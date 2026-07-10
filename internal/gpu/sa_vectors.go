package gpu

import "bytes"

// This file is the one source of truth for the adversarial SA corpus and the
// reference it is checked against. Both the test suite (sa_synthetic_test.go)
// and the runtime startup gate (Pipeline.SelfTest) consume it, so the gate
// verifies exactly the corners the tests verify — on whatever GPU/driver the
// miner actually starts on, which the test machine cannot stand in for.

// SyntheticSACase is one adversarial suffix-array input with a known-correct
// answer (via saisRef).
type SyntheticSACase struct {
	Name string
	Data []byte
	// Slow marks cases whose big, deeply-tied inputs make the O(n^3) saisRef
	// reference take seconds — the startup gate and -short test runs skip them.
	Slow bool
}

// SyntheticSACases returns the adversarial inputs for the coarse depth-4 sort
// that random real buffers never produce: suffixes whose 4-byte window runs off
// the end of the text (where the implicit sentinel must sort below every byte,
// including a literal 0x00), buckets far wider than a workgroup, and texts
// whose length is not a multiple of the 4-byte packing word.
func SyntheticSACases() []SyntheticSACase {
	return []SyntheticSACase{
		// The sentinel-vs-literal-zero trap: a real 0x00 byte must NOT tie with
		// a suffix that ended. Trailing zeros put both in play at once.
		{"trailing zeros x1", append(bytes.Repeat([]byte("ab"), 40), 0), false},
		{"trailing zeros x2", append(bytes.Repeat([]byte("ab"), 40), 0, 0), false},
		{"trailing zeros x3", append(bytes.Repeat([]byte("ab"), 40), 0, 0, 0), false},
		{"trailing zeros x4", append(bytes.Repeat([]byte("ab"), 40), 0, 0, 0, 0), false},
		{"all zeros", make([]byte, 300), false},
		{"zeros then byte", append(make([]byte, 300), 1), false},

		// Every suffix shares a depth-4 prefix, so the coarse sort resolves
		// nothing and the whole text enters the doubling tail as one bucket —
		// far wider than the 1024 the design doc calls "oversize", and wider
		// than the 256-lane workgroup.
		{"one bucket 4-gram", bytes.Repeat([]byte("abcd"), 1200), true},
		{"all equal", bytes.Repeat([]byte{0x5a}, 4096), true},

		// LCP > 256: the tail must keep doubling well past the depth where the
		// measured tie distribution collapses.
		{"lcp 2000", append(bytes.Repeat([]byte{7}, 2000), append([]byte{1}, bytes.Repeat([]byte{7}, 2000)...)...), true},

		// n % 4 != 0 — the last packed word holds bytes past the end of the
		// text, which rf_byte must never let into a key.
		{"n mod 4 = 1", bytes.Repeat([]byte("xyz"), 120)[:333], false},
		{"n mod 4 = 2", bytes.Repeat([]byte("xyz"), 120)[:334], false},
		{"n mod 4 = 3", bytes.Repeat([]byte("xyz"), 120)[:335], false},

		// Tiny n, where a suffix's whole 4-byte window is sentinel.
		{"n=1", []byte{0}, false},
		{"n=2", []byte{0, 0}, false},
		{"n=3", []byte{9, 9, 9}, false},
		{"n=4", []byte{9, 9, 9, 9}, false},
		{"n=5", []byte("aaaaa"), false},
	}
}

// saisRef is the plain-definition reference suffix array: stable comparison
// sort of suffixes with the shorter-suffix-first (implicit sentinel)
// convention, matching sais.go. O(n^2 log n)-ish — small inputs only.
func saisRef(text []byte) []int32 {
	n := len(text)
	sa := make([]int32, n)
	for i := range sa {
		sa[i] = int32(i)
	}
	less := func(a, b int32) bool {
		i, j := int(a), int(b)
		for i < n && j < n {
			if text[i] != text[j] {
				return text[i] < text[j]
			}
			i++
			j++
		}
		return i == n && j < n // shorter suffix first
	}
	// insertion-ish stable sort is fine for tiny inputs
	for i := 1; i < n; i++ {
		for j := i; j > 0 && less(sa[j], sa[j-1]); j-- {
			sa[j], sa[j-1] = sa[j-1], sa[j]
		}
	}
	return sa
}

func equalI32(a, b []int32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
