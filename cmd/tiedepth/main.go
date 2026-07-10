// Command tiedepth measures the tie-depth distribution of real AstroBWTv3
// data buffers — the single most decision-relevant unknown for the GPU
// suffix-array design (it sets the byte-MSD → rank-doubling phase switch and
// the round budget). It generates real v3 buffers via the CPU oracle, builds
// their true suffix arrays, and histograms the LCP of adjacent SA entries
// (= the byte depth at which a byte-MSD refine separates each pair) plus the
// tie-group sizes that survive a 4- and 5-byte coarse prefix sort.
//
// No GPU involved; this validates the ~87%-tied / ~60k-tied priors and the
// claim that pure byte-deepening has no bound while rank-doubling (≤ceil(log2 n))
// does.
package main

import (
	"flag"
	"fmt"
	"math"
	"math/rand"
	"sort"

	"github.com/Dirtybird99/Dirtybird-Go-GPU-Miner/internal/astrobwt"
)

func main() {
	nHashes := flag.Int("n", 256, "number of hashes to sample")
	seed := flag.Int64("seed", 4070, "RNG seed")
	flag.Parse()

	rng := rand.New(rand.NewSource(*seed))

	// aggregate accumulators
	var (
		totalSuffixes uint64
		sumLen        uint64
		minLen        = uint32(math.MaxUint32)
		maxLen        uint32
		// LCP threshold counters: adjacent pairs with LCP >= t
		thresholds = []int{1, 2, 3, 4, 5, 6, 8, 12, 16, 24, 32, 64, 128, 256, 512, 1024, 4096}
		geCount    = make([]uint64, len(thresholds))
		maxLCP     int
		sumLCP     uint64
		// tie-group sizes surviving a k-byte prefix
		tie4Sizes []int // max group sizes per hash after 4-byte prefix
		tie5Sizes []int
		// round-budget check: max ceil(log2 n) across hashes
		maxLog2 int
	)

	lenSamples := make([]int, 0, *nHashes)

	for h := 0; h < *nHashes; h++ {
		blob := make([]byte, 76)
		rng.Read(blob)
		data, sa, dl := astrobwt.OracleSA(blob)
		n := int(dl)
		if n < 2 {
			continue
		}
		totalSuffixes += uint64(n)
		sumLen += uint64(n)
		lenSamples = append(lenSamples, n)
		if dl < minLen {
			minLen = dl
		}
		if dl > maxLen {
			maxLen = dl
		}
		if l := ceilLog2(n); l > maxLog2 {
			maxLog2 = l
		}

		lcp := kasaiLCP(data, sa)
		// lcp[k] = LCP(sa[k-1], sa[k]) for k in 1..n-1
		for k := 1; k < n; k++ {
			l := lcp[k]
			sumLCP += uint64(l)
			if l > maxLCP {
				maxLCP = l
			}
			for ti, t := range thresholds {
				if l >= t {
					geCount[ti]++
				}
			}
		}

		tie4Sizes = append(tie4Sizes, maxGroupSize(data, sa, 4))
		tie5Sizes = append(tie5Sizes, maxGroupSize(data, sa, 5))
	}

	adjacentPairs := totalSuffixes - uint64(len(lenSamples)) // n-1 per hash

	fmt.Printf("=== AstroBWTv3 tie-depth report (%d hashes, seed %d) ===\n\n", len(lenSamples), *seed)
	fmt.Printf("data_len:  min=%d  max=%d  mean=%.0f  (MAX_LENGTH=70912)\n", minLen, maxLen, float64(sumLen)/float64(len(lenSamples)))
	fmt.Printf("suffixes:  total=%d  mean/hash=%.0f\n", totalSuffixes, float64(totalSuffixes)/float64(len(lenSamples)))
	fmt.Printf("adjacent SA pairs analyzed: %d\n\n", adjacentPairs)

	fmt.Printf("LCP of adjacent SA entries (= byte-MSD separation depth):\n")
	fmt.Printf("  mean LCP = %.1f   max LCP = %d   ceil(log2 n) max = %d\n\n", float64(sumLCP)/float64(adjacentPairs), maxLCP, maxLog2)
	fmt.Printf("  %-10s %-14s %s\n", "LCP >= k", "pairs", "fraction (still tied after k-byte coarse sort)")
	for ti, t := range thresholds {
		frac := float64(geCount[ti]) / float64(adjacentPairs)
		fmt.Printf("  %-10d %-14d %6.2f%%  %s\n", t, geCount[ti], frac*100, bar(frac))
	}

	fmt.Printf("\nKey priors check:\n")
	frac4 := float64(geCount[idxOf(thresholds, 4)]) / float64(adjacentPairs)
	frac5 := float64(geCount[idxOf(thresholds, 5)]) / float64(adjacentPairs)
	fmt.Printf("  tied after 4-byte prefix: %.1f%%   after 5-byte: %.1f%%   (CUDA prior: ~87%% at 5)\n", frac4*100, frac5*100)

	fmt.Printf("\nMax tie-group size surviving a coarse prefix sort (per hash):\n")
	fmt.Printf("  4-byte prefix: %s\n", stat(tie4Sizes))
	fmt.Printf("  5-byte prefix: %s\n", stat(tie5Sizes))

	fmt.Printf("\nDesign implications:\n")
	fmt.Printf("  - byte-MSD worst-case depth (max LCP) = %d bytes -> unbounded refine rounds if byte-only.\n", maxLCP)
	fmt.Printf("  - rank-doubling bound (ceil log2 n) = %d rounds -> the phase-2 tail's guarantee.\n", maxLog2)
	fmt.Printf("  - phase-1 (byte-MSD) resolves pairs up to the depth where the fraction curve collapses;\n")
	fmt.Printf("    switch to phase-2 (doubling) for the long-LCP tail above.\n")
}

// kasaiLCP returns lcp where lcp[k] = LCP(text[sa[k-1]:], text[sa[k]:]).
func kasaiLCP(text []byte, sa []int32) []int {
	n := len(sa)
	rank := make([]int, n)
	for i := 0; i < n; i++ {
		rank[sa[i]] = i
	}
	lcp := make([]int, n)
	h := 0
	for i := 0; i < n; i++ {
		if rank[i] > 0 {
			j := int(sa[rank[i]-1])
			for i+h < n && j+h < n && text[i+h] == text[j+h] {
				h++
			}
			lcp[rank[i]] = h
			if h > 0 {
				h--
			}
		} else {
			h = 0
		}
	}
	return lcp
}

// maxGroupSize returns the largest run of SA entries sharing their first k bytes.
func maxGroupSize(text []byte, sa []int32, k int) int {
	n := len(sa)
	best, run := 1, 1
	for i := 1; i < n; i++ {
		if sharePrefix(text, int(sa[i-1]), int(sa[i]), k) {
			run++
			if run > best {
				best = run
			}
		} else {
			run = 1
		}
	}
	return best
}

// sharePrefix reports whether suffixes a and b share their first k bytes
// (a suffix shorter than k never ties, matching sais end-of-buffer order).
func sharePrefix(text []byte, a, b, k int) bool {
	n := len(text)
	for i := 0; i < k; i++ {
		ai, bi := a+i, b+i
		if ai >= n || bi >= n {
			return false
		}
		if text[ai] != text[bi] {
			return false
		}
	}
	return true
}

func ceilLog2(n int) int {
	l := 0
	for (1 << l) < n {
		l++
	}
	return l
}

func idxOf(s []int, v int) int {
	for i, x := range s {
		if x == v {
			return i
		}
	}
	return 0
}

func stat(v []int) string {
	if len(v) == 0 {
		return "(none)"
	}
	sorted := append([]int(nil), v...)
	sort.Ints(sorted)
	sum := 0
	for _, x := range sorted {
		sum += x
	}
	return fmt.Sprintf("min=%d  median=%d  mean=%.0f  max=%d", sorted[0], sorted[len(sorted)/2], float64(sum)/float64(len(sorted)), sorted[len(sorted)-1])
}

func bar(frac float64) string {
	n := int(frac * 40)
	b := make([]byte, n)
	for i := range b {
		b[i] = '#'
	}
	return string(b)
}
