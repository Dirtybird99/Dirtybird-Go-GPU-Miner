// Command sabench is the optimize-loop METRIC harness (read-only ground truth).
// It builds a fixed seeded set of real AstroBWTv3 data buffers, warms the GPU
// suffix-array path, times several reps of the batched SA build, and prints one
// number: median milliseconds per hash. Lower is better. The input set, batch
// size, and rep count are fixed so the number is comparable across iterations —
// do NOT change them to move the metric.
package main

import (
	"flag"
	"fmt"
	"os"
	"sort"
	"time"

	"github.com/Dirtybird99/Dirtybird-Go-GPU-Miner/internal/astrobwt"
	"github.com/Dirtybird99/Dirtybird-Go-GPU-Miner/internal/gpu"
)

func main() {
	batch := flag.Int("batch", 256, "hashes per dispatch")
	reps := flag.Int("reps", 7, "timed reps (median reported)")
	verbose := flag.Bool("v", false, "print device + per-rep detail to stderr")
	maxRounds := flag.Int("maxrounds", -1, "PROFILING: cap the refine doubling loop at N rounds after the coarse sort (-1 = full/production; 0 = coarse only; the SA is invalid when capped, timing only)")
	flag.Parse()
	if *maxRounds >= 0 {
		gpu.SADebugRounds = uint32(*maxRounds) + 1 // encode: dbg_rounds-1 rounds run
	}

	ctx, err := gpu.NewCtx(gpu.PowerHigh)
	if err != nil {
		fmt.Fprintf(os.Stderr, "gpu init: %v\n", err)
		os.Exit(1)
	}
	defer ctx.Close()
	ctx.UseRefine(true)
	if *verbose {
		fmt.Fprintf(os.Stderr, "device: %s  batch=%d reps=%d\n", ctx.Name, *batch, *reps)
	}

	// Fixed seeded input set (deterministic across runs/iterations).
	datas := make([][]byte, *batch)
	seed := uint32(0x5AB0FACE)
	for i := range datas {
		blob := make([]byte, 76)
		for j := range blob {
			seed = seed*1664525 + 1013904223
			blob[j] = byte(seed >> 16)
		}
		d, dl := astrobwt.OracleStream(blob)
		datas[i] = d[:dl]
	}

	// Warm up (shader compile, pool priming).
	for w := 0; w < 2; w++ {
		if _, err := ctx.SuffixArrayBatch(datas); err != nil {
			fmt.Fprintf(os.Stderr, "warmup: %v\n", err)
			os.Exit(1)
		}
	}

	perHash := make([]float64, *reps)
	for r := 0; r < *reps; r++ {
		start := time.Now()
		if _, err := ctx.SuffixArrayBatch(datas); err != nil {
			fmt.Fprintf(os.Stderr, "rep %d: %v\n", r, err)
			os.Exit(1)
		}
		el := time.Since(start)
		perHash[r] = float64(el.Nanoseconds()) / 1e6 / float64(*batch)
		if *verbose {
			fmt.Fprintf(os.Stderr, "  rep %d: %.4f ms/hash (%.0f H/s)\n", r, perHash[r], 1000.0/perHash[r])
		}
	}
	sort.Float64s(perHash)
	median := perHash[len(perHash)/2]
	if *verbose {
		fmt.Fprintf(os.Stderr, "median: %.4f ms/hash = %.0f H/s\n", median, 1000.0/median)
	}
	// The single metric line the loop reads (ms/hash, minimize).
	fmt.Printf("%.4f\n", median)
}
