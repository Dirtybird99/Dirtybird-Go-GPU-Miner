// Command stabbench is the optimize-loop STABILITY METRIC harness (read-only
// ground truth). It drives the live miner's streaming path (FusedStreamer,
// double-buffered fused sessions) over a fixed seeded sequence of batches and
// times each steady-state Submit cycle. It prints ONE line with two numbers:
//
//	spread_pct median_ms_per_batch
//
// spread_pct = (p95 - p5) / median * 100 of per-batch wall times — the
// stability metric the loop MINIMIZES. median_ms_per_batch is the throughput
// guard: a change that lowers spread by slowing every batch down uniformly is
// not a win, so the loop must also hold the median within noise of baseline.
// The input sequence, batch size, and batch count are fixed so numbers are
// comparable across iterations — do NOT change them to move the metric.
package main

import (
	"flag"
	"fmt"
	"os"
	"sort"
	"time"

	"github.com/Dirtybird99/Dirtybird-Go-GPU-Miner/internal/gpu"
)

const (
	batchSize = 256 // matches the live miner
	blobSize  = 76
)

func main() {
	batches := flag.Int("batches", 100, "timed steady-state batches")
	warmup := flag.Int("warmup", 4, "untimed warmup batches")
	verbose := flag.Bool("v", false, "per-batch detail to stderr")
	flag.Parse()

	ctx, err := gpu.NewCtx(gpu.PowerHigh)
	if err != nil {
		fmt.Fprintf(os.Stderr, "gpu init: %v\n", err)
		os.Exit(1)
	}
	defer ctx.Close()
	if sg := ctx.SubgroupSize(); sg < 32 {
		fmt.Fprintf(os.Stderr, "stabbench needs the fused subgroup path (subgroup size %d < 32)\n", sg)
		os.Exit(1)
	}
	ctx.UseRefine(true)
	pipe := gpu.NewPipeline(ctx)
	streamer, err := pipe.NewFusedStreamer(batchSize)
	if err != nil {
		fmt.Fprintf(os.Stderr, "streamer init: %v\n", err)
		os.Exit(1)
	}
	if *verbose {
		fmt.Fprintf(os.Stderr, "device: %s  batch=%d batches=%d\n", ctx.Name, batchSize, *batches)
	}

	// Two host-side blob buffers, ping-ponged like mineLoopStream. Contents are
	// a fixed LCG sequence keyed on the global batch index so every run (and
	// every iteration of the loop) hashes the identical input stream.
	blobBuf := [2][][]byte{mkBlobs(), mkBlobs()}
	fill := func(blobs [][]byte, batchIdx int) {
		seed := uint32(0x5AB0FACE) ^ uint32(batchIdx)*2654435761
		for i := range blobs {
			for j := 0; j < blobSize; j++ {
				seed = seed*1664525 + 1013904223
				blobs[i][j] = byte(seed >> 16)
			}
		}
	}

	total := *warmup + *batches
	durs := make([]float64, 0, *batches)
	var prevT time.Time
	for k := 0; k < total; k++ {
		blobs := blobBuf[k%2]
		fill(blobs, k)
		pows, err := streamer.Submit(blobs)
		if err != nil {
			fmt.Fprintf(os.Stderr, "submit %d: %v\n", k, err)
			os.Exit(1)
		}
		// Consume like the miner: touch every returned hash (target scan).
		if pows != nil {
			var acc byte
			for i := range pows {
				acc ^= pows[i][31]
			}
			_ = acc
		}
		now := time.Now()
		if k > *warmup { // first timed interval starts after the warmup Submit
			d := now.Sub(prevT).Seconds() * 1000
			durs = append(durs, d)
			if *verbose {
				fmt.Fprintf(os.Stderr, "  batch %d: %.1f ms\n", k, d)
			}
		}
		prevT = now
	}
	if _, err := streamer.Drain(); err != nil {
		fmt.Fprintf(os.Stderr, "drain: %v\n", err)
		os.Exit(1)
	}

	sorted := append([]float64(nil), durs...)
	sort.Float64s(sorted)
	n := len(sorted)
	p5, p50, p95 := sorted[n*5/100], sorted[n/2], sorted[n*95/100]
	spread := (p95 - p5) / p50 * 100
	if *verbose {
		fmt.Fprintf(os.Stderr, "p5=%.1f p50=%.1f p95=%.1f max=%.1f ms  spread=%.2f%%  (%.0f H/s median)\n",
			p5, p50, p95, sorted[n-1], spread, float64(batchSize)/p50*1000)
	}
	// The single metric line the loop reads: spread%% (minimize) + median guard.
	fmt.Printf("%.2f %.2f\n", spread, p50)
}

func mkBlobs() [][]byte {
	b := make([][]byte, batchSize)
	for i := range b {
		b[i] = make([]byte, blobSize)
	}
	return b
}
