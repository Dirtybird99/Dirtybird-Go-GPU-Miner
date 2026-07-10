package gpu

import (
	"strings"
	"testing"

	"github.com/Dirtybird99/Dirtybird-Go-GPU-Miner/internal/astrobwt"
)

func TestSPIRVStreamerVsOracle(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping native 2048-row stream gate in short mode")
	}
	c, err := NewCtxBackend(PowerHigh, BackendVulkan)
	if err != nil {
		t.Skipf("Vulkan unavailable: %v", err)
	}
	defer c.Close()
	if !c.CanSPIRV() || c.SubgroupSize() != 32 {
		t.Skip("native stream needs NVIDIA Vulkan with subgroup size 32")
	}
	c.UseRefine(true)
	c.UseSPIRV(true)
	p := NewPipeline(c)
	streamer, err := p.NewFusedStreamer(SPIRVBatchSize)
	if err != nil {
		t.Fatal(err)
	}

	sizes := []int{SPIRVBatchSize, 37}
	batches := make([][][]byte, len(sizes))
	want := make([][][32]byte, len(sizes))
	seed := uint32(0x5ab0face)
	for b, n := range sizes {
		batches[b] = make([][]byte, n)
		want[b] = make([][32]byte, n)
		for i := 0; i < n; i++ {
			blob := make([]byte, 76)
			for j := range blob {
				seed = seed*1664525 + 1013904223
				blob[j] = byte(seed >> 16)
			}
			batches[b][i] = blob
		}
		parallelFor(n, func(i int) { want[b][i] = astrobwt.Sum(batches[b][i]) })
	}

	var got [][][32]byte
	for _, batch := range batches {
		previous, err := streamer.Submit(batch)
		if err != nil {
			t.Fatal(err)
		}
		if previous != nil {
			got = append(got, previous)
		}
	}
	last, err := streamer.Drain()
	if err != nil {
		t.Fatal(err)
	}
	got = append(got, last)
	for b := range want {
		if len(got[b]) != len(want[b]) {
			t.Fatalf("batch %d: got %d hashes, want %d", b, len(got[b]), len(want[b]))
		}
		for i := range want[b] {
			if got[b][i] != want[b][i] {
				t.Fatalf("batch %d hash %d mismatch: got %x want %x", b, i, got[b][i], want[b][i])
			}
		}
	}

	// A deliberately huge equal-prefix run must trip the shader flag instead
	// of indexing beyond the 2048-entry cooperative sort scratch. The mining
	// streamer handles the same flag by recomputing that slab on the CPU.
	_, err = streamer.fast.base.runSPIRVData([][]byte{make([]byte, 4096)})
	if err == nil || !strings.Contains(err.Error(), "tied run longer than 2048") {
		t.Fatalf("oversize native run: got %v, want guarded rejection", err)
	}
}
