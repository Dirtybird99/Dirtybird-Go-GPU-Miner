package gpu

import (
	"encoding/binary"
	"math/bits"
	"os"
	"sync"
	"testing"
)

// sharedCtx is the device shared by every test in this package; opening a
// device per test both slows the suite and can exhaust adapter handles.
var (
	sharedCtx  *Ctx
	sharedErr  error
	sharedOnce sync.Once
)

func testCtx(t *testing.T) *Ctx {
	t.Helper()
	sharedOnce.Do(func() {
		p := PowerHigh
		if os.Getenv("GPU_TEST_SOFTWARE") != "" {
			p = PowerSoftware
		}
		sharedCtx, sharedErr = NewCtx(p)
	})
	if sharedErr != nil {
		t.Skipf("no GPU device: %v", sharedErr)
	}
	return sharedCtx
}

// TestCtxLimitsAreGranted pins the invariant behind the SA slab guard: the
// limits a Ctx reports are the ones the device GRANTED, not what the adapter
// advertised. On backends where the two diverge (DX12 advertises a 128 GB
// buffer size while granting a 128 MiB storage binding), reporting the
// adapter's numbers let a slab through that CreateBindGroup then refused.
func TestCtxLimitsAreGranted(t *testing.T) {
	c := testCtx(t)
	got := c.Device.Limits()
	if c.MaxStorageBindingSize != got.MaxStorageBufferBindingSize {
		t.Errorf("MaxStorageBindingSize = %d, device granted %d", c.MaxStorageBindingSize, got.MaxStorageBufferBindingSize)
	}
	if c.MaxBufferSize != got.MaxBufferSize {
		t.Errorf("MaxBufferSize = %d, device granted %d", c.MaxBufferSize, got.MaxBufferSize)
	}
	if c.MaxStorageBuffers != got.MaxStorageBuffersPerShaderStage {
		t.Errorf("MaxStorageBuffers = %d, device granted %d", c.MaxStorageBuffers, got.MaxStorageBuffersPerShaderStage)
	}
	// A slab that passes the guard must actually build. MaxSlab is derived from
	// the granted binding size, so a session at (a small fraction of) it must
	// allocate and bind without error.
	if max := c.MaxSlab(); max > 0 {
		s := min(max, 8)
		sas, err := c.SuffixArrayBatch(make([][]byte, 0)) // exercise the nil path too
		_ = sas
		if err != nil {
			t.Fatalf("empty batch: %v", err)
		}
		datas := make([][]byte, s)
		for i := range datas {
			datas[i] = []byte("limits-probe")
		}
		if _, err := c.SuffixArrayBatch(datas); err != nil {
			t.Fatalf("slab %d within MaxSlab %d failed to build: %v", s, max, err)
		}
	}
}

const mixWGSL = `
@group(0) @binding(0) var<storage, read> input: array<u32>;
@group(0) @binding(1) var<storage, read_write> output: array<u32>;
struct Params { count: u32 }
@group(0) @binding(2) var<uniform> params: Params;

fn rotl(x: u32, r: u32) -> u32 { return (x << r) | (x >> (32u - r)); }

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= params.count) { return; }
    var x = input[i];
    x = rotl(x ^ 0x9e3779b9u, 13u) * 0x85ebca6bu;
    x = rotl(x, 7u) ^ (x >> 17u);
    output[i] = x;
}
`

func TestHarnessMixKernel(t *testing.T) {
	c := testCtx(t)

	const n = 4096
	in := make([]uint32, n)
	s := uint32(1)
	for i := range in {
		s = s*1664525 + 1013904223
		in[i] = s
	}

	out, err := c.Run(KernelRun{
		WGSL:    mixWGSL,
		Inputs:  [][]byte{U32s(in...)},
		Outputs: []int{n * 4},
		Uniform: U32s(n),
		Groups:  [3]uint32{(n + 255) / 256, 1, 1},
	})
	if err != nil {
		t.Fatalf("Run: %v", err)
	}

	rotl := func(x uint32, r uint) uint32 { return bits.RotateLeft32(x, int(r)) }
	for i := 0; i < n; i++ {
		x := rotl(in[i]^0x9e3779b9, 13) * 0x85ebca6b
		x = rotl(x, 7) ^ (x >> 17)
		got := binary.LittleEndian.Uint32(out[0][i*4:])
		if got != x {
			t.Fatalf("elem %d: got %08x want %08x", i, got, x)
		}
	}
}
