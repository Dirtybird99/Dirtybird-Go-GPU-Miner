// Command gpuprobe is the milestone-0 feasibility probe for the Dirtybird
// Go GPU miner: it enumerates adapters, runs a WGSL compute dispatch on the
// default adapter, and verifies the result bit-exactly against a CPU
// reference. Exit 0 = the pure-Go WebGPU compute path works on this machine.
package main

import (
	"context"
	"encoding/binary"
	"flag"
	"fmt"
	"log"
	"time"

	"github.com/gogpu/gputypes"
	"github.com/gogpu/wgpu"

	_ "github.com/gogpu/wgpu/hal/allbackends"
)

// probeWGSL exercises the integer ops AstroBWTv3 kernels will lean on:
// 32-bit adds/xors/rotates and per-element addressing. out[i] is a mix of
// rotations and multiplies over in[i] so a driver miscompile shows up as a
// mismatch, not a lucky pass.
const probeWGSL = `
@group(0) @binding(0) var<storage, read> input: array<u32>;
@group(0) @binding(1) var<storage, read_write> output: array<u32>;

struct Params { count: u32 }
@group(0) @binding(2) var<uniform> params: Params;

fn rotl(x: u32, r: u32) -> u32 {
    return (x << r) | (x >> (32u - r));
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= params.count) { return; }
    var x = input[i];
    x = rotl(x ^ 0x9e3779b9u, 13u) * 0x85ebca6bu;
    x = rotl(x, 7u) ^ (x >> 17u);
    x = x * 0xc2b2ae35u + i;
    output[i] = x;
}
`

const n = 1 << 20 // 1M elements

func cpuRef(in []uint32) []uint32 {
	rotl := func(x uint32, r uint) uint32 { return x<<r | x>>(32-r) }
	out := make([]uint32, len(in))
	for i, v := range in {
		x := rotl(v^0x9e3779b9, 13) * 0x85ebca6b
		x = rotl(x, 7) ^ (x >> 17)
		x = x*0xc2b2ae35 + uint32(i)
		out[i] = x
	}
	return out
}

func main() {
	matrix := flag.Bool("matrix", false, "sweep every registered backend and report limits/subgroup/kernel-compile")
	one := flag.String("one", "", "probe a single backend in-process (used by -matrix subprocesses)")
	power := flag.String("power", "high", "adapter class for -one: high|low|software")
	flag.Parse()
	if *one != "" {
		probeOne(*one, *power)
		return
	}
	if *matrix {
		runMatrix()
		return
	}

	instance, err := wgpu.CreateInstance(nil)
	if err != nil {
		log.Fatalf("FAIL: CreateInstance: %v", err)
	}
	defer instance.Release()

	// v0.30.10 has no EnumerateAdapters; reach distinct adapters through the
	// three RequestAdapterOptions variants and dedupe by name.
	variants := []struct {
		label string
		opts  *wgpu.RequestAdapterOptions
	}{
		{"high-performance", &wgpu.RequestAdapterOptions{PowerPreference: wgpu.PowerPreferenceHighPerformance}},
		{"low-power", &wgpu.RequestAdapterOptions{PowerPreference: wgpu.PowerPreferenceLowPower}},
		{"software-fallback", &wgpu.RequestAdapterOptions{ForceFallbackAdapter: true}},
	}

	seen := map[string]bool{}
	tested, failures := 0, 0
	for _, v := range variants {
		adapter, err := instance.RequestAdapter(v.opts)
		if err != nil {
			fmt.Printf("\n[%s] no adapter: %v\n", v.label, err)
			continue
		}
		info := adapter.Info()
		if seen[info.Name] {
			adapter.Release()
			continue
		}
		seen[info.Name] = true
		tested++
		fmt.Printf("\n[%s] %s (backend %v, type %v)\n", v.label, info.Name, info.Backend, info.DeviceType)
		lim := adapter.Limits()
		fmt.Printf("    limits: storageBinding=%dMiB buffer=%dMiB wgStorage=%dKiB wgSize=%d invocations=%d storageBufs/stage=%d\n",
			lim.MaxStorageBufferBindingSize>>20, lim.MaxBufferSize>>20,
			lim.MaxComputeWorkgroupStorageSize>>10, lim.MaxComputeWorkgroupSizeX,
			lim.MaxComputeInvocationsPerWorkgroup, lim.MaxStorageBuffersPerShaderStage)
		if err := runOn(adapter); err != nil {
			fmt.Printf("    FAIL: %v\n", err)
			failures++
		} else {
			fmt.Printf("    PASS\n")
		}
		adapter.Release()
	}
	if tested == 0 {
		log.Fatal("FAIL: no adapters found")
	}
	if failures > 0 {
		log.Fatalf("FAIL: %d/%d adapters failed", failures, tested)
	}
	fmt.Printf("\nPASS: pure-Go WebGPU compute verified on %d adapter(s)\n", tested)
}

func runOn(adapter *wgpu.Adapter) error {
	device, err := adapter.RequestDevice(nil)
	if err != nil {
		return fmt.Errorf("RequestDevice: %w", err)
	}
	defer device.Release()

	// Input: deterministic pseudo-random u32s.
	in := make([]uint32, n)
	s := uint32(0xdead4070)
	for i := range in {
		s = s*1664525 + 1013904223
		in[i] = s
	}
	inBytes := make([]byte, n*4)
	for i, v := range in {
		binary.LittleEndian.PutUint32(inBytes[i*4:], v)
	}

	mk := func(label string, size uint64, usage wgpu.BufferUsage) (*wgpu.Buffer, error) {
		return device.CreateBuffer(&wgpu.BufferDescriptor{Label: label, Size: size, Usage: usage})
	}
	inBuf, err := mk("in", n*4, wgpu.BufferUsageStorage|wgpu.BufferUsageCopyDst)
	if err != nil {
		return err
	}
	defer inBuf.Release()
	outBuf, err := mk("out", n*4, wgpu.BufferUsageStorage|wgpu.BufferUsageCopySrc)
	if err != nil {
		return err
	}
	defer outBuf.Release()
	stgBuf, err := mk("stg", n*4, wgpu.BufferUsageCopyDst|wgpu.BufferUsageMapRead)
	if err != nil {
		return err
	}
	defer stgBuf.Release()
	uniBuf, err := mk("uni", 4, wgpu.BufferUsageUniform|wgpu.BufferUsageCopyDst)
	if err != nil {
		return err
	}
	defer uniBuf.Release()

	uni := make([]byte, 4)
	binary.LittleEndian.PutUint32(uni, n)
	if err := device.Queue().WriteBuffer(inBuf, 0, inBytes); err != nil {
		return err
	}
	if err := device.Queue().WriteBuffer(uniBuf, 0, uni); err != nil {
		return err
	}

	shader, err := device.CreateShaderModule(&wgpu.ShaderModuleDescriptor{Label: "probe", WGSL: probeWGSL})
	if err != nil {
		return fmt.Errorf("CreateShaderModule: %w", err)
	}
	defer shader.Release()

	bgl, err := device.CreateBindGroupLayout(&wgpu.BindGroupLayoutDescriptor{
		Label: "probe-bgl",
		Entries: []wgpu.BindGroupLayoutEntry{
			{Binding: 0, Visibility: wgpu.ShaderStageCompute, Buffer: &gputypes.BufferBindingLayout{Type: gputypes.BufferBindingTypeReadOnlyStorage}},
			{Binding: 1, Visibility: wgpu.ShaderStageCompute, Buffer: &gputypes.BufferBindingLayout{Type: gputypes.BufferBindingTypeStorage}},
			{Binding: 2, Visibility: wgpu.ShaderStageCompute, Buffer: &gputypes.BufferBindingLayout{Type: gputypes.BufferBindingTypeUniform}},
		},
	})
	if err != nil {
		return err
	}
	defer bgl.Release()

	bg, err := device.CreateBindGroup(&wgpu.BindGroupDescriptor{
		Label: "probe-bg", Layout: bgl,
		Entries: []wgpu.BindGroupEntry{
			{Binding: 0, Buffer: inBuf, Size: n * 4},
			{Binding: 1, Buffer: outBuf, Size: n * 4},
			{Binding: 2, Buffer: uniBuf, Size: 4},
		},
	})
	if err != nil {
		return err
	}
	defer bg.Release()

	pl, err := device.CreatePipelineLayout(&wgpu.PipelineLayoutDescriptor{Label: "probe-pl", BindGroupLayouts: []*wgpu.BindGroupLayout{bgl}})
	if err != nil {
		return err
	}
	defer pl.Release()

	pipe, err := device.CreateComputePipeline(&wgpu.ComputePipelineDescriptor{Label: "probe-pipe", Layout: pl, Module: shader, EntryPoint: "main"})
	if err != nil {
		return fmt.Errorf("CreateComputePipeline: %w", err)
	}
	defer pipe.Release()

	t0 := time.Now()
	enc, err := device.CreateCommandEncoder(nil)
	if err != nil {
		return err
	}
	pass, err := enc.BeginComputePass(nil)
	if err != nil {
		return err
	}
	pass.SetPipeline(pipe)
	pass.SetBindGroup(0, bg, nil)
	pass.Dispatch((n+255)/256, 1, 1)
	if err := pass.End(); err != nil {
		return err
	}
	enc.CopyBufferToBuffer(outBuf, 0, stgBuf, 0, n*4)
	cmd, err := enc.Finish()
	if err != nil {
		return err
	}
	if _, err := device.Queue().Submit(cmd); err != nil {
		return fmt.Errorf("Submit: %w", err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	if err := stgBuf.Map(ctx, wgpu.MapModeRead, 0, n*4); err != nil {
		return fmt.Errorf("Map: %w", err)
	}
	rng, err := stgBuf.MappedRange(0, n*4)
	if err != nil {
		_ = stgBuf.Unmap()
		return err
	}
	got := rng.Bytes()
	want := cpuRef(in)
	mismatches := 0
	firstBad := -1
	for i := 0; i < n; i++ {
		if binary.LittleEndian.Uint32(got[i*4:]) != want[i] {
			mismatches++
			if firstBad < 0 {
				firstBad = i
			}
		}
	}
	if err := stgBuf.Unmap(); err != nil {
		return err
	}
	fmt.Printf("    dispatch+readback: %v for %d elements\n", time.Since(t0), n)

	if mismatches != 0 {
		return fmt.Errorf("%d/%d mismatches (first at %d)", mismatches, n, firstBad)
	}
	return nil
}
