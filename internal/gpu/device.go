// Package gpu wraps the pure-Go WebGPU stack (gogpu/wgpu) behind the small
// surface the miner needs: device setup, buffer plumbing, and one-shot
// compute dispatches for kernel parity tests.
package gpu

import (
	"context"
	"encoding/binary"
	"fmt"
	"time"

	"github.com/gogpu/gputypes"
	"github.com/gogpu/wgpu"

	_ "github.com/gogpu/wgpu/hal/allbackends"
)

// Power selects which adapter class to request.
type Power int

const (
	PowerDefault Power = iota
	PowerHigh
	PowerLow
	PowerSoftware
)

// Backend selects which HAL backend the instance may use. BackendAuto lets
// wgpu pick (the shipping default); the rest pin one backend so a single
// machine can verify more than one of them. On Windows, hal/allbackends
// registers vulkan, dx12 and gles; software is registered on every platform.
type Backend int

const (
	BackendAuto Backend = iota
	BackendVulkan
	BackendDX12
	BackendGL
	BackendSoftware
)

// ParseBackend maps a CLI name to a Backend. Unknown names yield BackendAuto.
func ParseBackend(s string) Backend {
	switch s {
	case "vulkan":
		return BackendVulkan
	case "dx12":
		return BackendDX12
	case "gl", "gles":
		return BackendGL
	case "software":
		return BackendSoftware
	default:
		return BackendAuto
	}
}

// Ctx owns an instance/adapter/device triple.
type Ctx struct {
	instance *wgpu.Instance
	adapter  *wgpu.Adapter
	Device   *wgpu.Device
	Name     string

	// Backend is the backend wgpu actually selected, which is what a
	// verification record must report (BackendAuto resolves to whatever wins).
	Backend gputypes.Backend

	// MaxBufferSize is the device's per-allocation buffer cap, in bytes.
	//
	// MaxStorageBindingSize is the cap on a single *storage binding*, which is
	// what actually bounds the SA slab (every SA row is bound as storage). The
	// two differ wildly and the smaller one wins: DX12 advertises
	// MaxBufferSize = 128 GB while granting a 128 MiB storage binding, so sizing
	// a slab off MaxBufferSize would ask for a buffer ~1000x larger than the
	// device will bind.
	//
	// MaxStorageBuffers is the device's storage-buffers-per-shader-stage cap. The
	// fused pipeline binds 9; wgpu's default is 8, so a device that falls back to
	// default limits cannot build a fused session at all.
	//
	// All three are read from the *device* (granted), never from the adapter
	// (advertised) — RequestDevice may hand back fewer limits than it was asked
	// for, and the fallback branch below asks for none at all.
	MaxBufferSize         uint64
	MaxStorageBindingSize uint64
	MaxStorageBuffers     uint32

	// useRefine selects the active-set-compaction SA kernel over plain prefix
	// doubling. Set via UseRefine; defaults off until the miner enables it.
	useRefine bool
	// useSPIRV selects the fixed-slab NVIDIA/Vulkan native fast path.
	useSPIRV bool

	// saSess is the cached persistent SA buffer/pipeline session, rebuilt only
	// when the slab size or kernel changes.
	saSess *saSession
	// fused is the cached front-half+SA fused session (miner path).
	fused *fusedSession
	// fusedPipe holds the two ping-ponging WGSL streaming sessions.
	fusedPipe [2]*fusedSession
	// spirvFast is the fixed 2048-hash native streaming session.
	spirvFast *spirvFastStreamer
}

// UseRefine enables the active-set-compaction suffix-array kernel.
func (c *Ctx) UseRefine(v bool) { c.useRefine = v }

// UseSPIRV enables the embedded Vulkan suffix-array fast path. Callers must
// gate it with CanSPIRV and a measured subgroup size of exactly 32.
func (c *Ctx) UseSPIRV(v bool) { c.useSPIRV = v }

// CanSPIRV reports the static adapter and buffer requirements of the embedded
// NVIDIA Vulkan kernels. Subgroup size is intentionally checked by the caller
// using a throwaway device because probing can remove unsupported devices.
func (c *Ctx) CanSPIRV() bool {
	info := c.Info()
	nvidia := info.VendorID == 0x10de || info.Vendor == "NVIDIA"
	buffersOK := c.MaxStorageBuffers == 0 || c.MaxStorageBuffers >= 8
	required := max(spirvDescriptorBytes, spirvStreamSABytes)
	bindingOK := c.MaxStorageBindingSize == 0 || c.MaxStorageBindingSize >= required
	allocationOK := c.MaxBufferSize == 0 || c.MaxBufferSize >= required
	return c.Backend == gputypes.BackendVulkan && nvidia && buffersOK && bindingOK && allocationOK
}

// FusedStorageBuffers is how many storage buffers the fused SA bind group binds
// (fused.go, bindings 0..8). wgpu's default limit is 8, so a device that fell
// back to default limits cannot build a fused session.
const FusedStorageBuffers = 9

// MaxSlab is the largest batch whose SA row buffer still fits one storage
// binding on this device, or 0 if the device reported no limit. On DX12 this is
// 473 — below the miner's default batch of 512 — so callers must size to it
// rather than assume the default works everywhere.
func (c *Ctx) MaxSlab() int {
	if c.MaxStorageBindingSize == 0 {
		return 0
	}
	return int(c.MaxStorageBindingSize / (saStride * 4))
}

// CanFuse reports whether the device grants enough storage buffers per stage for
// the fused pipeline. It says nothing about whether the kernels compute
// correctly here — only SelfTest can answer that.
func (c *Ctx) CanFuse() bool {
	return c.MaxStorageBuffers == 0 || c.MaxStorageBuffers >= FusedStorageBuffers
}

// ProbeSubgroupSize reports the subgroup size a device opened with (p, b)
// would have, probing on a THROWAWAY device. On backends without subgroup
// support the probe dispatch can remove the device itself (DX12 dies with
// DXGI_ERROR_INVALID_CALL rather than failing the compile), which would poison
// every later dispatch — so the miner must never run the probe on the device
// it intends to mine with. No HAL populates the FeatureSubgroup* flags in wgpu
// v0.30.10, so dispatching a probe kernel is the only way to learn the size.
func ProbeSubgroupSize(p Power, b Backend) uint32 {
	scratch, err := NewCtxBackend(p, b)
	if err != nil {
		return 0
	}
	defer scratch.Close()
	return scratch.SubgroupSize()
}

// SubgroupSize returns the device's subgroup (warp/wave) size, or 0 if
// subgroups are unavailable on this backend. The subgroup-accelerated SA
// (sa_refine.wgsl / the fused path) requires size >= 32; below that its
// cross-subgroup histogram overflows and silently corrupts the sort, so
// callers must fall back to the subgroup-free kernel (sa_gpu.wgsl).
//
// CAUTION: on a backend without subgroup support the probe dispatch can REMOVE
// THE DEVICE (observed on DX12). Use ProbeSubgroupSize from anything that will
// keep using its device afterwards.
func (c *Ctx) SubgroupSize() uint32 {
	const probe = `enable subgroups;
@group(0) @binding(0) var<storage, read_write> o: array<u32>;
@compute @workgroup_size(64)
fn main(@builtin(subgroup_size) s: u32, @builtin(local_invocation_id) l: vec3<u32>) {
  if (l.x == 0u) { o[0] = s; }
}`
	out, err := c.Run(KernelRun{WGSL: probe, Entry: "main", Outputs: []int{4}, Groups: [3]uint32{1, 1, 1}})
	if err != nil {
		return 0 // backend does not support subgroups
	}
	return binary.LittleEndian.Uint32(out[0])
}

// NewCtx opens a device on the requested adapter class, letting wgpu choose the
// backend. This is the shipping path.
func NewCtx(p Power) (*Ctx, error) { return NewCtxBackend(p, BackendAuto) }

// NewCtxBackend opens a device on the requested adapter class, pinned to one
// backend. Pinning is what lets a single machine verify more than one backend;
// BackendAuto reproduces NewCtx exactly.
func NewCtxBackend(p Power, b Backend) (*Ctx, error) {
	var desc *wgpu.InstanceDescriptor
	switch b {
	case BackendVulkan:
		desc = &wgpu.InstanceDescriptor{Backends: gputypes.BackendsVulkan}
	case BackendDX12:
		desc = &wgpu.InstanceDescriptor{Backends: gputypes.BackendsDX12}
	case BackendGL:
		desc = &wgpu.InstanceDescriptor{Backends: gputypes.BackendsGL}
	case BackendSoftware:
		// The software HAL registers as BackendEmpty.
		desc = &wgpu.InstanceDescriptor{Backends: wgpu.Backends(1 << gputypes.BackendEmpty)}
	}
	instance, err := wgpu.CreateInstance(desc)
	if err != nil {
		return nil, fmt.Errorf("CreateInstance: %w", err)
	}
	var opts *wgpu.RequestAdapterOptions
	switch p {
	case PowerHigh:
		opts = &wgpu.RequestAdapterOptions{PowerPreference: wgpu.PowerPreferenceHighPerformance}
	case PowerLow:
		opts = &wgpu.RequestAdapterOptions{PowerPreference: wgpu.PowerPreferenceLowPower}
	case PowerSoftware:
		opts = &wgpu.RequestAdapterOptions{ForceFallbackAdapter: true}
	}
	if b == BackendSoftware {
		// An explicit -backend software wins over any power preference.
		opts = &wgpu.RequestAdapterOptions{ForceFallbackAdapter: true}
	}
	adapter, err := instance.RequestAdapter(opts)
	if err != nil {
		instance.Release()
		return nil, fmt.Errorf("RequestAdapter: %w", err)
	}
	// wgpu leaves MaxBufferSize at its 256 MiB default no matter what the device
	// can do (the Vulkan backend never overrides it), and that default — not the
	// hardware — is what caps the SA slab at ~940 hashes. Ask for the real
	// per-binding ceiling instead. Start from the adapter's own limits so the
	// values the backend DID derive (workgroup storage, binding size) survive.
	lim := adapter.Limits()
	if lim.MaxStorageBufferBindingSize > lim.MaxBufferSize {
		lim.MaxBufferSize = lim.MaxStorageBufferBindingSize
	}
	device, err := adapter.RequestDevice(&wgpu.DeviceDescriptor{RequiredLimits: lim})
	if err != nil {
		// A backend may refuse the raised limit; fall back to its defaults
		// rather than failing to mine at all.
		device, err = adapter.RequestDevice(nil)
		if err != nil {
			adapter.Release()
			instance.Release()
			return nil, fmt.Errorf("RequestDevice: %w", err)
		}
	}
	// Read back what the device GRANTED. On the fallback branch above it granted
	// wgpu's defaults, which are far below what the adapter advertised; reporting
	// the adapter's numbers there would let the SA slab guard wave through a
	// buffer the device will refuse to bind.
	got := device.Limits()
	return &Ctx{
		instance: instance, adapter: adapter, Device: device,
		Name: adapter.Info().Name, Backend: adapter.Info().Backend,
		MaxBufferSize:         got.MaxBufferSize,
		MaxStorageBindingSize: got.MaxStorageBufferBindingSize,
		MaxStorageBuffers:     got.MaxStorageBuffersPerShaderStage,
	}, nil
}

// Info returns the adapter's identity, for verification records.
func (c *Ctx) Info() gputypes.AdapterInfo { return c.adapter.Info() }

// Close releases the device, adapter and instance.
func (c *Ctx) Close() {
	if c.spirvFast != nil {
		c.spirvFast.release()
		c.spirvFast = nil
	}
	if c.saSess != nil {
		c.saSess.release()
		c.saSess = nil
	}
	if c.fused != nil {
		c.fused.release()
		c.fused = nil
	}
	for i := range c.fusedPipe {
		if c.fusedPipe[i] != nil {
			c.fusedPipe[i].release()
			c.fusedPipe[i] = nil
		}
	}
	c.Device.Release()
	c.adapter.Release()
	c.instance.Release()
}

// KernelRun describes a one-shot compute dispatch for tests and prototypes.
//
// Binding convention (group 0): bindings 0..len(Inputs)-1 are read-only
// storage buffers holding Inputs; the next len(Outputs) bindings are
// read_write storage buffers sized by Outputs; if Uniform is non-nil it is
// the final binding as a uniform buffer. Kernel WGSL must match this order.
type KernelRun struct {
	WGSL    string
	Entry   string
	Inputs  [][]byte
	Outputs []int
	Uniform []byte
	Groups  [3]uint32
	// InitOutputs seeds read_write output buffers before dispatch, keyed by
	// output index. Used for in-place kernels (e.g. sort) whose input and
	// output are the same storage buffer.
	InitOutputs map[int][]byte
}

// Run executes the kernel once and returns the contents of each output buffer.
func (c *Ctx) Run(kr KernelRun) ([][]byte, error) {
	dev := c.Device
	if kr.Entry == "" {
		kr.Entry = "main"
	}

	shader, err := dev.CreateShaderModule(&wgpu.ShaderModuleDescriptor{Label: "kr-shader", WGSL: kr.WGSL})
	if err != nil {
		return nil, fmt.Errorf("CreateShaderModule: %w", err)
	}
	defer shader.Release()

	var layoutEntries []wgpu.BindGroupLayoutEntry
	var bindEntries []wgpu.BindGroupEntry
	var release []func()
	defer func() {
		for i := len(release) - 1; i >= 0; i-- {
			release[i]()
		}
	}()
	binding := uint32(0)

	for i, in := range kr.Inputs {
		buf, err := dev.CreateBuffer(&wgpu.BufferDescriptor{
			Label: fmt.Sprintf("in%d", i), Size: uint64(len(in)),
			Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst,
		})
		if err != nil {
			return nil, fmt.Errorf("create input %d: %w", i, err)
		}
		release = append(release, buf.Release)
		if err := dev.Queue().WriteBuffer(buf, 0, in); err != nil {
			return nil, fmt.Errorf("write input %d: %w", i, err)
		}
		layoutEntries = append(layoutEntries, wgpu.BindGroupLayoutEntry{
			Binding: binding, Visibility: wgpu.ShaderStageCompute,
			Buffer: &gputypes.BufferBindingLayout{Type: gputypes.BufferBindingTypeReadOnlyStorage},
		})
		bindEntries = append(bindEntries, wgpu.BindGroupEntry{Binding: binding, Buffer: buf, Size: uint64(len(in))})
		binding++
	}

	outBufs := make([]*wgpu.Buffer, len(kr.Outputs))
	for i, size := range kr.Outputs {
		buf, err := dev.CreateBuffer(&wgpu.BufferDescriptor{
			Label: fmt.Sprintf("out%d", i), Size: uint64(size),
			Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopySrc | wgpu.BufferUsageCopyDst,
		})
		if err != nil {
			return nil, fmt.Errorf("create output %d: %w", i, err)
		}
		release = append(release, buf.Release)
		outBufs[i] = buf
		if init, ok := kr.InitOutputs[i]; ok {
			if err := dev.Queue().WriteBuffer(buf, 0, init); err != nil {
				return nil, fmt.Errorf("init output %d: %w", i, err)
			}
		}
		layoutEntries = append(layoutEntries, wgpu.BindGroupLayoutEntry{
			Binding: binding, Visibility: wgpu.ShaderStageCompute,
			Buffer: &gputypes.BufferBindingLayout{Type: gputypes.BufferBindingTypeStorage},
		})
		bindEntries = append(bindEntries, wgpu.BindGroupEntry{Binding: binding, Buffer: buf, Size: uint64(size)})
		binding++
	}

	if kr.Uniform != nil {
		buf, err := dev.CreateBuffer(&wgpu.BufferDescriptor{
			Label: "uniform", Size: uint64(len(kr.Uniform)),
			Usage: wgpu.BufferUsageUniform | wgpu.BufferUsageCopyDst,
		})
		if err != nil {
			return nil, fmt.Errorf("create uniform: %w", err)
		}
		release = append(release, buf.Release)
		if err := dev.Queue().WriteBuffer(buf, 0, kr.Uniform); err != nil {
			return nil, fmt.Errorf("write uniform: %w", err)
		}
		layoutEntries = append(layoutEntries, wgpu.BindGroupLayoutEntry{
			Binding: binding, Visibility: wgpu.ShaderStageCompute,
			Buffer: &gputypes.BufferBindingLayout{Type: gputypes.BufferBindingTypeUniform},
		})
		bindEntries = append(bindEntries, wgpu.BindGroupEntry{Binding: binding, Buffer: buf, Size: uint64(len(kr.Uniform))})
	}

	bgl, err := dev.CreateBindGroupLayout(&wgpu.BindGroupLayoutDescriptor{Label: "kr-bgl", Entries: layoutEntries})
	if err != nil {
		return nil, fmt.Errorf("CreateBindGroupLayout: %w", err)
	}
	defer bgl.Release()
	bg, err := dev.CreateBindGroup(&wgpu.BindGroupDescriptor{Label: "kr-bg", Layout: bgl, Entries: bindEntries})
	if err != nil {
		return nil, fmt.Errorf("CreateBindGroup: %w", err)
	}
	defer bg.Release()
	pl, err := dev.CreatePipelineLayout(&wgpu.PipelineLayoutDescriptor{Label: "kr-pl", BindGroupLayouts: []*wgpu.BindGroupLayout{bgl}})
	if err != nil {
		return nil, fmt.Errorf("CreatePipelineLayout: %w", err)
	}
	defer pl.Release()
	pipe, err := dev.CreateComputePipeline(&wgpu.ComputePipelineDescriptor{Label: "kr-pipe", Layout: pl, Module: shader, EntryPoint: kr.Entry})
	if err != nil {
		return nil, fmt.Errorf("CreateComputePipeline: %w", err)
	}
	defer pipe.Release()

	enc, err := dev.CreateCommandEncoder(nil)
	if err != nil {
		return nil, fmt.Errorf("CreateCommandEncoder: %w", err)
	}
	pass, err := enc.BeginComputePass(nil)
	if err != nil {
		return nil, fmt.Errorf("BeginComputePass: %w", err)
	}
	pass.SetPipeline(pipe)
	pass.SetBindGroup(0, bg, nil)
	pass.Dispatch(kr.Groups[0], max(1, kr.Groups[1]), max(1, kr.Groups[2]))
	if err := pass.End(); err != nil {
		return nil, fmt.Errorf("End: %w", err)
	}

	stagings := make([]*wgpu.Buffer, len(kr.Outputs))
	for i, size := range kr.Outputs {
		stg, err := dev.CreateBuffer(&wgpu.BufferDescriptor{
			Label: fmt.Sprintf("stg%d", i), Size: uint64(size),
			Usage: wgpu.BufferUsageCopyDst | wgpu.BufferUsageMapRead,
		})
		if err != nil {
			return nil, fmt.Errorf("create staging %d: %w", i, err)
		}
		release = append(release, stg.Release)
		stagings[i] = stg
		enc.CopyBufferToBuffer(outBufs[i], 0, stg, 0, uint64(size))
	}

	cmd, err := enc.Finish()
	if err != nil {
		return nil, fmt.Errorf("Finish: %w", err)
	}
	if _, err := dev.Queue().Submit(cmd); err != nil {
		return nil, fmt.Errorf("Submit: %w", err)
	}

	results := make([][]byte, len(kr.Outputs))
	for i, stg := range stagings {
		size := uint64(kr.Outputs[i])
		ctx, cancel := context.WithTimeout(context.Background(), 120*time.Second)
		err := stg.Map(ctx, wgpu.MapModeRead, 0, size)
		cancel()
		if err != nil {
			return nil, fmt.Errorf("map staging %d: %w", i, err)
		}
		rng, err := stg.MappedRange(0, size)
		if err != nil {
			_ = stg.Unmap()
			return nil, fmt.Errorf("MappedRange %d: %w", i, err)
		}
		out := make([]byte, size)
		copy(out, rng.Bytes())
		if err := stg.Unmap(); err != nil {
			return nil, fmt.Errorf("unmap staging %d: %w", i, err)
		}
		results[i] = out
	}
	return results, nil
}

// U32s packs uint32 words little-endian, the layout WGSL storage buffers use.
func U32s(vals ...uint32) []byte {
	b := make([]byte, 4*len(vals))
	for i, v := range vals {
		binary.LittleEndian.PutUint32(b[i*4:], v)
	}
	return b
}
