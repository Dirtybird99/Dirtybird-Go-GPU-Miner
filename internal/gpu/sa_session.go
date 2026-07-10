package gpu

import (
	"context"
	"fmt"
	"time"
	"unsafe"

	"github.com/gogpu/gputypes"
	"github.com/gogpu/wgpu"
)

// saSession owns the GPU buffers, pipeline and bind group for the batched
// suffix-array kernel, sized for a fixed slab of `slab` hashes. Reusing it
// across dispatches avoids re-allocating ~430 MB of buffers (and a staging
// buffer) on every call — the dominant host-side overhead of the per-call
// KernelRun path. Rebuilt only when the slab size or kernel (refine vs plain)
// changes.
type saSession struct {
	dev    *wgpu.Device
	refine bool
	slab   int

	text, lens *wgpu.Buffer
	outs       []*wgpu.Buffer // [0]=sa (read back); rest are scratch
	uniform    *wgpu.Buffer
	staging    *wgpu.Buffer // sa rows, mappable for readback

	shader   *wgpu.ShaderModule
	bgl      *wgpu.BindGroupLayout
	bg       *wgpu.BindGroup
	pl       *wgpu.PipelineLayout
	pipe     *wgpu.ComputePipeline
	rowBytes uint64
}

func (s *saSession) release() {
	if s == nil {
		return
	}
	// Guard concrete pointers (a nil pointer boxed into an interface is not nil).
	for _, b := range append([]*wgpu.Buffer{s.staging, s.uniform, s.text, s.lens}, s.outs...) {
		if b != nil {
			b.Release()
		}
	}
	if s.pipe != nil {
		s.pipe.Release()
	}
	if s.pl != nil {
		s.pl.Release()
	}
	if s.bg != nil {
		s.bg.Release()
	}
	if s.bgl != nil {
		s.bgl.Release()
	}
	if s.shader != nil {
		s.shader.Release()
	}
}

// numOuts is the number of read_write storage buffers each kernel binds.
func numOuts(refine bool) int {
	if refine {
		return 7 // sa, rank, rankTmp, av, avTmp, active, keyTmp
	}
	return 4 // sa, sa2, rank, rank2
}

func (c *Ctx) buildSASession(slab int, refine bool) (_ *saSession, retErr error) {
	dev := c.Device
	rowBytes := uint64(saStride * 4)
	textWords := saStride / 4
	s := &saSession{dev: dev, refine: refine, slab: slab, rowBytes: rowBytes}
	defer func() {
		if retErr != nil {
			s.release()
		}
	}()

	mk := func(label string, size uint64, usage wgpu.BufferUsage) (*wgpu.Buffer, error) {
		return dev.CreateBuffer(&wgpu.BufferDescriptor{Label: label, Size: size, Usage: usage})
	}
	var err error
	if s.text, err = mk("sa-text", uint64(slab*textWords*4), wgpu.BufferUsageStorage|wgpu.BufferUsageCopyDst); err != nil {
		return nil, err
	}
	if s.lens, err = mk("sa-lens", uint64(slab*4), wgpu.BufferUsageStorage|wgpu.BufferUsageCopyDst); err != nil {
		return nil, err
	}
	no := numOuts(refine)
	s.outs = make([]*wgpu.Buffer, no)
	for i := 0; i < no; i++ {
		usage := wgpu.BufferUsageStorage
		if i == 0 {
			usage |= wgpu.BufferUsageCopySrc // sa is read back
		}
		if s.outs[i], err = mk(fmt.Sprintf("sa-out%d", i), uint64(slab)*rowBytes, usage); err != nil {
			return nil, err
		}
	}
	if s.uniform, err = mk("sa-uniform", 16, wgpu.BufferUsageUniform|wgpu.BufferUsageCopyDst); err != nil {
		return nil, err
	}
	if s.staging, err = mk("sa-staging", uint64(slab)*rowBytes, wgpu.BufferUsageCopyDst|wgpu.BufferUsageMapRead); err != nil {
		return nil, err
	}

	wgsl, entry := SAGpuWGSL, "sa_batch_main"
	if refine {
		wgsl, entry = SARefineWGSL, "sa_refine_main"
	}
	if s.shader, err = dev.CreateShaderModule(&wgpu.ShaderModuleDescriptor{Label: "sa-shader", WGSL: wgsl}); err != nil {
		return nil, fmt.Errorf("CreateShaderModule: %w", err)
	}

	// Bindings: 0=text(ro), 1=lens(ro), 2..=rw outs, then uniform.
	var le []wgpu.BindGroupLayoutEntry
	var be []wgpu.BindGroupEntry
	ro := func(b uint32, buf *wgpu.Buffer, sz uint64) {
		le = append(le, wgpu.BindGroupLayoutEntry{Binding: b, Visibility: wgpu.ShaderStageCompute,
			Buffer: &gputypes.BufferBindingLayout{Type: gputypes.BufferBindingTypeReadOnlyStorage}})
		be = append(be, wgpu.BindGroupEntry{Binding: b, Buffer: buf, Size: sz})
	}
	rw := func(b uint32, buf *wgpu.Buffer, sz uint64) {
		le = append(le, wgpu.BindGroupLayoutEntry{Binding: b, Visibility: wgpu.ShaderStageCompute,
			Buffer: &gputypes.BufferBindingLayout{Type: gputypes.BufferBindingTypeStorage}})
		be = append(be, wgpu.BindGroupEntry{Binding: b, Buffer: buf, Size: sz})
	}
	ro(0, s.text, uint64(slab*textWords*4))
	ro(1, s.lens, uint64(slab*4))
	for i := 0; i < no; i++ {
		rw(uint32(2+i), s.outs[i], uint64(slab)*rowBytes)
	}
	ub := uint32(2 + no)
	le = append(le, wgpu.BindGroupLayoutEntry{Binding: ub, Visibility: wgpu.ShaderStageCompute,
		Buffer: &gputypes.BufferBindingLayout{Type: gputypes.BufferBindingTypeUniform}})
	be = append(be, wgpu.BindGroupEntry{Binding: ub, Buffer: s.uniform, Size: 16})

	if s.bgl, err = dev.CreateBindGroupLayout(&wgpu.BindGroupLayoutDescriptor{Label: "sa-bgl", Entries: le}); err != nil {
		return nil, err
	}
	if s.bg, err = dev.CreateBindGroup(&wgpu.BindGroupDescriptor{Label: "sa-bg", Layout: s.bgl, Entries: be}); err != nil {
		return nil, err
	}
	if s.pl, err = dev.CreatePipelineLayout(&wgpu.PipelineLayoutDescriptor{Label: "sa-pl", BindGroupLayouts: []*wgpu.BindGroupLayout{s.bgl}}); err != nil {
		return nil, err
	}
	if s.pipe, err = dev.CreateComputePipeline(&wgpu.ComputePipelineDescriptor{Label: "sa-pipe", Layout: s.pl, Module: s.shader, EntryPoint: entry}); err != nil {
		return nil, fmt.Errorf("CreateComputePipeline: %w", err)
	}
	return s, nil
}

// run uploads text/lens, dispatches one workgroup per hash, and reads back the
// sa rows for `n` hashes into saBytes (packed, lens[i] entries per hash).
// Each hash's bytes are uploaded directly to its strided row — no host-side
// pack buffer and no padding traffic. Bytes past lens[i] in a row are stale
// from earlier dispatches, which is fine: the kernels never read past lens[i]
// (little-endian byte packing == byte i in word i>>2).
func (s *saSession) run(datas [][]byte, lens []uint32, n int) ([]byte, error) {
	dev := s.dev
	for i, d := range datas {
		off := uint64(i) * uint64(saStride) // text rows are saStride bytes, not rowBytes
		m := len(d) &^ 3                    // WriteBuffer sizes must be 4-byte multiples
		if m > 0 {
			if err := dev.Queue().WriteBuffer(s.text, off, d[:m]); err != nil {
				return nil, err
			}
		}
		if r := len(d) - m; r > 0 {
			var w [4]byte
			copy(w[:], d[m:])
			if err := dev.Queue().WriteBuffer(s.text, off+uint64(m), w[:]); err != nil {
				return nil, err
			}
		}
	}
	if err := dev.Queue().WriteBuffer(s.lens, 0, U32s(lens...)); err != nil {
		return nil, err
	}
	if err := dev.Queue().WriteBuffer(s.uniform, 0, U32s(uint32(n), uint32(saStride), SADebugRounds)); err != nil {
		return nil, err
	}

	enc, err := dev.CreateCommandEncoder(nil)
	if err != nil {
		return nil, err
	}
	pass, err := enc.BeginComputePass(nil)
	if err != nil {
		return nil, err
	}
	pass.SetPipeline(s.pipe)
	pass.SetBindGroup(0, s.bg, nil)
	pass.Dispatch(uint32(n), 1, 1)
	if err := pass.End(); err != nil {
		return nil, err
	}
	// Pack only the meaningful prefix of each SA row (lens[i] entries) into
	// the staging buffer, back to back — the rest of each 70912-entry row is
	// garbage and not worth the PCIe traffic.
	var readBytes uint64
	for i := 0; i < n; i++ {
		sz := uint64(lens[i]) * 4
		if sz == 0 {
			continue
		}
		enc.CopyBufferToBuffer(s.outs[0], uint64(i)*s.rowBytes, s.staging, readBytes, sz)
		readBytes += sz
	}
	cmd, err := enc.Finish()
	if err != nil {
		return nil, err
	}
	if _, err := dev.Queue().Submit(cmd); err != nil {
		return nil, err
	}

	if readBytes == 0 {
		return nil, nil
	}
	ctx, cancel := context.WithTimeout(context.Background(), 120*time.Second)
	defer cancel()
	if err := s.staging.Map(ctx, wgpu.MapModeRead, 0, readBytes); err != nil {
		return nil, fmt.Errorf("map staging: %w", err)
	}
	rng, err := s.staging.MappedRange(0, readBytes)
	if err != nil {
		_ = s.staging.Unmap()
		return nil, err
	}
	out := make([]byte, readBytes)
	copy(out, rng.Bytes())
	if err := s.staging.Unmap(); err != nil {
		return nil, err
	}
	return out, nil
}

// decodeSA views the per-hash int32 suffix arrays over the packed read-back
// bytes (lens[i] entries per hash, back to back). Rows are disjoint views of
// the one buffer, so callers see isolated slices without a copy or per-element
// decode. Little-endian host assumed, like the rest of the GPU byte packing.
func decodeSA(saBytes []byte, lens []uint32) [][]int32 {
	res := make([][]int32, len(lens))
	if len(saBytes) == 0 {
		return res
	}
	words := unsafe.Slice((*int32)(unsafe.Pointer(&saBytes[0])), len(saBytes)/4)
	base := 0
	for i := range lens {
		n := int(lens[i])
		res[i] = words[base : base+n : base+n]
		base += n
	}
	return res
}
