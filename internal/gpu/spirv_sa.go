package gpu

import (
	"embed"
	"encoding/binary"
	"fmt"

	"github.com/gogpu/gputypes"
	"github.com/gogpu/wgpu"
)

const (
	spirvRows        = 512
	spirvTilesPerRow = 19
	spirvPairBytes   = uint64(spirvRows * saStride * 4)
	spirvTotalTiles  = uint32(spirvRows * spirvTilesPerRow)
	spirvAuxBytes    = uint64(spirvTotalTiles) * 256 * 4
	spirvControlSize = uint64(256)
	spirvDirectDesc  = uint64(spirvRows*saStride/2+1) +
		uint64(spirvRows*saStride/3+1) +
		uint64(spirvRows*saStride/9+1) +
		uint64(spirvRows*saStride/17+1) +
		uint64(spirvRows*saStride/33+1) +
		uint64(spirvRows*saStride/129+1) +
		uint64(spirvRows*saStride/513+1)
	spirvDescriptorBytes = spirvDirectDesc * 8
)

//go:embed spirv/*.spv
var spirvFiles embed.FS

type spirvRadixPass struct {
	histogram *wgpu.ComputePipeline
	scatter   *wgpu.ComputePipeline
	bind      *wgpu.BindGroup
}

// spirvSA is the measured Vulkan fast path: four stable radix passes over a
// three-symbol key, followed by direct LCP refinement of the tied runs.
type spirvSA struct {
	hist, offsets, descriptors, control *wgpu.Buffer
	sortBGL, refineBGL                  *wgpu.BindGroupLayout
	sortLayout, refineLayout            *wgpu.PipelineLayout
	bindAB, bindBA, refineBind          *wgpu.BindGroup
	pipelines                           []*wgpu.ComputePipeline
	init, scan, detect, prepare         *wgpu.ComputePipeline
	radix                               [4]spirvRadixPass
	tiny                                [4]*wgpu.ComputePipeline
	large                               [3]*wgpu.ComputePipeline
}

func buildSPIRVSA(s *fusedSession) (_ *spirvSA, retErr error) {
	dev := s.dev
	sp := &spirvSA{}
	defer func() {
		if retErr != nil {
			sp.release()
		}
	}()

	mk := func(label string, size uint64, usage wgpu.BufferUsage) (*wgpu.Buffer, error) {
		return dev.CreateBuffer(&wgpu.BufferDescriptor{Label: label, Size: size, Usage: usage})
	}
	var err error
	if sp.hist, err = mk("spv-hist", spirvAuxBytes, wgpu.BufferUsageStorage); err != nil {
		return nil, err
	}
	if sp.offsets, err = mk("spv-offsets", spirvAuxBytes, wgpu.BufferUsageStorage); err != nil {
		return nil, err
	}
	if sp.descriptors, err = mk("spv-descriptors", spirvDescriptorBytes, wgpu.BufferUsageStorage); err != nil {
		return nil, err
	}
	if sp.control, err = mk("spv-control", spirvControlSize, wgpu.BufferUsageStorage|wgpu.BufferUsageIndirect|wgpu.BufferUsageCopySrc|wgpu.BufferUsageCopyDst); err != nil {
		return nil, err
	}

	makeLayout := func(label string, count int) (*wgpu.BindGroupLayout, error) {
		entries := make([]wgpu.BindGroupLayoutEntry, count)
		for i := range entries {
			entries[i] = wgpu.BindGroupLayoutEntry{
				Binding: uint32(i), Visibility: wgpu.ShaderStageCompute,
				Buffer: &gputypes.BufferBindingLayout{Type: gputypes.BufferBindingTypeStorage},
			}
		}
		return dev.CreateBindGroupLayout(&wgpu.BindGroupLayoutDescriptor{Label: label, Entries: entries})
	}
	makeBind := func(label string, layout *wgpu.BindGroupLayout, buffers ...*wgpu.Buffer) (*wgpu.BindGroup, error) {
		entries := make([]wgpu.BindGroupEntry, len(buffers))
		for i, buffer := range buffers {
			entries[i] = wgpu.BindGroupEntry{Binding: uint32(i), Buffer: buffer, Size: buffer.Size()}
		}
		return dev.CreateBindGroup(&wgpu.BindGroupDescriptor{Label: label, Layout: layout, Entries: entries})
	}

	if sp.sortBGL, err = makeLayout("spv-sort-bgl", 8); err != nil {
		return nil, err
	}
	if sp.sortLayout, err = dev.CreatePipelineLayout(&wgpu.PipelineLayoutDescriptor{
		Label: "spv-sort-layout", BindGroupLayouts: []*wgpu.BindGroupLayout{sp.sortBGL},
	}); err != nil {
		return nil, err
	}
	// rank/rankTmp are A and av/sa are B. Four depth-3 passes finish in A.
	if sp.bindAB, err = makeBind("spv-a-to-b", sp.sortBGL,
		s.rankBuf, s.rankTmpBuf, s.avBuf, s.saBuf,
		sp.hist, sp.offsets, s.textBuf, s.lensBuf); err != nil {
		return nil, err
	}
	if sp.bindBA, err = makeBind("spv-b-to-a", sp.sortBGL,
		s.avBuf, s.saBuf, s.rankBuf, s.rankTmpBuf,
		sp.hist, sp.offsets, s.textBuf, s.lensBuf); err != nil {
		return nil, err
	}

	if sp.refineBGL, err = makeLayout("spv-refine-bgl", 7); err != nil {
		return nil, err
	}
	if sp.refineLayout, err = dev.CreatePipelineLayout(&wgpu.PipelineLayoutDescriptor{
		Label: "spv-refine-layout", BindGroupLayouts: []*wgpu.BindGroupLayout{sp.refineBGL},
	}); err != nil {
		return nil, err
	}
	if sp.refineBind, err = makeBind("spv-refine", sp.refineBGL,
		s.textBuf, s.lensBuf, s.rankBuf, s.rankTmpBuf,
		sp.descriptors, s.avTmpBuf, sp.control); err != nil {
		return nil, err
	}

	makePipeline := func(layout *wgpu.PipelineLayout, name string) (*wgpu.ComputePipeline, error) {
		words, err := embeddedSPIRV(name)
		if err != nil {
			return nil, err
		}
		module, err := dev.CreateShaderModule(&wgpu.ShaderModuleDescriptor{Label: name, SPIRV: words})
		if err != nil {
			return nil, fmt.Errorf("SPIR-V shader %s: %w", name, err)
		}
		defer module.Release()
		pipe, err := dev.CreateComputePipeline(&wgpu.ComputePipelineDescriptor{
			Label: name, Layout: layout, Module: module, EntryPoint: "main",
		})
		if err != nil {
			return nil, fmt.Errorf("SPIR-V pipeline %s: %w", name, err)
		}
		sp.pipelines = append(sp.pipelines, pipe)
		return pipe, nil
	}
	if sp.init, err = makePipeline(sp.sortLayout, "init_0"); err != nil {
		return nil, err
	}
	if sp.scan, err = makePipeline(sp.sortLayout, "scan_0"); err != nil {
		return nil, err
	}
	for i, shift := range []int{0, 8, 16, 24} {
		histogram, err := makePipeline(sp.sortLayout, fmt.Sprintf("histogram_%d", shift))
		if err != nil {
			return nil, err
		}
		scatter, err := makePipeline(sp.sortLayout, fmt.Sprintf("scatter_%d", shift))
		if err != nil {
			return nil, err
		}
		bind := sp.bindAB
		if i&1 != 0 {
			bind = sp.bindBA
		}
		sp.radix[i] = spirvRadixPass{histogram: histogram, scatter: scatter, bind: bind}
	}
	if sp.detect, err = makePipeline(sp.refineLayout, "detectdirect"); err != nil {
		return nil, err
	}
	if sp.prepare, err = makePipeline(sp.refineLayout, "preparedirect"); err != nil {
		return nil, err
	}
	if sp.tiny[0], err = makePipeline(sp.refineLayout, "direct_tiny_0"); err != nil {
		return nil, err
	}
	for i := 1; i < len(sp.tiny); i++ {
		if sp.tiny[i], err = makePipeline(sp.refineLayout, fmt.Sprintf("global_%d", i)); err != nil {
			return nil, err
		}
	}
	for i := range sp.large {
		if sp.large[i], err = makePipeline(sp.refineLayout, fmt.Sprintf("direct_large_%d", i+4)); err != nil {
			return nil, err
		}
	}
	return sp, nil
}

func embeddedSPIRV(name string) ([]uint32, error) {
	code, err := spirvFiles.ReadFile("spirv/" + name + ".spv")
	if err != nil {
		return nil, fmt.Errorf("read embedded SPIR-V %s: %w", name, err)
	}
	if len(code)%4 != 0 || len(code) < 4 || binary.LittleEndian.Uint32(code) != 0x07230203 {
		return nil, fmt.Errorf("invalid embedded SPIR-V %s", name)
	}
	words := make([]uint32, len(code)/4)
	for i := range words {
		words[i] = binary.LittleEndian.Uint32(code[i*4:])
	}
	return words, nil
}

func (s *spirvSA) release() {
	if s == nil {
		return
	}
	for _, pipe := range s.pipelines {
		pipe.Release()
	}
	for _, bind := range []*wgpu.BindGroup{s.refineBind, s.bindBA, s.bindAB} {
		if bind != nil {
			bind.Release()
		}
	}
	for _, layout := range []*wgpu.PipelineLayout{s.refineLayout, s.sortLayout} {
		if layout != nil {
			layout.Release()
		}
	}
	for _, bgl := range []*wgpu.BindGroupLayout{s.refineBGL, s.sortBGL} {
		if bgl != nil {
			bgl.Release()
		}
	}
	for _, buffer := range []*wgpu.Buffer{s.control, s.descriptors, s.offsets, s.hist} {
		if buffer != nil {
			buffer.Release()
		}
	}
}

func (s *spirvSA) encode(enc *wgpu.CommandEncoder) error {
	dispatch := func(pipe *wgpu.ComputePipeline, bind *wgpu.BindGroup, x, y uint32) error {
		pass, err := enc.BeginComputePass(nil)
		if err != nil {
			return err
		}
		// Pipeline must precede the bind group on the direct Vulkan HAL path.
		pass.SetPipeline(pipe)
		pass.SetBindGroup(0, bind, nil)
		pass.Dispatch(x, y, 1)
		return pass.End()
	}
	dispatchIndirect := func(pipe *wgpu.ComputePipeline, offset uint64) error {
		pass, err := enc.BeginComputePass(nil)
		if err != nil {
			return err
		}
		pass.SetPipeline(pipe)
		pass.SetBindGroup(0, s.refineBind, nil)
		pass.DispatchIndirect(s.control, offset)
		return pass.End()
	}

	enc.ClearBuffer(s.control, 0, spirvControlSize)
	groups := uint32((uint64(spirvRows*saStride) + 255) / 256)
	if err := dispatch(s.init, s.bindAB, min(groups, uint32(65535)), (groups+65534)/65535); err != nil {
		return err
	}
	for _, radix := range s.radix {
		if err := dispatch(radix.histogram, radix.bind, spirvTotalTiles, 1); err != nil {
			return err
		}
		if err := dispatch(s.scan, radix.bind, spirvRows, 1); err != nil {
			return err
		}
		if err := dispatch(radix.scatter, radix.bind, spirvTotalTiles, 1); err != nil {
			return err
		}
	}
	if err := dispatch(s.detect, s.refineBind, (saStride+255)/256, spirvRows); err != nil {
		return err
	}
	if err := dispatch(s.prepare, s.refineBind, 1, 1); err != nil {
		return err
	}
	for bin, pipe := range s.tiny {
		if err := dispatchIndirect(pipe, uint64((16+bin*3)*4)); err != nil {
			return err
		}
	}
	for i, pipe := range s.large {
		bin := i + 4
		if err := dispatchIndirect(pipe, uint64((16+bin*3)*4)); err != nil {
			return err
		}
	}
	return nil
}
