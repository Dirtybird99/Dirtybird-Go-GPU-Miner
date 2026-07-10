package gpu

import (
	"encoding/binary"
	"fmt"
	"time"

	"github.com/Dirtybird99/Dirtybird-Go-GPU-Miner/internal/astrobwt"
	"github.com/gogpu/gputypes"
	"github.com/gogpu/wgpu"
)

const (
	spirvStreamSlabs       = 4
	spirvStreamRows        = spirvRows * spirvStreamSlabs
	spirvStreamTextBytes   = uint64(spirvStreamRows * saStride)
	spirvStreamSABytes     = spirvStreamTextBytes * 4
	spirvStreamDigestBytes = uint64(spirvStreamRows * 32)
	spirvStreamResultBytes = spirvStreamDigestBytes + spirvStreamSlabs*4
)

// SPIRVBatchSize is the measured native stream group: four 512-row SA slabs.
const SPIRVBatchSize = spirvStreamRows

type spirvStreamBinds struct {
	ab, ba, final, refine *wgpu.BindGroup
}

// spirvFastStreamer is the measured mining path. Go workers build the branchy
// Wolf stream directly in one of two mapped upload buffers while the GPU sorts
// the previous group. Four fixed 512-row sorts feed one 2048-row GPU SHA, so
// only 64 KiB of digests crosses PCIe per group.
type spirvFastStreamer struct {
	dev  *wgpu.Device
	base *fusedSession

	text, lens, archive, overflow, digests *wgpu.Buffer
	upload                                 [2]*wgpu.Buffer
	staging                                [2]*wgpu.Buffer
	binds                                  [spirvStreamSlabs]spirvStreamBinds
	shaBGL                                 *wgpu.BindGroupLayout
	shaBG                                  *wgpu.BindGroup
	shaLayout                              *wgpu.PipelineLayout
	shaPipe                                *wgpu.ComputePipeline

	uploadPending [2]*wgpu.MapPending
	resultPending [2]*wgpu.MapPending
	uploadMapped  [2]bool
	pendN         [2]int
	pendInputs    [2][][]byte
	lensHost      []uint32
	cur           int
	have          bool
}

func buildSPIRVFastStreamer(c *Ctx) (_ *spirvFastStreamer, retErr error) {
	if !c.CanSPIRV() {
		return nil, fmt.Errorf("SPIR-V streamer requires NVIDIA Vulkan and sufficient buffer limits")
	}
	base, err := c.buildFusedSession(spirvRows)
	if err != nil {
		return nil, err
	}
	f := &spirvFastStreamer{dev: c.Device, base: base, lensHost: make([]uint32, spirvStreamRows)}
	f.uploadMapped = [2]bool{true, true}
	defer func() {
		if retErr != nil {
			f.release()
		}
	}()

	mk := func(label string, size uint64, usage wgpu.BufferUsage) (*wgpu.Buffer, error) {
		return f.dev.CreateBuffer(&wgpu.BufferDescriptor{Label: label, Size: size, Usage: usage})
	}
	if f.text, err = mk("spv-stream-text", spirvStreamTextBytes, wgpu.BufferUsageStorage|wgpu.BufferUsageCopyDst); err != nil {
		return nil, err
	}
	if f.lens, err = mk("spv-stream-lens", spirvStreamRows*4, wgpu.BufferUsageStorage|wgpu.BufferUsageCopyDst); err != nil {
		return nil, err
	}
	if f.archive, err = mk("spv-stream-sa", spirvStreamSABytes, wgpu.BufferUsageStorage); err != nil {
		return nil, err
	}
	if f.overflow, err = mk("spv-stream-overflow", spirvStreamSlabs*4, wgpu.BufferUsageCopyDst|wgpu.BufferUsageCopySrc); err != nil {
		return nil, err
	}
	if f.digests, err = mk("spv-stream-digests", spirvStreamDigestBytes, wgpu.BufferUsageStorage|wgpu.BufferUsageCopySrc); err != nil {
		return nil, err
	}
	for i := range f.upload {
		f.upload[i], err = f.dev.CreateBuffer(&wgpu.BufferDescriptor{
			Label: fmt.Sprintf("spv-stream-upload-%d", i), Size: spirvStreamTextBytes,
			Usage: wgpu.BufferUsageMapWrite | wgpu.BufferUsageCopySrc, MappedAtCreation: true,
		})
		if err != nil {
			return nil, err
		}
		if f.staging[i], err = mk(fmt.Sprintf("spv-stream-staging-%d", i), spirvStreamResultBytes, wgpu.BufferUsageCopyDst|wgpu.BufferUsageMapRead); err != nil {
			return nil, err
		}
	}

	makeBind := func(label string, layout *wgpu.BindGroupLayout, buffers []*wgpu.Buffer, sizes, offsets []uint64) (*wgpu.BindGroup, error) {
		entries := make([]wgpu.BindGroupEntry, len(buffers))
		for i := range entries {
			entries[i] = wgpu.BindGroupEntry{Binding: uint32(i), Buffer: buffers[i], Offset: offsets[i], Size: sizes[i]}
		}
		return f.dev.CreateBindGroup(&wgpu.BindGroupDescriptor{Label: label, Layout: layout, Entries: entries})
	}
	sp := base.spirv
	sortSizes := []uint64{spirvPairBytes, spirvPairBytes, spirvPairBytes, spirvPairBytes, spirvAuxBytes, spirvAuxBytes, spirvPairBytes / 4, spirvRows * 4}
	for slab := 0; slab < spirvStreamSlabs; slab++ {
		textOffset := uint64(slab) * spirvPairBytes / 4
		lensOffset := uint64(slab * spirvRows * 4)
		saOffset := uint64(slab) * spirvPairBytes
		f.binds[slab].ab, err = makeBind(fmt.Sprintf("spv-stream-ab-%d", slab), sp.sortBGL,
			[]*wgpu.Buffer{base.rankBuf, base.rankTmpBuf, base.avBuf, base.saBuf, sp.hist, sp.offsets, f.text, f.lens},
			sortSizes, []uint64{0, 0, 0, 0, 0, 0, textOffset, lensOffset})
		if err != nil {
			return nil, err
		}
		f.binds[slab].ba, err = makeBind(fmt.Sprintf("spv-stream-ba-%d", slab), sp.sortBGL,
			[]*wgpu.Buffer{base.avBuf, base.saBuf, base.rankBuf, base.rankTmpBuf, sp.hist, sp.offsets, f.text, f.lens},
			sortSizes, []uint64{0, 0, 0, 0, 0, 0, textOffset, lensOffset})
		if err != nil {
			return nil, err
		}
		f.binds[slab].final, err = makeBind(fmt.Sprintf("spv-stream-final-%d", slab), sp.sortBGL,
			[]*wgpu.Buffer{base.avBuf, base.saBuf, base.rankBuf, f.archive, sp.hist, sp.offsets, f.text, f.lens},
			sortSizes, []uint64{0, 0, 0, saOffset, 0, 0, textOffset, lensOffset})
		if err != nil {
			return nil, err
		}
		f.binds[slab].refine, err = makeBind(fmt.Sprintf("spv-stream-refine-%d", slab), sp.refineBGL,
			[]*wgpu.Buffer{f.text, f.lens, base.rankBuf, f.archive, sp.descriptors, base.avTmpBuf, sp.control},
			[]uint64{spirvPairBytes / 4, spirvRows * 4, spirvPairBytes, spirvPairBytes, spirvDescriptorBytes, spirvPairBytes, spirvControlSize},
			[]uint64{textOffset, lensOffset, 0, saOffset, 0, 0, 0})
		if err != nil {
			return nil, err
		}
	}

	entries := make([]wgpu.BindGroupLayoutEntry, 4)
	for i := range entries {
		ty := gputypes.BufferBindingTypeReadOnlyStorage
		if i == 3 {
			ty = gputypes.BufferBindingTypeStorage
		}
		entries[i] = wgpu.BindGroupLayoutEntry{Binding: uint32(i), Visibility: wgpu.ShaderStageCompute, Buffer: &gputypes.BufferBindingLayout{Type: ty}}
	}
	if f.shaBGL, err = f.dev.CreateBindGroupLayout(&wgpu.BindGroupLayoutDescriptor{Label: "spv-stream-sha-bgl", Entries: entries}); err != nil {
		return nil, err
	}
	if f.shaLayout, err = f.dev.CreatePipelineLayout(&wgpu.PipelineLayoutDescriptor{Label: "spv-stream-sha-layout", BindGroupLayouts: []*wgpu.BindGroupLayout{f.shaBGL}}); err != nil {
		return nil, err
	}
	if f.shaBG, err = makeBind("spv-stream-sha", f.shaBGL,
		[]*wgpu.Buffer{f.archive, f.lens, base.shakBuf, f.digests},
		[]uint64{spirvStreamSABytes, spirvStreamRows * 4, base.shakBuf.Size(), spirvStreamDigestBytes},
		[]uint64{0, 0, 0, 0}); err != nil {
		return nil, err
	}
	words, err := embeddedSPIRV("final_sha")
	if err != nil {
		return nil, err
	}
	module, err := f.dev.CreateShaderModule(&wgpu.ShaderModuleDescriptor{Label: "spv-stream-sha", SPIRV: words})
	if err != nil {
		return nil, err
	}
	defer module.Release()
	if f.shaPipe, err = f.dev.CreateComputePipeline(&wgpu.ComputePipelineDescriptor{Label: "spv-stream-sha", Layout: f.shaLayout, Module: module, EntryPoint: "main"}); err != nil {
		return nil, err
	}
	return f, nil
}

func (f *spirvFastStreamer) release() {
	if f == nil {
		return
	}
	for i := range f.staging {
		if f.resultPending[i] != nil {
			f.resultPending[i].Release()
			f.resultPending[i] = nil
		}
		if f.uploadPending[i] != nil {
			f.uploadPending[i].Release()
			f.uploadPending[i] = nil
		}
		if f.staging[i] != nil {
			_ = f.staging[i].Unmap()
			f.staging[i].Release()
		}
		if f.upload[i] != nil {
			_ = f.upload[i].Unmap()
			f.upload[i].Release()
		}
	}
	for i := range f.binds {
		for _, bg := range []*wgpu.BindGroup{f.binds[i].refine, f.binds[i].final, f.binds[i].ba, f.binds[i].ab} {
			if bg != nil {
				bg.Release()
			}
		}
	}
	if f.shaPipe != nil {
		f.shaPipe.Release()
	}
	if f.shaBG != nil {
		f.shaBG.Release()
	}
	if f.shaLayout != nil {
		f.shaLayout.Release()
	}
	if f.shaBGL != nil {
		f.shaBGL.Release()
	}
	for _, b := range []*wgpu.Buffer{f.digests, f.overflow, f.archive, f.lens, f.text} {
		if b != nil {
			b.Release()
		}
	}
	if f.base != nil {
		f.base.release()
		f.base = nil
	}
}

func (f *spirvFastStreamer) wait(p *wgpu.MapPending) error {
	if p == nil {
		return fmt.Errorf("missing GPU map")
	}
	start := time.Now()
	for spins := 0; ; spins++ {
		f.dev.Poll(wgpu.PollPoll)
		if done, err := p.Status(); done {
			return err
		}
		if time.Since(start) > 120*time.Second {
			return fmt.Errorf("GPU map timed out")
		}
		if spins > 8 {
			time.Sleep(100 * time.Microsecond)
		}
	}
}

func (f *spirvFastStreamer) prepare(index int, inputs [][]byte) error {
	if len(inputs) == 0 || len(inputs) > spirvStreamRows {
		return fmt.Errorf("SPIR-V stream batch must contain 1..%d hashes, got %d", spirvStreamRows, len(inputs))
	}
	if !f.uploadMapped[index] {
		pending := f.uploadPending[index]
		f.uploadPending[index] = nil
		err := f.wait(pending)
		if pending != nil {
			pending.Release()
		}
		if err != nil {
			return fmt.Errorf("map upload %d: %w", index, err)
		}
		f.uploadMapped[index] = true
	}
	rng, err := f.upload[index].MappedRange(0, spirvStreamTextBytes)
	if err != nil {
		return err
	}
	text := rng.Bytes()
	clear(f.lensHost)
	parallelFor(len(inputs), func(i int) {
		row := text[i*saStride : (i+1)*saStride]
		n, _ := astrobwt.WolfStreamInto(inputs[i], row)
		f.lensHost[i] = n
	})
	rng.Release()
	if err := f.upload[index].Unmap(); err != nil {
		return err
	}
	f.uploadMapped[index] = false
	if err := f.dev.Queue().WriteBuffer(f.lens, 0, U32s(f.lensHost...)); err != nil {
		return err
	}
	f.pendN[index] = len(inputs)
	f.pendInputs[index] = inputs
	return nil
}

func (f *spirvFastStreamer) dispatch(enc *wgpu.CommandEncoder, pipe *wgpu.ComputePipeline, bind *wgpu.BindGroup, x, y uint32) error {
	pass, err := enc.BeginComputePass(nil)
	if err != nil {
		return err
	}
	pass.SetPipeline(pipe)
	pass.SetBindGroup(0, bind, nil)
	pass.Dispatch(x, y, 1)
	return pass.End()
}

func (f *spirvFastStreamer) dispatchIndirect(enc *wgpu.CommandEncoder, pipe *wgpu.ComputePipeline, bind *wgpu.BindGroup, offset uint64) error {
	pass, err := enc.BeginComputePass(nil)
	if err != nil {
		return err
	}
	pass.SetPipeline(pipe)
	pass.SetBindGroup(0, bind, nil)
	pass.DispatchIndirect(f.base.spirv.control, offset)
	return pass.End()
}

func (f *spirvFastStreamer) encodeSlab(enc *wgpu.CommandEncoder, slab int) error {
	sp := f.base.spirv
	b := f.binds[slab]
	enc.ClearBuffer(sp.control, 0, spirvControlSize)
	groups := uint32((uint64(spirvRows*saStride) + 255) / 256)
	if err := f.dispatch(enc, sp.init, b.ab, min(groups, uint32(65535)), (groups+65534)/65535); err != nil {
		return err
	}
	for i, radix := range sp.radix {
		bind := b.ab
		if i == 3 {
			bind = b.final
		} else if i&1 != 0 {
			bind = b.ba
		}
		if err := f.dispatch(enc, radix.histogram, bind, spirvTotalTiles, 1); err != nil {
			return err
		}
		if err := f.dispatch(enc, sp.scan, bind, spirvRows, 1); err != nil {
			return err
		}
		if err := f.dispatch(enc, radix.scatter, bind, spirvTotalTiles, 1); err != nil {
			return err
		}
	}
	if err := f.dispatch(enc, sp.detect, b.refine, (saStride+255)/256, spirvRows); err != nil {
		return err
	}
	if err := f.dispatch(enc, sp.prepare, b.refine, 1, 1); err != nil {
		return err
	}
	for bin, pipe := range sp.tiny {
		if err := f.dispatchIndirect(enc, pipe, b.refine, uint64((16+bin*3)*4)); err != nil {
			return err
		}
	}
	for i, pipe := range sp.large {
		bin := i + 4
		if err := f.dispatchIndirect(enc, pipe, b.refine, uint64((16+bin*3)*4)); err != nil {
			return err
		}
	}
	enc.CopyBufferToBuffer(sp.control, 63*4, f.overflow, uint64(slab*4), 4)
	return nil
}

func (f *spirvFastStreamer) submitGroup(index int) error {
	enc, err := f.dev.CreateCommandEncoder(nil)
	if err != nil {
		return err
	}
	enc.CopyBufferToBuffer(f.upload[index], 0, f.text, 0, spirvStreamTextBytes)
	cmd, err := enc.Finish()
	if err != nil {
		return err
	}
	if _, err := f.dev.Queue().Submit(cmd); err != nil {
		return err
	}
	for slab := 0; slab < spirvStreamSlabs; slab++ {
		enc, err = f.dev.CreateCommandEncoder(nil)
		if err != nil {
			return err
		}
		if err := f.encodeSlab(enc, slab); err != nil {
			return err
		}
		cmd, err = enc.Finish()
		if err != nil {
			return err
		}
		if _, err := f.dev.Queue().Submit(cmd); err != nil {
			return err
		}
	}
	enc, err = f.dev.CreateCommandEncoder(nil)
	if err != nil {
		return err
	}
	if err := f.dispatch(enc, f.shaPipe, f.shaBG, (spirvStreamRows+31)/32, 1); err != nil {
		return err
	}
	enc.CopyBufferToBuffer(f.digests, 0, f.staging[index], 0, spirvStreamDigestBytes)
	enc.CopyBufferToBuffer(f.overflow, 0, f.staging[index], spirvStreamDigestBytes, spirvStreamSlabs*4)
	cmd, err = enc.Finish()
	if err != nil {
		return err
	}
	if _, err := f.dev.Queue().Submit(cmd); err != nil {
		return err
	}
	if f.resultPending[index], err = f.staging[index].MapAsync(wgpu.MapModeRead, 0, spirvStreamResultBytes); err != nil {
		return err
	}
	if f.uploadPending[index], err = f.upload[index].MapAsync(wgpu.MapModeWrite, 0, spirvStreamTextBytes); err != nil {
		return err
	}
	return nil
}

func (f *spirvFastStreamer) collect(index int) ([][32]byte, error) {
	pending := f.resultPending[index]
	f.resultPending[index] = nil
	err := f.wait(pending)
	if pending != nil {
		pending.Release()
	}
	if err != nil {
		return nil, err
	}
	rng, err := f.staging[index].MappedRange(0, spirvStreamResultBytes)
	if err != nil {
		return nil, err
	}
	raw := rng.Bytes()
	n := f.pendN[index]
	out := make([][32]byte, n)
	for i := range out {
		copy(out[i][:], raw[i*32:(i+1)*32])
	}
	var fallback []int
	for slab := 0; slab < spirvStreamSlabs; slab++ {
		if binary.LittleEndian.Uint32(raw[spirvStreamDigestBytes+uint64(slab*4):]) == 0 {
			continue
		}
		begin, end := slab*spirvRows, min((slab+1)*spirvRows, n)
		for i := begin; i < end; i++ {
			fallback = append(fallback, i)
		}
	}
	rng.Release()
	if err := f.staging[index].Unmap(); err != nil {
		return nil, err
	}
	inputs := f.pendInputs[index]
	parallelFor(len(fallback), func(j int) {
		i := fallback[j]
		out[i] = astrobwt.Sum(inputs[i])
	})
	f.pendInputs[index] = nil
	return out, nil
}

func (f *spirvFastStreamer) Submit(inputs [][]byte) ([][32]byte, error) {
	index := f.cur
	if err := f.prepare(index, inputs); err != nil {
		return nil, err
	}
	if err := f.submitGroup(index); err != nil {
		return nil, err
	}
	var out [][32]byte
	if f.have {
		var err error
		out, err = f.collect(1 - index)
		if err != nil {
			return nil, err
		}
	}
	f.have = true
	f.cur = 1 - index
	return out, nil
}

func (f *spirvFastStreamer) Drain() ([][32]byte, error) {
	if !f.have {
		return nil, nil
	}
	out, err := f.collect(1 - f.cur)
	if err == nil {
		f.have = false
	}
	return out, err
}
