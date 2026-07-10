package gpu

import (
	"context"
	"encoding/binary"
	"fmt"
	"time"
	"unsafe"

	"github.com/gogpu/gputypes"
	"github.com/gogpu/wgpu"

	"github.com/Dirtybird99/Dirtybird-Go-GPU-Miner/internal/astrobwt"
)

// fusedSession chains the front half and the suffix array in ONE submission
// with no host round-trip of the data stream: the front-half kernel writes the
// packed data stream + lengths into buffers that the SA kernel reads directly
// on the GPU. Only the SA rows (and lengths) are read back. This removes the
// CPU front half from the critical path.
type fusedSession struct {
	dev   *wgpu.Device
	slab  int
	spirv *spirvSA

	// front-half inputs
	inBuf, metaBuf, shakBuf, optBuf *wgpu.Buffer
	// shared on-GPU handoff: front-half writes, SA reads
	textBuf, lensBuf *wgpu.Buffer
	// SA scratch
	saBuf, rankBuf, rankTmpBuf, avBuf, avTmpBuf, activeBuf, keyTmpBuf *wgpu.Buffer
	// readback
	saStaging, lensStaging *wgpu.Buffer
	fhUniform, saUniform   *wgpu.Buffer

	fhShader, saShader *wgpu.ShaderModule
	fhBGL, saBGL       *wgpu.BindGroupLayout
	fhBG, saBG         *wgpu.BindGroup
	fhPL, saPL         *wgpu.PipelineLayout
	fhPipe, saPipe     *wgpu.ComputePipeline

	inStride  int // bytes per input blob row (padded)
	textWords int // packed words per hash (saStride/4)

	// host-side staging reused across submits (WriteBuffer copies synchronously,
	// so the scratch is free for reuse as soon as the call returns)
	inScratch, metaScratch []byte

	pendN    int  // hashes in the in-flight submission (submitCompute -> collect)
	saMapped bool // saStaging is currently mapped (zero-copy views handed out)
}

const fusedInStride = 128 // per-nonce input blob row (blobs are <= 76 B; padded)

func (s *fusedSession) release() {
	if s.saMapped {
		_ = s.saStaging.Unmap()
		s.saMapped = false
	}
	if s.spirv != nil {
		s.spirv.release()
		s.spirv = nil
	}
	// Guard each concrete pointer: a nil *wgpu.Buffer boxed into an interface is
	// not nil, so releasing via an interface slice would panic during cleanup of
	// a partially-built session.
	for _, b := range []*wgpu.Buffer{
		s.inBuf, s.metaBuf, s.shakBuf, s.optBuf, s.textBuf, s.lensBuf,
		s.saBuf, s.rankBuf, s.rankTmpBuf, s.avBuf, s.avTmpBuf, s.activeBuf, s.keyTmpBuf,
		s.saStaging, s.lensStaging, s.fhUniform, s.saUniform,
	} {
		if b != nil {
			b.Release()
		}
	}
	if s.fhPipe != nil {
		s.fhPipe.Release()
	}
	if s.saPipe != nil {
		s.saPipe.Release()
	}
	if s.fhPL != nil {
		s.fhPL.Release()
	}
	if s.saPL != nil {
		s.saPL.Release()
	}
	if s.fhBG != nil {
		s.fhBG.Release()
	}
	if s.saBG != nil {
		s.saBG.Release()
	}
	if s.fhBGL != nil {
		s.fhBGL.Release()
	}
	if s.saBGL != nil {
		s.saBGL.Release()
	}
	if s.fhShader != nil {
		s.fhShader.Release()
	}
	if s.saShader != nil {
		s.saShader.Release()
	}
}

func (c *Ctx) buildFusedSession(slab int) (_ *fusedSession, retErr error) {
	if c.useSPIRV {
		if !c.CanSPIRV() {
			return nil, fmt.Errorf("SPIR-V suffix array requires NVIDIA Vulkan and sufficient buffer limits")
		}
		if slab > spirvRows {
			return nil, fmt.Errorf("SPIR-V suffix array supports at most %d hashes per batch, got %d", spirvRows, slab)
		}
		// The embedded shaders have ROWS=512 baked in. Padding smaller calls is
		// safe because their unused lens entries are cleared before dispatch.
		slab = spirvRows
	}
	// Fail here with a message naming the limit, rather than deep inside
	// CreateBindGroup with a driver error. Both limits are real on hardware this
	// project targets: DX12 grants a 128 MiB storage binding (473-hash slab max),
	// and wgpu's default limits grant 8 storage buffers where this session binds 9.
	storageBuffers := uint32(FusedStorageBuffers)
	if c.useSPIRV {
		storageBuffers = 8
	}
	if c.MaxStorageBuffers != 0 && c.MaxStorageBuffers < storageBuffers {
		return nil, fmt.Errorf("fused pipeline needs %d storage buffers/stage, device grants %d — use the portable path",
			storageBuffers, c.MaxStorageBuffers)
	}
	if lim := c.MaxStorageBindingSize; lim > 0 && uint64(slab)*saStride*4 > lim {
		return nil, fmt.Errorf("fused slab too large: %d hashes exceed the %d MiB per-binding limit (~%d max)",
			slab, lim>>20, lim/(saStride*4))
	}
	if lim := c.MaxStorageBindingSize; c.useSPIRV && lim > 0 && spirvDescriptorBytes > lim {
		return nil, fmt.Errorf("SPIR-V suffix array needs a %d MiB descriptor binding, device grants %d MiB",
			spirvDescriptorBytes>>20, lim>>20)
	}
	dev := c.Device
	s := &fusedSession{dev: dev, slab: slab, inStride: fusedInStride, textWords: saStride / 4}
	s.inScratch = make([]byte, slab*s.inStride*4)
	s.metaScratch = make([]byte, slab*2*4)
	// Release any partially-allocated GPU resources if we bail mid-build.
	defer func() {
		if retErr != nil {
			s.release()
		}
	}()
	mk := func(label string, size uint64, usage wgpu.BufferUsage) (*wgpu.Buffer, error) {
		return dev.CreateBuffer(&wgpu.BufferDescriptor{Label: label, Size: size, Usage: usage})
	}
	rowBytes := uint64(saStride * 4)
	var err error
	// front-half inputs
	if s.inBuf, err = mk("fu-in", uint64(slab*s.inStride*4), wgpu.BufferUsageStorage|wgpu.BufferUsageCopyDst); err != nil {
		return nil, err
	}
	if s.metaBuf, err = mk("fu-meta", uint64(slab*2*4), wgpu.BufferUsageStorage|wgpu.BufferUsageCopyDst); err != nil {
		return nil, err
	}
	if s.shakBuf, err = mk("fu-shak", 64*4, wgpu.BufferUsageStorage|wgpu.BufferUsageCopyDst); err != nil {
		return nil, err
	}
	if s.optBuf, err = mk("fu-opt", 256*5*4, wgpu.BufferUsageStorage|wgpu.BufferUsageCopyDst); err != nil {
		return nil, err
	}
	// CopyDst lets SelfTest feed its adversarial SA corpus directly to the
	// SPIR-V sorter after the full-pipeline KAT.
	if s.textBuf, err = mk("fu-text", uint64(slab*s.textWords*4), wgpu.BufferUsageStorage|wgpu.BufferUsageCopyDst); err != nil {
		return nil, err
	}
	if s.lensBuf, err = mk("fu-lens", uint64(slab*4), wgpu.BufferUsageStorage|wgpu.BufferUsageCopySrc|wgpu.BufferUsageCopyDst); err != nil {
		return nil, err
	}
	// SA scratch
	scratch := []**wgpu.Buffer{&s.saBuf, &s.rankBuf, &s.rankTmpBuf, &s.avBuf, &s.avTmpBuf}
	if !c.useSPIRV {
		scratch = append(scratch, &s.activeBuf, &s.keyTmpBuf)
	}
	for _, p := range scratch {
		usage := wgpu.BufferUsageStorage
		if p == &s.saBuf || (c.useSPIRV && p == &s.rankTmpBuf) {
			usage |= wgpu.BufferUsageCopySrc
		}
		if *p, err = mk("fu-sa", uint64(slab)*rowBytes, usage); err != nil {
			return nil, err
		}
	}
	if s.saStaging, err = mk("fu-sastg", uint64(slab)*rowBytes, wgpu.BufferUsageCopyDst|wgpu.BufferUsageMapRead); err != nil {
		return nil, err
	}
	// One extra word carries the native sorter's oversize-run flag. WGSL maps
	// only its n lens words and ignores the spare slot.
	if s.lensStaging, err = mk("fu-lenstg", uint64((slab+1)*4), wgpu.BufferUsageCopyDst|wgpu.BufferUsageMapRead); err != nil {
		return nil, err
	}
	if s.fhUniform, err = mk("fu-fhu", 8, wgpu.BufferUsageUniform|wgpu.BufferUsageCopyDst); err != nil {
		return nil, err
	}
	if !c.useSPIRV {
		if s.saUniform, err = mk("fu-sau", 16, wgpu.BufferUsageUniform|wgpu.BufferUsageCopyDst); err != nil {
			return nil, err
		}
	}

	// static uploads
	if err = dev.Queue().WriteBuffer(s.shakBuf, 0, U32s(sha256K[:]...)); err != nil {
		return nil, err
	}
	if err = dev.Queue().WriteBuffer(s.optBuf, 0, U32s(astrobwt.WolfOpTable()...)); err != nil {
		return nil, err
	}
	if err = dev.Queue().WriteBuffer(s.fhUniform, 0, U32s(uint32(slab), uint32(s.textWords))); err != nil {
		return nil, err
	}
	if !c.useSPIRV {
		if err = dev.Queue().WriteBuffer(s.saUniform, 0, U32s(uint32(slab), uint32(saStride), 0)); err != nil {
			return nil, err
		}
	}

	roStore := gputypes.BufferBindingTypeReadOnlyStorage
	rwStore := gputypes.BufferBindingTypeStorage
	uni := gputypes.BufferBindingTypeUniform
	entry := func(b uint32, ty gputypes.BufferBindingType) wgpu.BindGroupLayoutEntry {
		return wgpu.BindGroupLayoutEntry{Binding: b, Visibility: wgpu.ShaderStageCompute, Buffer: &gputypes.BufferBindingLayout{Type: ty}}
	}
	bge := func(b uint32, buf *wgpu.Buffer) wgpu.BindGroupEntry {
		return wgpu.BindGroupEntry{Binding: b, Buffer: buf, Size: buf.Size()}
	}

	// front-half pipeline
	if s.fhShader, err = dev.CreateShaderModule(&wgpu.ShaderModuleDescriptor{Label: "fu-fh", WGSL: U64WGSL + FrontHalfWGSL}); err != nil {
		return nil, fmt.Errorf("fh shader: %w", err)
	}
	if s.fhBGL, err = dev.CreateBindGroupLayout(&wgpu.BindGroupLayoutDescriptor{Label: "fu-fhbgl", Entries: []wgpu.BindGroupLayoutEntry{
		entry(0, roStore), entry(1, roStore), entry(2, roStore), entry(3, roStore), entry(4, rwStore), entry(5, rwStore), entry(6, uni),
	}}); err != nil {
		return nil, err
	}
	if s.fhBG, err = dev.CreateBindGroup(&wgpu.BindGroupDescriptor{Label: "fu-fhbg", Layout: s.fhBGL, Entries: []wgpu.BindGroupEntry{
		bge(0, s.inBuf), bge(1, s.metaBuf), bge(2, s.shakBuf), bge(3, s.optBuf), bge(4, s.textBuf), bge(5, s.lensBuf), bge(6, s.fhUniform),
	}}); err != nil {
		return nil, err
	}
	if s.fhPL, err = dev.CreatePipelineLayout(&wgpu.PipelineLayoutDescriptor{Label: "fu-fhpl", BindGroupLayouts: []*wgpu.BindGroupLayout{s.fhBGL}}); err != nil {
		return nil, err
	}
	if s.fhPipe, err = dev.CreateComputePipeline(&wgpu.ComputePipelineDescriptor{Label: "fu-fhpipe", Layout: s.fhPL, Module: s.fhShader, EntryPoint: "fronthalf_main"}); err != nil {
		return nil, fmt.Errorf("fh pipeline: %w", err)
	}

	if c.useSPIRV {
		if s.spirv, err = buildSPIRVSA(s); err != nil {
			return nil, err
		}
		return s, nil
	}

	// SA (refine) pipeline reading textBuf/lensBuf
	if s.saShader, err = dev.CreateShaderModule(&wgpu.ShaderModuleDescriptor{Label: "fu-sa", WGSL: SARefineWGSL}); err != nil {
		return nil, fmt.Errorf("sa shader: %w", err)
	}
	if s.saBGL, err = dev.CreateBindGroupLayout(&wgpu.BindGroupLayoutDescriptor{Label: "fu-sabgl", Entries: []wgpu.BindGroupLayoutEntry{
		entry(0, roStore), entry(1, roStore), entry(2, rwStore), entry(3, rwStore), entry(4, rwStore), entry(5, rwStore), entry(6, rwStore), entry(7, rwStore), entry(8, rwStore), entry(9, uni),
	}}); err != nil {
		return nil, err
	}
	if s.saBG, err = dev.CreateBindGroup(&wgpu.BindGroupDescriptor{Label: "fu-sabg", Layout: s.saBGL, Entries: []wgpu.BindGroupEntry{
		bge(0, s.textBuf), bge(1, s.lensBuf), bge(2, s.saBuf), bge(3, s.rankBuf), bge(4, s.rankTmpBuf), bge(5, s.avBuf), bge(6, s.avTmpBuf), bge(7, s.activeBuf), bge(8, s.keyTmpBuf), bge(9, s.saUniform),
	}}); err != nil {
		return nil, err
	}
	if s.saPL, err = dev.CreatePipelineLayout(&wgpu.PipelineLayoutDescriptor{Label: "fu-sapl", BindGroupLayouts: []*wgpu.BindGroupLayout{s.saBGL}}); err != nil {
		return nil, err
	}
	if s.saPipe, err = dev.CreateComputePipeline(&wgpu.ComputePipelineDescriptor{Label: "fu-sapipe", Layout: s.saPL, Module: s.saShader, EntryPoint: "sa_refine_main"}); err != nil {
		return nil, fmt.Errorf("sa pipeline: %w", err)
	}
	return s, nil
}

func (s *fusedSession) fits(c *Ctx, hashes int) bool {
	return s != nil && s.slab >= hashes && (s.spirv != nil) == c.useSPIRV
}

func (s *fusedSession) saResultBuf() *wgpu.Buffer {
	if s.spirv != nil {
		return s.rankTmpBuf // four depth-3 radix passes finish back in A
	}
	return s.saBuf
}

// run feeds n input blobs (each <= fusedInStride bytes), dispatches front-half
// then SA in one submission, and returns per-hash suffix arrays + data lengths.
func (s *fusedSession) run(inputs [][]byte) ([][]int32, []uint32, error) {
	dev := s.dev
	n := len(inputs)
	if n > s.slab {
		return nil, nil, fmt.Errorf("batch has %d hashes, session holds %d", n, s.slab)
	}
	in := make([]byte, n*s.inStride)
	meta := make([]uint32, n*2)
	for i, b := range inputs {
		copy(in[i*s.inStride:], b)
		meta[i*2] = uint32(i * s.inStride)
		meta[i*2+1] = uint32(len(b))
	}
	if err := dev.Queue().WriteBuffer(s.inBuf, 0, bytesToU32(in)); err != nil {
		return nil, nil, err
	}
	if err := dev.Queue().WriteBuffer(s.metaBuf, 0, U32s(meta...)); err != nil {
		return nil, nil, err
	}
	// count = actual n this call (session may be sized for a larger slab).
	if err := dev.Queue().WriteBuffer(s.fhUniform, 0, U32s(uint32(n), uint32(s.textWords))); err != nil {
		return nil, nil, err
	}
	if s.spirv == nil {
		if err := dev.Queue().WriteBuffer(s.saUniform, 0, U32s(uint32(n), uint32(saStride), 0)); err != nil {
			return nil, nil, err
		}
	}

	enc, err := dev.CreateCommandEncoder(nil)
	if err != nil {
		return nil, nil, err
	}
	if s.spirv != nil {
		enc.ClearBuffer(s.lensBuf, 0, uint64(s.slab)*4)
	}
	fhPass, err := enc.BeginComputePass(nil)
	if err != nil {
		return nil, nil, err
	}
	fhPass.SetPipeline(s.fhPipe)
	fhPass.SetBindGroup(0, s.fhBG, nil)
	fhPass.Dispatch((uint32(n)+63)/64, 1, 1)
	if err := fhPass.End(); err != nil {
		return nil, nil, err
	}
	if s.spirv != nil {
		if err := s.spirv.encode(enc); err != nil {
			return nil, nil, err
		}
	} else {
		saPass, err := enc.BeginComputePass(nil)
		if err != nil {
			return nil, nil, err
		}
		saPass.SetPipeline(s.saPipe)
		saPass.SetBindGroup(0, s.saBG, nil)
		saPass.Dispatch(uint32(n), 1, 1)
		if err := saPass.End(); err != nil {
			return nil, nil, err
		}
	}
	enc.CopyBufferToBuffer(s.lensBuf, 0, s.lensStaging, 0, uint64(n)*4)
	if s.spirv != nil {
		enc.CopyBufferToBuffer(s.spirv.control, 63*4, s.lensStaging, uint64(s.slab)*4, 4)
	}
	cmd, err := enc.Finish()
	if err != nil {
		return nil, nil, err
	}
	if _, err := dev.Queue().Submit(cmd); err != nil {
		return nil, nil, err
	}

	// Phase 1: map the GPU-computed lens (waits for the compute to finish).
	lensRead := n
	if s.spirv != nil {
		lensRead = s.slab + 1
	}
	lens, err := mapU32(dev, s.lensStaging, uint64(lensRead)*4)
	if err != nil {
		return nil, nil, fmt.Errorf("map lens: %w", err)
	}
	if s.spirv != nil {
		if lens[s.slab] != 0 {
			return nil, nil, fmt.Errorf("native suffix sort encountered a tied run longer than 2048")
		}
		lens = lens[:n]
	}

	// Phase 2: with lens known, read back only the meaningful prefix of each
	// SA row, packed back to back — the rest of each 70912-entry row is
	// garbage and not worth the PCIe traffic. Compute is already done, so
	// these copies are pure DMA.
	enc2, err := dev.CreateCommandEncoder(nil)
	if err != nil {
		return nil, nil, err
	}
	rowBytes := uint64(saStride * 4)
	var total uint64
	for i := 0; i < n; i++ {
		sz := uint64(lens[i]) * 4
		if sz == 0 {
			continue
		}
		enc2.CopyBufferToBuffer(s.saResultBuf(), uint64(i)*rowBytes, s.saStaging, total, sz)
		total += sz
	}
	cmd2, err := enc2.Finish()
	if err != nil {
		return nil, nil, err
	}
	if _, err := dev.Queue().Submit(cmd2); err != nil {
		return nil, nil, err
	}

	sas := make([][]int32, n)
	if total == 0 {
		return sas, lens, nil
	}
	saRaw, err := mapBytes(dev, s.saStaging, total)
	if err != nil {
		return nil, nil, fmt.Errorf("map sa: %w", err)
	}
	// Zero-copy decode: disjoint int32 views over the packed readback buffer
	// (little-endian host, like the rest of the GPU byte packing).
	words := unsafe.Slice((*int32)(unsafe.Pointer(&saRaw[0])), len(saRaw)/4)
	base := 0
	for i := 0; i < n; i++ {
		dl := int(lens[i])
		sas[i] = words[base : base+dl : base+dl]
		base += dl
	}
	return sas, lens, nil
}

// runSPIRVData validates the native sorter directly on caller-provided text.
// It is only used by the startup adversarial SA gate.
func (s *fusedSession) runSPIRVData(datas [][]byte) ([][]int32, error) {
	if s.spirv == nil {
		return nil, fmt.Errorf("direct SPIR-V SA validation needs the SPIR-V session")
	}
	if len(datas) > s.slab {
		return nil, fmt.Errorf("SA corpus has %d rows, session holds %d", len(datas), s.slab)
	}
	text := make([]byte, s.slab*saStride)
	lens := make([]uint32, s.slab)
	for i, data := range datas {
		if len(data) > saStride {
			return nil, fmt.Errorf("SA corpus row %d has %d bytes, max %d", i, len(data), saStride)
		}
		copy(text[i*saStride:], data)
		lens[i] = uint32(len(data))
	}
	if err := s.dev.Queue().WriteBuffer(s.textBuf, 0, text); err != nil {
		return nil, err
	}
	if err := s.dev.Queue().WriteBuffer(s.lensBuf, 0, U32s(lens...)); err != nil {
		return nil, err
	}

	enc, err := s.dev.CreateCommandEncoder(nil)
	if err != nil {
		return nil, err
	}
	if err := s.spirv.encode(enc); err != nil {
		return nil, err
	}
	enc.CopyBufferToBuffer(s.spirv.control, 63*4, s.lensStaging, 0, 4)
	rowBytes := uint64(saStride * 4)
	var total uint64
	for i, data := range datas {
		size := uint64(len(data) * 4)
		if size != 0 {
			enc.CopyBufferToBuffer(s.saResultBuf(), uint64(i)*rowBytes, s.saStaging, total, size)
			total += size
		}
	}
	cmd, err := enc.Finish()
	if err != nil {
		return nil, err
	}
	if _, err := s.dev.Queue().Submit(cmd); err != nil {
		return nil, err
	}
	overflow, err := mapU32(s.dev, s.lensStaging, 4)
	if err != nil {
		return nil, fmt.Errorf("map native overflow flag: %w", err)
	}
	if overflow[0] != 0 {
		return nil, fmt.Errorf("native suffix sort encountered a tied run longer than 2048")
	}
	if total == 0 {
		return make([][]int32, len(datas)), nil
	}
	raw, err := mapBytes(s.dev, s.saStaging, total)
	if err != nil {
		return nil, err
	}
	words := unsafe.Slice((*int32)(unsafe.Pointer(&raw[0])), len(raw)/4)
	result := make([][]int32, len(datas))
	base := 0
	for i, data := range datas {
		n := len(data)
		result[i] = words[base : base+n : base+n]
		base += n
	}
	return result, nil
}

// submitCompute encodes the front-half + SA + a FULL-row SA readback copy and
// submits it WITHOUT blocking on the result. It is the producer half of the
// double-buffered streaming path (collect is the consumer). Single-phase
// readback (full rows, no lens-dependent second submit) is deliberate: a
// second, lens-sized copy would queue behind the NEXT batch's compute on the
// single ordered queue and destroy the overlap. The full-row map is pure DMA
// (~2.5ms) and is hidden behind the next batch's ~300ms compute.
func (s *fusedSession) submitCompute(inputs [][]byte) error {
	dev := s.dev
	n := len(inputs)
	if n > s.slab {
		return fmt.Errorf("batch has %d hashes, session holds %d", n, s.slab)
	}
	s.pendN = n
	// Release any zero-copy view from a prior collect before the GPU copy
	// below rewrites saStaging (wgpu requires unmap before a buffer is a copy
	// target again). The driver has already consumed those views by now.
	if s.saMapped {
		_ = s.saStaging.Unmap()
		s.saMapped = false
	}
	in := s.inScratch[:n*s.inStride*4]
	clear(in) // row padding must read as zeros, same as the old fresh-alloc path
	meta := s.metaScratch[:n*2*4]
	for i, b := range inputs {
		base := i * s.inStride * 4
		for j, v := range b {
			binary.LittleEndian.PutUint32(in[base+j*4:], uint32(v))
		}
		binary.LittleEndian.PutUint32(meta[i*8:], uint32(i*s.inStride))
		binary.LittleEndian.PutUint32(meta[i*8+4:], uint32(len(b)))
	}
	if err := dev.Queue().WriteBuffer(s.inBuf, 0, in); err != nil {
		return err
	}
	if err := dev.Queue().WriteBuffer(s.metaBuf, 0, meta); err != nil {
		return err
	}
	if err := dev.Queue().WriteBuffer(s.fhUniform, 0, U32s(uint32(n), uint32(s.textWords))); err != nil {
		return err
	}
	if s.spirv == nil {
		if err := dev.Queue().WriteBuffer(s.saUniform, 0, U32s(uint32(n), uint32(saStride), 0)); err != nil {
			return err
		}
	}

	enc, err := dev.CreateCommandEncoder(nil)
	if err != nil {
		return err
	}
	if s.spirv != nil {
		enc.ClearBuffer(s.lensBuf, 0, uint64(s.slab)*4)
	}
	fhPass, err := enc.BeginComputePass(nil)
	if err != nil {
		return err
	}
	fhPass.SetPipeline(s.fhPipe)
	fhPass.SetBindGroup(0, s.fhBG, nil)
	fhPass.Dispatch((uint32(n)+63)/64, 1, 1)
	if err := fhPass.End(); err != nil {
		return err
	}
	if s.spirv != nil {
		if err := s.spirv.encode(enc); err != nil {
			return err
		}
	} else {
		saPass, err := enc.BeginComputePass(nil)
		if err != nil {
			return err
		}
		saPass.SetPipeline(s.saPipe)
		saPass.SetBindGroup(0, s.saBG, nil)
		saPass.Dispatch(uint32(n), 1, 1)
		if err := saPass.End(); err != nil {
			return err
		}
	}
	rowBytes := uint64(saStride * 4)
	enc.CopyBufferToBuffer(s.saResultBuf(), 0, s.saStaging, 0, uint64(n)*rowBytes)
	enc.CopyBufferToBuffer(s.lensBuf, 0, s.lensStaging, 0, uint64(n)*4)
	cmd, err := enc.Finish()
	if err != nil {
		return err
	}
	_, err = dev.Queue().Submit(cmd)
	return err
}

// collect blocks until THIS session's in-flight submission resolves — using
// MapAsync + a non-draining Poll(PollPoll) spin so it waits only on this
// session's fence, never on another session's queued compute (Buffer.Map would
// WaitIdle the whole device and defeat the overlap). Returns per-hash SA views
// (zero-copy over the mapped staging) and lengths. The returned slices alias
// the mapped buffer and are valid until this session's next submitCompute.
func (s *fusedSession) collect() (sas [][]int32, lens []uint32, retErr error) {
	dev := s.dev
	n := s.pendN
	rowBytes := uint64(saStride * 4)

	// On any error path, unmap both staging buffers so the session stays
	// re-usable. Otherwise a single transient map error leaves a buffer mapped
	// or a MapAsync pending, the next submitCompute's copy into it fails, and
	// every subsequent Submit errors — permanently wedging the stream. Unmap
	// cancels a pending MapAsync, so this recovers both cases. The success path
	// deliberately keeps saStaging mapped (zero-copy views), so guard on retErr.
	defer func() {
		if retErr != nil {
			_ = s.lensStaging.Unmap()
			_ = s.saStaging.Unmap()
			s.saMapped = false
		}
	}()

	pLens, err := s.lensStaging.MapAsync(wgpu.MapModeRead, 0, uint64(n)*4)
	if err != nil {
		return nil, nil, fmt.Errorf("map lens: %w", err)
	}
	pSa, err := s.saStaging.MapAsync(wgpu.MapModeRead, 0, uint64(n)*rowBytes)
	if err != nil {
		return nil, nil, fmt.Errorf("map sa: %w", err)
	}
	// Non-draining wait: advance fences with PollPoll (reads the completed
	// fence value, no WaitIdle) until BOTH maps resolve. Other sessions'
	// compute keeps running on the GPU meanwhile.
	deadline := 120 * time.Second
	start := time.Now()
	lensDone, saDone := false, false
	for spins := 0; !lensDone || !saDone; spins++ {
		if spins > 8 {
			// The map usually resolves ~a full batch-compute later (~300ms):
			// spinning Poll flat-out burns a whole core for that window, and on
			// a laptop the CPU and GPU share one power budget. 200µs naps cost
			// <0.1% latency per batch and give the wattage back to the GPU.
			time.Sleep(200 * time.Microsecond)
		}
		dev.Poll(wgpu.PollPoll)
		if !lensDone {
			if d, e := pLens.Status(); d {
				if e != nil {
					return nil, nil, fmt.Errorf("lens map: %w", e)
				}
				lensDone = true
			}
		}
		if !saDone {
			if d, e := pSa.Status(); d {
				if e != nil {
					return nil, nil, fmt.Errorf("sa map: %w", e)
				}
				saDone = true
			}
		}
		if (!lensDone || !saDone) && time.Since(start) > deadline {
			return nil, nil, fmt.Errorf("collect: map timed out after %v", deadline)
		}
	}

	// lens is tiny (n*4 B); copy it out and unmap immediately.
	lrng, err := s.lensStaging.MappedRange(0, uint64(n)*4)
	if err != nil {
		return nil, nil, err
	}
	lb := lrng.Bytes()
	lens = make([]uint32, n)
	for i := 0; i < n; i++ {
		lens[i] = binary.LittleEndian.Uint32(lb[i*4:])
	}
	if err := s.lensStaging.Unmap(); err != nil {
		return nil, nil, err
	}

	// SA stays MAPPED: hand back zero-copy int32 views over each full row's
	// meaningful prefix. The views alias saStaging and are valid until this
	// session's next submitCompute (which unmaps). The stream driver consumes
	// them (final SHA) before that happens.
	srng, err := s.saStaging.MappedRange(0, uint64(n)*rowBytes)
	if err != nil {
		return nil, nil, err
	}
	sb := srng.Bytes()
	words := unsafe.Slice((*int32)(unsafe.Pointer(&sb[0])), len(sb)/4)
	sas = make([][]int32, n)
	rowWords := saStride
	for i := 0; i < n; i++ {
		dl := int(lens[i])
		base := i * rowWords
		sas[i] = words[base : base+dl : base+dl]
	}
	s.saMapped = true
	return sas, lens, nil
}

func bytesToU32(b []byte) []byte {
	// b is byte-per-slot input; the front-half reads fh_in as one byte per u32.
	out := make([]byte, len(b)*4)
	for i, v := range b {
		binary.LittleEndian.PutUint32(out[i*4:], uint32(v))
	}
	return out
}

func mapBytes(dev *wgpu.Device, buf *wgpu.Buffer, size uint64) ([]byte, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 120*time.Second)
	defer cancel()
	if err := buf.Map(ctx, wgpu.MapModeRead, 0, size); err != nil {
		return nil, err
	}
	rng, err := buf.MappedRange(0, size)
	if err != nil {
		_ = buf.Unmap()
		return nil, err
	}
	out := make([]byte, size)
	copy(out, rng.Bytes())
	return out, buf.Unmap()
}

func mapU32(dev *wgpu.Device, buf *wgpu.Buffer, size uint64) ([]uint32, error) {
	b, err := mapBytes(dev, buf, size)
	if err != nil {
		return nil, err
	}
	out := make([]uint32, size/4)
	for i := range out {
		out[i] = binary.LittleEndian.Uint32(b[i*4:])
	}
	return out, nil
}
