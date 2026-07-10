// Command go-gpu-miner is the Dirtybird pure-Go GPU miner for DERO
// (AstroBWTv3). The portable WGSL path runs through gogpu/wgpu on
// Vulkan/Metal/DX12. On validated NVIDIA/Vulkan hardware, the measured fast
// path builds the branchy Wolf stream in Go, then runs suffix sorting and the
// final SHA in embedded native SPIR-V. Every candidate share is re-hashed on
// the CPU before submission, so a GPU miscompute can cost a share but can
// never submit a bad block.
package main

import (
	"context"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"flag"
	"fmt"
	"math/rand"
	"os"
	"os/signal"
	"sync/atomic"
	"time"

	"github.com/Dirtybird99/Dirtybird-Go-GPU-Miner/internal/astrobwt"
	"github.com/Dirtybird99/Dirtybird-Go-GPU-Miner/internal/config"
	"github.com/Dirtybird99/Dirtybird-Go-GPU-Miner/internal/getwork"
	"github.com/Dirtybird99/Dirtybird-Go-GPU-Miner/internal/gpu"
	"github.com/Dirtybird99/Dirtybird-Go-GPU-Miner/internal/miner"
)

func main() {
	var (
		daemon    = flag.String("d", "", "daemon/pool host:port (bare host:port = TLS)")
		wallet    = flag.String("w", "", "DERO wallet address to mine to")
		cfgPath   = flag.String("c", "config.json", "config.json path")
		power     = flag.String("gpu", "high", "adapter: high|low|software")
		backend   = flag.String("backend", "auto", "GPU backend: auto|vulkan|dx12|gl|software")
		sapath    = flag.String("sapath", "auto", "SA path: auto|spirv|refine|portable")
		selftest  = flag.Bool("selftest", false, "verify the KAT + SA corpus through the mining pipeline and exit")
		jsonOut   = flag.Bool("json", false, "with -selftest: emit one machine-readable verification record on stdout")
		skipGate  = flag.Bool("skip-selftest", false, "mine even if the startup gate fails (the per-share CPU re-hash still blocks bad submissions)")
		bench     = flag.Int("bench", 0, "hash N inputs through the pipeline and report H/s, then exit")
		benchpipe = flag.Int("benchpipe", 0, "like -bench but via the double-buffered streaming path")
		batch     = flag.Int("batch", 1664, "hashes per streaming batch (path-specific default when omitted)")
	)
	flag.Parse()
	batchSet := false
	flag.Visit(func(f *flag.Flag) { batchSet = batchSet || f.Name == "batch" })
	batchSize = *batch

	// Probe the subgroup size BEFORE opening the mining device: on backends
	// without subgroup support the probe dispatch removes the device it ran on
	// (observed on DX12), so it must run on a throwaway device.
	sg := gpu.ProbeSubgroupSize(parsePower(*power), gpu.ParseBackend(*backend))

	ctx, err := gpu.NewCtxBackend(parsePower(*power), gpu.ParseBackend(*backend))
	if err != nil {
		fatal("GPU init: %v", err)
	}
	defer ctx.Close()
	fmt.Printf("GPU: %s (%v)\n", ctx.Name, ctx.Backend)
	pipe := gpu.NewPipeline(ctx)

	// Path selection. The subgroup-accelerated WGSL SA (refine path) needs subgroup
	// size >= 32 — below that its cross-subgroup histogram silently corrupts the
	// sort. The faster native SPIR-V path is deliberately narrower: it is only
	// validated on NVIDIA Vulkan with 32-lane subgroups. -sapath overrides auto
	// so every implementation remains directly testable.
	var fused, spirv bool
	var pathErr error
	switch *sapath {
	case "auto":
		spirv = sg == 32 && ctx.CanSPIRV()
		fused = spirv || (sg >= 32 && ctx.CanFuse())
		if !spirv && sg >= 32 && !ctx.CanFuse() {
			fmt.Printf("device grants %d storage buffers/stage, fused needs %d — using the portable path\n",
				ctx.MaxStorageBuffers, gpu.FusedStorageBuffers)
		}
	case "spirv":
		if sg != 32 || !ctx.CanSPIRV() {
			info := ctx.Info()
			pathErr = fmt.Errorf("-sapath spirv needs NVIDIA Vulkan, subgroup size 32, and sufficient storage-buffer limits; device is vendor=%q backend=%v subgroup=%d storage-buffers=%d binding-limit=%d MiB",
				info.Vendor, ctx.Backend, sg, ctx.MaxStorageBuffers, ctx.MaxStorageBindingSize/(1024*1024))
		} else {
			fused, spirv = true, true
		}
	case "refine":
		if sg < 32 {
			pathErr = fmt.Errorf("-sapath refine needs subgroup size >= 32, device reports %d", sg)
		} else if !ctx.CanFuse() {
			pathErr = fmt.Errorf("-sapath refine: device grants %d storage buffers/stage, fused needs %d",
				ctx.MaxStorageBuffers, gpu.FusedStorageBuffers)
		} else {
			fused = true
		}
	case "portable":
		fused = false
	default:
		fatal("unknown -sapath %q (auto|spirv|refine|portable)", *sapath)
	}
	if pathErr != nil {
		// A demanded-but-impossible combination is a SKIP in a verification
		// sweep (the combo doesn't exist on this device), not a failure.
		if *selftest && *jsonOut {
			path := "fused"
			if *sapath == "spirv" {
				path = "spirv"
			}
			emitRecord(ctx, sg, path, "skip", pathErr.Error(), 0)
			return
		}
		fatal("%v", pathErr)
	}
	if spirv {
		if !batchSet {
			batchSize = gpu.SPIRVBatchSize
		} else if batchSize > gpu.SPIRVBatchSize {
			fmt.Printf("SPIR-V stream supports at most %d hashes per batch — clamping %d to %d\n", gpu.SPIRVBatchSize, batchSize, gpu.SPIRVBatchSize)
			batchSize = gpu.SPIRVBatchSize
		}
	}
	// The device's storage-binding limit caps the SA slab. Clamp rather than
	// fail deep inside a session build.
	if ms := ctx.MaxSlab(); ms > 0 && batchSize > ms {
		fmt.Printf("batch %d exceeds this device's %d-hash binding limit — clamping to %d\n", batchSize, ms, ms)
		batchSize = ms
	}
	ctx.UseRefine(fused)
	ctx.UseSPIRV(spirv)
	hash := pipe.HashBatch
	mode := "portable (CPU front + subgroup-free SA)"
	path := "portable"
	if fused {
		hash = pipe.HashBatchFused
		mode = "fused all-GPU (WGSL refine)"
		path = "fused"
	}
	if spirv {
		// The native one-shot session is deliberately fixed at 512 rows for its
		// startup corpus. Benchmarks use the same 2048-row grouped streamer as
		// live mining, including its oversize-run correctness fallback.
		hash = func(inputs [][]byte) ([][32]byte, error) {
			batches, err := pipe.HashBatchFusedStream([][][]byte{inputs})
			if err != nil {
				return nil, err
			}
			return batches[0], nil
		}
		mode = "CPU Wolf + native SPIR-V SA+SHA"
		path = "spirv"
	}
	fmt.Printf("subgroup size %d — %s path\n", sg, mode)

	switch {
	case *selftest:
		runSelfTest(pipe, ctx, sg, path, mode, *jsonOut)
	case *benchpipe > 0:
		if !fused {
			fatal("benchpipe needs the fused subgroup path (subgroup %d, sapath %s)", sg, *sapath)
		}
		runBenchStream(pipe, *benchpipe)
	case *bench > 0:
		runBench(hash, *bench)
	default:
		// Startup gate on this exact GPU/driver/binary: the KAT and the SA
		// corpus through the pipeline selected above — the one that will mine.
		// Failing it means every hash would be wasted effort indistinguishable
		// from bad luck; --skip-selftest exists because the gate is an
		// optimization, not the consensus guard (the CPU re-hash is).
		start := time.Now()
		if ok, err := pipe.SelfTest(); err != nil || !ok {
			if !*skipGate {
				fatal("startup gate failed on the %s path (%v) — this GPU/driver miscomputes; refusing to mine pure waste (override: --skip-selftest)", mode, err)
			}
			fmt.Printf("WARNING: startup gate failed (%v) — mining anyway per --skip-selftest\n", err)
		} else {
			fmt.Printf("startup gate passed in %v (%s path)\n", time.Since(start).Round(time.Millisecond), mode)
		}
		d, w := resolveConfig(*cfgPath, *daemon, *wallet)
		runMiner(pipe, hash, fused, d, w)
	}
}

// hashFunc is the selected batch hash entrypoint.
type hashFunc = func([][]byte) ([][32]byte, error)

// commit is stamped by -ldflags "-X main.commit=<sha>" so verification records
// say which build they verified.
var commit string

// selftestRecord is one machine-readable backend-verification record — one
// line of docs/backends.jsonl, the artifact behind STRATEGY.md's "Backends
// verified" metric. It comes from the same SelfTest on the same selected path
// the miner would run, so "verified" and "what mines" cannot diverge.
type selftestRecord struct {
	Name       string `json:"name"`
	Backend    string `json:"backend"`
	Subgroup   uint32 `json:"subgroup"`
	Path       string `json:"path"`   // spirv | fused | portable
	Result     string `json:"result"` // pass | fail | skip
	Error      string `json:"error,omitempty"`
	Ms         int64  `json:"ms"`
	Commit     string `json:"commit,omitempty"`
	MeasuredAt string `json:"measured_at"`
}

func emitRecord(ctx *gpu.Ctx, sg uint32, path, result, errStr string, ms int64) {
	rec := selftestRecord{
		Name: ctx.Name, Backend: ctx.Backend.String(), Subgroup: sg,
		Path: path, Result: result, Error: errStr, Ms: ms,
		Commit: commit, MeasuredAt: time.Now().Format("2006-01-02"),
	}
	b, err := json.Marshal(rec)
	if err != nil {
		fatal("marshal record: %v", err)
	}
	fmt.Println(string(b))
}

func runSelfTest(pipe *gpu.Pipeline, ctx *gpu.Ctx, sg uint32, path, mode string, jsonOut bool) {
	start := time.Now()
	ok, err := pipe.SelfTest()
	ms := time.Since(start).Milliseconds()
	if err != nil || !ok {
		if jsonOut {
			emitRecord(ctx, sg, path, "fail", fmt.Sprint(err), ms)
		} else {
			fmt.Printf("SELFTEST FAIL (%s path): %v\n", mode, err)
		}
		os.Exit(1)
	}
	if jsonOut {
		emitRecord(ctx, sg, path, "pass", "", ms)
		return
	}
	fmt.Printf("SELFTEST PASS in %v: pow(\"a\") and the synthetic SA corpus reproduced through the %s path\n",
		time.Duration(ms)*time.Millisecond, mode)
}

func runBench(hash hashFunc, n int) {
	rng := rand.New(rand.NewSource(1))
	mk := func(count int) [][]byte {
		b := make([][]byte, count)
		for i := range b {
			b[i] = make([]byte, 76)
			rng.Read(b[i])
			binary.BigEndian.PutUint32(b[i][43:47], uint32(i))
		}
		return b
	}
	if _, err := hash(mk(batchSize)); err != nil { // warmup
		fatal("bench warmup: %v", err)
	}
	start := time.Now()
	done := 0
	for done < n {
		b := mk(min(batchSize, n-done))
		if _, err := hash(b); err != nil {
			fatal("bench: %v", err)
		}
		done += len(b)
	}
	el := time.Since(start)
	fmt.Printf("bench: %d hashes (batched) in %v = %.1f H/s\n", done, el.Round(time.Millisecond), float64(done)/el.Seconds())
}

// runBenchStream benchmarks the double-buffered streaming path. The WGSL path
// overlaps batch k's readback + CPU SHA with batch k+1's GPU work; the native
// path overlaps Go Wolf generation with GPU SA + SHA. Same deterministic input
// shape as runBench for comparability.
func runBenchStream(pipe *gpu.Pipeline, n int) {
	rng := rand.New(rand.NewSource(1))
	mk := func(count int) [][]byte {
		b := make([][]byte, count)
		for i := range b {
			b[i] = make([]byte, 76)
			rng.Read(b[i])
			binary.BigEndian.PutUint32(b[i][43:47], uint32(i))
		}
		return b
	}
	nb := (n + batchSize - 1) / batchSize
	batches := make([][][]byte, nb)
	done := 0
	for i := range batches {
		c := min(batchSize, n-done)
		batches[i] = mk(c)
		done += c
	}
	if _, err := pipe.HashBatchFusedStream(batches[:1]); err != nil { // warmup
		fatal("benchpipe warmup: %v", err)
	}
	start := time.Now()
	if _, err := pipe.HashBatchFusedStream(batches); err != nil {
		fatal("benchpipe: %v", err)
	}
	el := time.Since(start)
	fmt.Printf("benchpipe: %d hashes (%d batches, double-buffered) in %v = %.1f H/s\n",
		done, nb, el.Round(time.Millisecond), float64(done)/el.Seconds())
}

func runMiner(pipe *gpu.Pipeline, hash hashFunc, stream bool, daemon, wallet string) {
	root, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()

	st := &miner.State{}
	// 256 deep: a Submit is ~120 bytes, and a mailbox this size absorbs any
	// realistic share burst, so the backpressure path in submitShare is reserved
	// for genuine network stalls rather than ordinary bursts.
	submits := make(chan getwork.Submit, 256)

	client := &getwork.Client{
		Endpoint: daemon,
		Wallet:   wallet,
		Submits:  submits,
		Logf:     func(f string, a ...interface{}) { fmt.Printf(f+"\n", a...) },
		OnJob: func(j getwork.Job) {
			if _, err := st.SetJob(j); err != nil {
				fmt.Printf("bad job: %v\n", err)
			}
		},
	}
	go client.Run(root)

	// hashrate + accounting line
	go func() {
		t := time.NewTicker(5 * time.Second)
		defer t.Stop()
		var last uint64
		var rate float64
		lastT := time.Now()
		for {
			select {
			case <-root.Done():
				return
			case now := <-t.C:
				h := st.TotalHashes.Load()
				raw := float64(h-last) / now.Sub(lastT).Seconds()
				last, lastT = h, now
				// A 5s window holds only ~15 batches of 512, so the raw rate
				// wobbles a few percent from batch-boundary quantization alone.
				// Smooth with a ~20s EMA; a real throughput change still shows
				// within about three ticks.
				if rate == 0 {
					rate = raw
				} else {
					rate += (raw - rate) * 0.22
				}
				fmt.Printf("[%s] %.1f H/s  height=%d diff=%d  accepted(miniblocks)=%d rejected=%d submitted=%d miscompute=%d dropped=%d sendfails=%d\n",
					now.Format("15:04:05"), rate, st.Height.Load(), st.Diff.Load(),
					st.MiniBlocks.Load(), st.Rejected.Load(), st.Submitted.Load(), st.Miscompute.Load(), st.Stale.Load(), client.SendFails.Load())
			}
		}
	}()

	env := &mineEnv{st: st, submits: submits, connected: client.Connected.Load, pipe: pipe}

	fmt.Printf("mining to %s via %s (Ctrl-C to stop)\n", short(wallet), daemon)
	if stream {
		streamer, err := pipe.NewFusedStreamer(batchSize)
		if err != nil {
			fatal("streamer init: %v", err)
		}
		mineLoopStream(root, streamer, env)
	} else {
		mineLoop(root, hash, env)
	}
}

// batchSize is the active streaming group size (override with -batch). Main
// selects the measured path default: 2048 for native SPIR-V, 1664 for WGSL.
// Tests that call the mining helpers without parsing flags retain 512 here.
var batchSize = 512

// batchCtx is the per-batch state that must outlive the GPU dispatch so the
// CPU re-hash gate and submit can run against the exact job the batch was
// built for — the unit the streaming miner double-buffers.
type batchCtx struct {
	blobs  [][]byte
	jobid  string
	target [32]byte
	epoch  uint64
}

// mineEnv is what the mine loops need to verify and submit shares: the shared
// job state, the submit mailbox, and whether the daemon link is up. connected
// matters because while the link is down no jobs arrive, so the epoch never
// rotates — it is the only signal that a waiting share can no longer earn.
type mineEnv struct {
	st        *miner.State
	submits   chan<- getwork.Submit
	connected func() bool

	// pipe runs the KAT re-probe after a miscompute (nil in unit tests);
	// reprobed makes that probe once-per-session.
	pipe     *gpu.Pipeline
	reprobed atomic.Bool
}

// gateOutcome is what the CPU re-hash gate decided about one target-meeting
// candidate.
type gateOutcome int

const (
	gateSubmit     gateOutcome = iota // fresh, and the CPU reproduced the pow
	gateStale                         // job rotated between snapshot and gate
	gateMiscompute                    // GPU pow the CPU could not reproduce
)

// classify orders the two independent checks behind the gate: job freshness
// (an atomic load) before the CPU re-hash (a full AstroBWTv3 hash), so a
// rotated batch never pays for hashing. rehashOK is lazy for exactly that
// reason. Staleness can never cause a re-hash mismatch — the CPU re-hashes the
// same blob the GPU hashed — so a mismatch is a genuine miscompute.
func classify(epochNow, epochBuilt uint64, rehashOK func() bool) gateOutcome {
	if epochNow != epochBuilt {
		return gateStale
	}
	if !rehashOK() {
		return gateMiscompute
	}
	return gateSubmit
}

// onMiscompute is the loud path: a miscompute means this GPU/driver/kernel
// combination computed a hash wrong, which the whole portability bet rides on.
// The CPU gate already made a bad submission impossible, so the response is
// diagnosis, not abort: on the first miscompute of the session re-run the
// startup KAT — if the device now fails a check it passed at startup it has
// degraded (thermal, overclock, driver reset) and a supervisor restart beats
// mining blind; if the KAT still passes, treat the flip as transient.
func (env *mineEnv) onMiscompute(jobid string, idx int) {
	fmt.Printf("ERROR: GPU MISCOMPUTE — CPU re-hash rejected candidate %d of job %s (share lost, block safety unaffected)\n", idx, jobid)
	if env.pipe == nil || !env.reprobed.CompareAndSwap(false, true) {
		return
	}
	ok, err := env.pipe.SelfTest()
	if err != nil || !ok {
		fatal("GPU degraded since startup: KAT re-probe failed (ok=%v err=%v) — restart the miner", ok, err)
	}
	fmt.Printf("KAT re-probe passed — treating the miscompute as a transient flip and continuing\n")
}

// newBlobs allocates a fresh batch of miniblock-sized buffers.
func newBlobs() [][]byte {
	blobs := make([][]byte, batchSize)
	for i := range blobs {
		blobs[i] = make([]byte, miner.MiniblockSize)
	}
	return blobs
}

// fillBatch stamps `blobs` with distinct-nonce copies of the current template.
func fillBatch(blobs [][]byte, tmpl []byte, nonce *uint32) {
	for i := range blobs {
		copy(blobs[i], tmpl)
		rndFill(blobs[i][36:48])
		*nonce++
		binary.BigEndian.PutUint32(blobs[i][43:47], *nonce)
	}
}

// processBatch runs the CPU re-hash gate over a computed batch and queues any
// valid shares. It discards the whole batch if the job has since rotated, so a
// stale candidate is never submitted.
func processBatch(root context.Context, env *mineEnv, pows [][32]byte, b *batchCtx) {
	st := env.st
	st.TotalHashes.Add(uint64(len(b.blobs)))
	if st.Epoch() != b.epoch {
		return // job rotated; discard (avoids stale submits)
	}
	for i := range b.blobs {
		pow := pows[i]
		if !miner.MeetsTarget(&pow, &b.target) {
			continue
		}
		// CPU re-hash gate: a GPU miscompute can cost a share but must never
		// submit a bad block.
		switch classify(st.Epoch(), b.epoch, func() bool { return astrobwt.Sum(b.blobs[i]) == pow }) {
		case gateStale:
			st.GateStale.Add(1)
			return // job rotated mid-batch; every later candidate is stale too
		case gateMiscompute:
			st.Miscompute.Add(1)
			env.onMiscompute(b.jobid, i)
		case gateSubmit:
			submitShare(root, env, b.blobs[i], b.jobid, b.epoch)
		}
	}
}

// submitShare queues one validated share. Under backpressure it waits exactly
// as long as the share can still earn — while its job epoch is live and the
// daemon link is up — instead of an arbitrary wall-clock timeout. Once the job
// rotates the daemon would reject the share anyway, and while the link is down
// no jobs arrive so the epoch never rotates; the connected check is what keeps
// an outage from parking the mine loop here forever.
func submitShare(root context.Context, env *mineEnv, blob []byte, jobid string, epoch uint64) {
	sub := getwork.Submit{JobID: jobid, Blob: hex.EncodeToString(blob)}
	select {
	case env.submits <- sub:
		env.st.Submitted.Add(1)
		fmt.Printf("share found, queued for submit (nonce region, job %s)\n", jobid)
		return
	default:
	}
	t := time.NewTicker(100 * time.Millisecond)
	defer t.Stop()
	for {
		select {
		case env.submits <- sub:
			env.st.Submitted.Add(1)
			fmt.Printf("share found, queued for submit after backpressure (job %s)\n", jobid)
			return
		case <-t.C:
			if env.st.Epoch() != epoch {
				env.st.Stale.Add(1)
				fmt.Printf("share abandoned: job rotated while the submit buffer was full\n")
				return
			}
			if !env.connected() {
				env.st.Stale.Add(1)
				fmt.Printf("share abandoned: daemon link down and submit buffer full\n")
				return
			}
		case <-root.Done():
			return
		}
	}
}

// maxHashErrs bounds consecutive GPU hash failures before the miner aborts. A
// transient hiccup recovers within a few; a persistent stream means the device
// is lost (or the streaming session is wedged), and spinning silently forever
// is worse than exiting so a supervisor/watchdog can restart the process.
const maxHashErrs = 50

// mineLoop grinds nonces in GPU batches on the current job until interrupted,
// re-verifying every candidate on the CPU before it is queued for submission.
func mineLoop(root context.Context, hash hashFunc, env *mineEnv) {
	var nonce uint32
	var errs int
	blobs := newBlobs()
	for root.Err() == nil {
		tmpl, jobid, target, epoch := env.st.Job()
		if epoch == 0 {
			time.Sleep(50 * time.Millisecond)
			continue
		}
		fillBatch(blobs, tmpl[:], &nonce)
		pows, err := hash(blobs)
		if err != nil {
			errs++
			fmt.Printf("hash error (%d/%d): %v\n", errs, maxHashErrs, err)
			if errs >= maxHashErrs {
				fatal("aborting after %d consecutive GPU hash errors — device may be lost; restart the miner", errs)
			}
			time.Sleep(100 * time.Millisecond)
			continue
		}
		errs = 0
		processBatch(root, env, pows, &batchCtx{blobs: blobs, jobid: jobid, target: target, epoch: epoch})
	}
}

// mineLoopStream is mineLoop on the double-buffered fused streaming path: while
// the GPU computes batch k+1, the CPU re-hash + submit of batch k runs. It
// double-buffers BOTH the GPU sessions (inside FusedStreamer) and the host-side
// blobs/job context, so a batch is verified against the exact job it was built
// for even though its results arrive one iteration late.
func mineLoopStream(root context.Context, streamer *gpu.FusedStreamer, env *mineEnv) {
	var nonce uint32
	var errs int
	blobBuf := [2][][]byte{newBlobs(), newBlobs()}
	var prev *batchCtx // job context of the in-flight (previously submitted) batch
	bi := 0
	for root.Err() == nil {
		tmpl, jobid, target, epoch := env.st.Job()
		if epoch == 0 {
			time.Sleep(50 * time.Millisecond)
			continue
		}
		// Build into the buffer NOT holding the in-flight previous batch.
		blobs := blobBuf[bi]
		fillBatch(blobs, tmpl[:], &nonce)
		pows, err := streamer.Submit(blobs) // returns the PREVIOUS batch's hashes
		if err != nil {
			errs++
			fmt.Printf("hash error (%d/%d): %v\n", errs, maxHashErrs, err)
			if errs >= maxHashErrs {
				fatal("aborting after %d consecutive GPU hash errors — device may be lost; restart the miner", errs)
			}
			time.Sleep(100 * time.Millisecond)
			continue
		}
		errs = 0
		if pows != nil && prev != nil {
			processBatch(root, env, pows, prev) // overlaps current GPU compute
		}
		prev = &batchCtx{blobs: blobs, jobid: jobid, target: target, epoch: epoch}
		bi = 1 - bi
	}
	// Drain the last in-flight batch.
	if prev != nil {
		if pows, err := streamer.Drain(); err == nil && pows != nil {
			processBatch(root, env, pows, prev)
		}
	}
}

func rndFill(b []byte) {
	for i := range b {
		b[i] = byte(rand.Intn(256))
	}
}

func parsePower(s string) gpu.Power {
	switch s {
	case "low":
		return gpu.PowerLow
	case "software":
		return gpu.PowerSoftware
	default:
		return gpu.PowerHigh
	}
}

func resolveConfig(path, daemon, wallet string) (string, string) {
	if f, err := config.Load(path); err == nil && f != nil {
		if daemon == "" && f.DaemonAddress != nil {
			daemon = *f.DaemonAddress
		}
		if wallet == "" && f.Wallet != nil {
			wallet = *f.Wallet
		}
	}
	if daemon == "" || wallet == "" {
		fatal("need -d <pool host:port> and -w <wallet> (or a config.json). " +
			"Use --selftest or --bench N to run offline.")
	}
	return daemon, wallet
}

func short(w string) string {
	if len(w) > 12 {
		return w[:8] + "…" + w[len(w)-4:]
	}
	return w
}

func fatal(f string, a ...interface{}) {
	fmt.Fprintf(os.Stderr, "FATAL: "+f+"\n", a...)
	os.Exit(1)
}
