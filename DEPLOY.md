# DEPLOY — Dirtybird Go GPU Miner (operator guide)

Pure-Go, zero-CGO GPU miner for DERO (AstroBWTv3). One binary, no CUDA,
ROCm, vendor SDK, or external shader compiler. Embedded native SPIR-V
accelerates validated NVIDIA/Vulkan hardware; WGSL supplies the other eligible
GPU paths.

---

## 1. Build

```
CGO_ENABLED=0 go build ./cmd/go-gpu-miner
```

Go 1.25+. `CGO_ENABLED=0` produces the zero-CGO binary — copy it to a compatible
machine of the same OS/architecture and run. On Linux, the pure-Go FFI layer
dynamically imports the system `libdl`, `libc`, and `libpthread`. A supported GPU
driver and its Vulkan/Metal/DX12 runtime are still required.

---

## 2. Verify before you mine (do this on every new machine / driver)

```
# 1) Prove the full GPU pipeline reproduces the consensus KAT (pow("a")), then exits.
go-gpu-miner --selftest
# Expect: SELFTEST PASS: pow("a") = 54e2324d… reproduced through the GPU pipeline

# 2) Benchmark end-to-end hashes on this GPU.
go-gpu-miner --benchpipe 40960
```

If `--selftest` does NOT print PASS, do not mine on that machine — the GPU/driver
combination is miscomputing. See "Known limitations" for the subgroup caveat that
is the most likely cause on Intel iGPUs and the `software` backend.

> Operational note: the miner also runs the KAT automatically at the start of a
> live mining session and refuses to mine if it fails, so a miscomputing
> GPU/driver aborts rather than wasting effort. Running `--selftest` yourself
> after a driver change or binary update is still a good habit.

---

## 3. Mine live

```
go-gpu-miner -d <pool-host:port> -w <your-dero-wallet>
```

Release archives include a blank `config.json` beside the binary; fill it in:

```json
{"daemon-address": "pool-host:port", "wallet": "dero1..."}
```

Flags:

| Flag | Meaning |
|---|---|
| `-d` | pool/daemon `host:port` (bare `host:port` = TLS) |
| `-w` | DERO wallet address |
| `-c` | config.json path (default `config.json`) |
| `--gpu high\|low\|software` | adapter selection (see caveat below) |
| `--backend auto\|vulkan\|dx12\|gl\|software` | GPU backend selection |
| `--selftest` | run the KAT through the GPU pipeline and exit |
| `--bench N` | hash N inputs, report H/s, exit |
| `--benchpipe N` | like `--bench` via the double-buffered streaming path |
| `--sapath auto\|spirv\|refine\|portable` | select or force the suffix-array implementation |
| `--batch N` | hashes per streaming batch (auto default: SPIR-V 2048, WGSL 1664) |

The 5-second status line shows daemon-authoritative Height, Diff, accepted
MiniBlocks, Rejected, and the local `submitted`, `dropped` (submit-queue drops),
and `sendfails` counters.

---

## 4. Safety guarantee — why you cannot submit a bad block

Every candidate that meets the target is **re-hashed on the CPU reference oracle**
(`astrobwt.Sum`) and compared byte-for-byte to the GPU result **before** it is
submitted (`processBatch`, `cmd/go-gpu-miner/main.go`). A candidate that does not match is
logged `REJECTED` and dropped, never sent.

Consequence: a GPU miscompute (wrong driver, bad kernel path, unsupported GPU)
can **cost you a share** (a real solution is discarded), but it can **never submit
an invalid block** to the pool. Correctness of what you submit is guaranteed; the
risk of a bad GPU is lost/zero shares, not a bad reputation with the pool.

---

## 5. Current performance

- **Native production stream: 12,335.5 H/s three-run median** over 40,960 hashes
  / 20 batches on the RTX 4070 Laptop GPU. Runs were 12,331.9 / 12,350.3 /
  12,335.5 H/s (`--backend vulkan --sapath spirv --benchpipe 40960`). Laptop
  power and driver state will move this number.
- **Portable WGSL refine: ~1.88 KH/s** at its measured 1,664-row batch on the
  same GPU.
- **Native front half:** parallel Go workers build Wolf streams directly in two
  mapped upload buffers while the GPU processes the other buffer.
- **Native suffix array:** four fixed 512-row slabs use stable depth-3 radix
  sorting plus direct LCP refinement.
- **Native final SHA:** one SPIR-V dispatch hashes the 2,048-row SA archive;
  only 64 KiB of digests returns over PCIe instead of ~581 MB of SA data.

The implementation remains pure Go at runtime. Slang is a developer-only tool
for rebuilding `internal/gpu/spirv/*.spv`; the compiled modules are embedded by
`go:embed`.

---

## 6. Supported GPUs

What a backend actually did on real hardware is recorded in
`docs/backends.jsonl` (append a row with `go-gpu-miner --selftest --json`, or
sweep every combination with `cmd/verifybackends`). As of 2026-07-10 that
artifact holds:

| Device / backend | native SPIR-V | WGSL fused | portable | Note |
|---|---|---|---|---|
| NVIDIA RTX 4070 Laptop / Vulkan | **pass** | pass | pass | subgroup 32; native median 12.34 KH/s |
| Intel RaptorLake iGPU / Vulkan | n/a | pass | pass | subgroup 32; slower but correct |
| NVIDIA / DX12 | n/a | n/a | **fail** | wgpu v0.30.10 HAL removes the device before the first dispatch |
| Intel / GL | n/a | n/a | **fail** | computes a **wrong** KAT — the silent-miscompute class the gate exists for |
| Software renderer | n/a | n/a | **fail** | single-threaded interpreter with no-op barriers; no cooperative kernel can work |

AMD/Vulkan passed an earlier manual run (subgroup 64) but has no recorded row
yet — treat it as unverified until someone appends one. Apple/Metal has never
been run. The startup gate is what makes this table safe to be wrong about: a
combination that fails it **refuses to mine** instead of wasting power silently.

### Native SPIR-V gate and subgroup caveat

The native path is deliberately narrow: NVIDIA + Vulkan + subgroup size 32,
at least eight storage buffers per stage, and a storage binding/allocation large
enough for the 2,048-row SA archive (~554 MiB). `-sapath auto` evaluates these
limits and leaves every other device on WGSL. `-sapath spirv` reports a clear
error when the requirements are not met.

The production suffix-array kernel (`sa_refine.wgsl`) is **hard-coded to assume a
GPU subgroup size of at least 32** (its per-subgroup histogram is sized 8×256).
On a GPU that reports a smaller subgroup size that histogram overflows, the sort
is corrupted, and every hash comes out wrong.

The miner **probes the subgroup size at startup** (on a throwaway device — the
probe itself can remove the device on backends without subgroup support) and,
when it is below 32, automatically switches to the portable path — the
subgroup-free SA kernel (`sa_gpu.wgsl`) with the front half on the CPU. The
startup gate is the backstop, and it is not a formality: it runs the KAT plus an
adversarial SA corpus through the exact kernels that will mine and refuses to
start if they miscompute (see the fail rows above — those are real backends the
gate rejected). `-sapath portable` forces the fallback path for testing on
hardware that would not otherwise use it.

**Guidance:** use NVIDIA/Vulkan for the measured native path. Other Vulkan
devices with subgroup ≥32 use WGSL refine; smaller subgroups use portable WGSL.
`--gpu software` is a debugging aid but fails the gate and will not mine. Run
`--selftest` after any driver change.

---

## 7. Known limitations (updated 2026-07-10)

1. **Subgroup < 32 GPUs run the slower portable path** — the fast fused SA kernel
   needs subgroup ≥ 32; below that the miner auto-falls-back to the subgroup-free
   kernel with a CPU front half (lower throughput; trust it only where the gate
   passes — check `docs/backends.jsonl`). Verify with `--selftest`.
2. ~~Submit-queue drops under network stall~~ — **resolved 2026-07-09**: a
   validated share now waits exactly as long as it can still earn (job epoch
   live and daemon link up) instead of a fixed 2 s, and the mailbox deepened
   16 → 256. A share is abandoned only when the daemon would reject it anyway.
3. ~~Status line diff/height poisoned by a malformed job~~ — **resolved
   2026-07-09**: daemon counters mirror only after job validation.
4. **DX12 and GL backends fail the startup gate** (see §6) — upstream wgpu
   v0.30.10 HAL issues, not kernel bugs; the miner refuses to mine there rather
   than waste power. Re-test on a wgpu upgrade.
5. **Native SPIR-V is currently NVIDIA/Vulkan-specific.** It is not enabled on
   AMD, Intel, Metal, or DX12 until the same subgroup and full-pipeline parity
   gates are measured there; those devices retain the WGSL implementations.

None of these can cause a bad submission; the CPU re-hash gate (§4) is the hard
backstop for consensus safety.
