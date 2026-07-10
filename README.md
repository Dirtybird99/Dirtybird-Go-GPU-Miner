# Dirtybird Go GPU Miner

A **pure-Go, zero-CGO GPU miner for DERO** (AstroBWTv3). The suffix array — the
dominant stage of the proof-of-work — runs through
[gogpu/wgpu](https://github.com/gogpu/wgpu). A measured native SPIR-V path
accelerates validated NVIDIA/Vulkan devices; WGSL provides the implementation
used by supported non-native backend/path combinations. Both ship inside one Go
binary, with no CUDA, CGO, vendor SDK, or external shader compiler. A startup
gate refuses to mine on a combination that miscomputes.
What each backend actually did is recorded in `docs/backends.jsonl`, including
native, WGSL-refine, and portable path results where they have been measured.
Sibling of the Dirtybird [CPU](https://github.com/Dirtybird99/Dirtybird-Go-Miner),
C++, Zig, and Rust miners.

- **Portable.** One `go build`, one binary, zero CGO. Native SPIR-V is
  embedded at build time; Slang is needed only to rebuild those shader assets.
- **Consensus-correct.** Every candidate share is re-hashed on the CPU reference
  before submission, so a GPU miscompute can cost a share but can never submit a
  bad block. Startup is gated on the `pow("a")` KAT plus an adversarial
  suffix-array corpus, run through the same kernels that mine.
- **Verified against the CPU oracle.** The GPU suffix array is diffed byte-for-byte
  against `sais.go` on real AstroBWTv3 buffers; the composed hash matches
  `astrobwt.Sum` exactly.

## Status

Correctness-complete end to end; **pure Go, zero CGO** (`CGO_ENABLED=0` builds
successfully). The production NVIDIA/Vulkan stream sustains **12.34 KH/s** on
the development RTX 4070 Laptop GPU (three `--benchpipe 40960` runs).

| Stage | Native NVIDIA/Vulkan | Portable WGSL | Verified |
|---|---|---|---|
| Front half (SHA-256 → Salsa20 → RC4 → wolf loop) | Parallel Go workers write mapped upload buffers | GPU fused kernel, or CPU on portable SA fallback | ✅ |
| Suffix array | Four 512-row SPIR-V slabs: stable depth-3 radix + direct LCP refinement | Batched WGSL segment-local refine | ✅ byte-exact vs `sais.go` |
| Final SHA-256 | SPIR-V over one 2,048-row archive; 64 KiB digest readback | CPU over SA readback | ✅ |

The native stream removes the two production bottlenecks exposed by profiling:
GPU Wolf generation and a ~139 MiB suffix-array readback per 512 hashes. Go now
prepares the next Wolf buffer while the GPU sorts the current group; four
512-row SA slabs feed one 2,048-row GPU SHA dispatch. Only the 32-byte digests
cross back to the host. Three production runs measured **12,331.9 / 12,350.3 /
12,335.5 H/s** over 40,960 hashes each (median **12,335.5 H/s**), compared with
**~1.88 KH/s** for the retained 1,664-row WGSL path on the same GPU.

`-sapath auto` selects native SPIR-V only for NVIDIA Vulkan, subgroup size 32,
and sufficient buffer limits. Otherwise it selects WGSL refine (subgroup ≥32)
or the subgroup-free portable SA. `-sapath spirv|refine|portable` forces a path
for verification.

## Downloads

GitHub releases use the same platform layout as the other Dirtybird miners:

| Platform | Asset | Status |
|---|---|---|
| Windows x64 | `Dirtybird-Go-GPU-Miner-win64-vX.Y.Z.zip` | NVIDIA/Vulkan validated |
| Linux x64 | `Dirtybird-Go-GPU-Miner-amd64-vX.Y.Z.tar.gz` | build-only until hardware verification |
| Linux arm64 | `Dirtybird-Go-GPU-Miner-arm64-vX.Y.Z.tar.gz` | build-only |
| macOS Apple Silicon | `Dirtybird-Go-GPU-Miner-macos-arm64-vX.Y.Z.tar.gz` | Metal unverified |
| HiveOS / MMPOS | `dirtybird-go-gpu-miner-vX.Y.Z.hiveos_mmpos.amd64.tar.gz` | experimental |

Verify every download against `SHA256SUMS.txt`, then run `--selftest` on the
target machine before mining.

## Build

Go 1.25+:

```
CGO_ENABLED=0 go build ./cmd/go-gpu-miner
```

On Linux, the zero-CGO GPU loader dynamically imports the system `libdl`,
`libc`, and `libpthread` through Go FFI. A compatible GPU driver/runtime is
required on every platform.

## Run

```
# verify the consensus KAT through the full GPU pipeline
go-gpu-miner --selftest

# benchmark end-to-end hashes
go-gpu-miner --bench 100

# mine (needs a DERO pool and your wallet)
go-gpu-miner -d <pool-host:port> -w <your-dero-wallet>
# or edit the blank config.json included in a release archive
```

Source checkouts provide `config.example.json`; copy it to the ignored
`config.json` before adding a wallet.

Flags: `-d` pool/daemon `host:port`, `-w` wallet, `-c` config path,
`--gpu high|low|software` (adapter selection), `--backend`,
`--sapath auto|spirv|refine|portable`, `--selftest`, `--bench N`,
`--benchpipe N` (streaming benchmark), and `--batch N`. When omitted, batch is
2,048 on native SPIR-V and 1,664 on WGSL.

## Layout

```
cmd/go-gpu-miner/   the miner
cmd/gpuprobe/       adapter + limits probe (M0)
cmd/tiedepth/       AstroBWTv3 tie-depth instrumentation (SA design calibration)
cmd/sabench/        SA-only throughput bench (median ms/hash; -maxrounds profiler)
cmd/stabbench/      streaming per-batch stability bench (spread% + median ms)
internal/gpu/       wgpu wrapper, WGSL kernels, embedded SPIR-V + Slang sources,
                    suffix arrays, streaming pipeline, parity tests
internal/astrobwt/  consensus-correct CPU oracle (copied from Dirtybird-Go-Miner)
internal/refpow/    second DERO-derived PoW implementation (verification oracle)
internal/getwork/   DERO getwork/submit WebSocket protocol
internal/{config,console,miner}/  host plumbing
```

## Verification

```
go test ./internal/gpu/     # GPU parity: crypto kernels, radix sort, suffix array, pipeline KAT
go test ./internal/astrobwt/ # CPU oracle incl. million-hash SA gate
go test ./...               # complete regression suite
```

All GPU kernels are parity-tested against the Go reference at live batch sizes —
the discipline that catches the class of silent-corruption bug where a gate tests
smaller-than-live conditions.

## License

The project is MIT-licensed; see `LICENSE`. BSD notices for the DERO AstroBWTv3
port and Go suffix-array code are retained under `internal/astrobwt/`. Native
shader attribution and dependency licenses are listed in
`THIRD-PARTY-LICENSES`.
