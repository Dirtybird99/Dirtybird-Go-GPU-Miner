---
name: Dirtybird Go GPU Miner
last_updated: 2026-07-10
---

# Dirtybird Go GPU Miner Strategy

## Product

A pure-Go, zero-CGO AstroBWTv3 GPU miner. Go owns the miner, networking,
scheduling, and consensus guard. Precompiled SPIR-V supplies the measured
NVIDIA/Vulkan fast path; WGSL supplies the other eligible GPU paths.

## Non-negotiable correctness

- Run `pow("a")` and the adversarial suffix-array corpus at startup.
- Re-hash every target-meeting candidate with the CPU oracle before submission.
- Refuse to mine when the selected GPU/backend path fails its gate.
- Record real backend results in `docs/backends.jsonl`; prose cannot substitute
  for a passing artifact.

## Current measured state

- NVIDIA RTX 4070 Laptop / Vulkan / subgroup 32: native SPIR-V passes and the
  production stream sustains about 12.34 KH/s.
- NVIDIA and Intel Vulkan WGSL paths have recorded passing configurations.
- DX12, GL, and the software renderer have recorded failures and are not mining
  targets in the current wgpu version.
- AMD/Vulkan and Apple/Metal remain unverified until real hardware passes the
  exact startup and streaming gates.

## Priorities

1. Preserve byte-exact consensus behavior and the CPU submission guard.
2. Keep the NVIDIA native stream at or above 12 KH/s on the reference laptop.
3. Qualify additional hardware with recorded self-tests before advertising it.
4. Improve unattended operation only when live mining exposes a concrete need.

## Distribution rule

Ship one Go binary per OS/architecture. Slang is a developer-only tool;
release users receive the embedded SPIR-V modules and need only a supported GPU
driver. Build-only assets must be labeled as such until verified on hardware.
