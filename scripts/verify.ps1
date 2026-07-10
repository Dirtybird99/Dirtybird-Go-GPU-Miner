# The local release gate — the ONLY place kernel correctness is verified.
# Cloud CI has no GPU and the software backend cannot stand in for one, so
# run this on a real GPU (and paste its output) before tagging a release.
#
#   pwsh scripts/verify.ps1
#
# What it runs:
#   1. The full internal/gpu suite (not -short): SA parity vs the SAIS oracle
#      at the live batch size, the adversarial synthetic corpus, big-batch
#      dispatch sizing, the fused + streaming paths.
#   2. The whole repo's tests.
#   3. The backend sweep, refreshing docs/backends.jsonl — commit the diff so
#      the "Backends verified" artifact tracks this build.

$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

$commit = (git rev-parse --short HEAD).Trim()
Write-Host "== verify @ $commit ==" -ForegroundColor Cyan

Write-Host "`n-- full GPU suite --" -ForegroundColor Cyan
go test ./internal/gpu -timeout 30m
if ($LASTEXITCODE -ne 0) { throw "internal/gpu FAILED" }

Write-Host "`n-- full repo tests --" -ForegroundColor Cyan
go test ./... -timeout 30m
if ($LASTEXITCODE -ne 0) { throw "repo tests FAILED" }

Write-Host "`n-- backend sweep -> docs/backends.jsonl --" -ForegroundColor Cyan
go build -ldflags "-X main.commit=$commit" -o go-gpu-miner.exe ./cmd/go-gpu-miner
if ($LASTEXITCODE -ne 0) { throw "miner build FAILED" }
go run ./cmd/verifybackends -bin ./go-gpu-miner.exe -out docs/backends.jsonl
if ($LASTEXITCODE -ne 0) { throw "backend sweep FAILED" }

Write-Host "`n== verify PASSED @ $commit — commit the docs/backends.jsonl diff ==" -ForegroundColor Green
