# Security Policy

## Reporting a vulnerability

Do not open a public issue for a security vulnerability. Use the repository's
private **Security → Report a vulnerability** form instead.

Do not include real wallet addresses, credentials, or private network details
in reports or reproduction files.

## Supported version

Only the latest release and the current `main` branch are supported.

## Scope

Reports should reproduce against a build made with Go 1.25 or newer. Include
the GPU model, driver, backend, selected SA path, and whether `--selftest`
passes. GPU or driver combinations that fail the startup correctness gate are
unsupported until a recorded backend verification passes.
