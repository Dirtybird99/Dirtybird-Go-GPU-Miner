package gpu

import _ "embed"

// AstroBWTv3 kernels are kept as .wgsl files so they can be linted/edited as
// shader source, and embedded so the miner stays a single binary.

//go:embed wgsl/common.wgsl
var CommonWGSL string

//go:embed wgsl/sha256.wgsl
var SHA256WGSL string

//go:embed wgsl/salsa20.wgsl
var Salsa20WGSL string

//go:embed wgsl/rc4.wgsl
var RC4WGSL string

//go:embed wgsl/sa_gpu.wgsl
var SAGpuWGSL string

//go:embed wgsl/sa_refine.wgsl
var SARefineWGSL string

//go:embed wgsl/u64.wgsl
var U64WGSL string

//go:embed wgsl/fnv1a64.wgsl
var Fnv1a64WGSL string

//go:embed wgsl/siphash.wgsl
var SipHashWGSL string

//go:embed wgsl/xxhash64.wgsl
var XXHash64WGSL string

//go:embed wgsl/fronthalf.wgsl
var FrontHalfWGSL string
