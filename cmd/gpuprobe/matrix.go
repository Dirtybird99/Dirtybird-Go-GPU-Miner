package main

import (
	"fmt"
	"os"
	"os/exec"
	"strings"
	"time"

	"github.com/gogpu/wgpu"

	"github.com/Dirtybird99/Dirtybird-Go-GPU-Miner/internal/gpu"
)

// fusedStorageBuffers is how many storage buffers the fused pipeline binds. A
// device granting fewer cannot build a fused session at all, no matter how much
// memory it has.
const fusedStorageBuffers = 9

// batch512Binding is the largest single storage binding the default mining
// batch needs: 512 hashes * saStride(70912) * 4 bytes.
const batch512Binding = 512 * 70912 * 4

type combo struct {
	backend string
	power   string
}

var combos = []combo{
	{"vulkan", "high"}, // discrete
	{"vulkan", "low"},  // integrated, if any
	{"dx12", "high"},   //
	{"dx12", "low"},    //
	{"gl", "high"},     //
	{"software", "software"},
}

// runMatrix sweeps every backend, each in its own subprocess. Isolation is not
// a nicety: the gles backend faults inside the driver on device release, and an
// in-process sweep loses every result after it.
func runMatrix() {
	self, err := os.Executable()
	if err != nil {
		fmt.Printf("cannot locate self: %v\n", err)
		return
	}
	fmt.Printf("batch-512 needs one %d MiB storage binding and %d storage buffers/stage\n\n",
		batch512Binding>>20, fusedStorageBuffers)

	for _, c := range combos {
		fmt.Printf("=== %s / power=%s ===\n", c.backend, c.power)
		cmd := exec.Command(self, "-one", c.backend, "-power", c.power)
		out, err := cmd.CombinedOutput()
		text := strings.TrimRight(string(out), "\n")
		// The probe's own lines are prefixed with two spaces; driver logs and
		// panic dumps are not, so keep only what we printed plus a verdict.
		for _, ln := range strings.Split(text, "\n") {
			if strings.HasPrefix(ln, "  ") {
				fmt.Println(ln)
			}
		}
		switch {
		case err != nil && cmd.ProcessState != nil && cmd.ProcessState.ExitCode() == 1:
			fmt.Printf("  RESULT: unavailable\n")
		case err != nil:
			fmt.Printf("  RESULT: CRASHED (%v)\n", err)
		default:
			fmt.Printf("  RESULT: ok\n")
		}
		fmt.Println()
	}
}

// probeOne reports, for a single backend, the three facts that decide what this
// machine can verify: does the fast SA kernel compile, is there a subgroup wide
// enough to run it, and will the device grant the limits the mining batch needs.
func probeOne(backend, power string) {
	var p gpu.Power
	switch power {
	case "low":
		p = gpu.PowerLow
	case "software":
		p = gpu.PowerSoftware
	default:
		p = gpu.PowerHigh
	}

	ctx, err := gpu.NewCtxBackend(p, gpu.ParseBackend(backend))
	if err != nil {
		fmt.Printf("  no adapter: %v\n", err)
		os.Exit(1)
	}
	info := ctx.Info()
	lim := ctx.Device.Limits()
	fmt.Printf("  adapter:  %s (%s, %v, backend=%v)\n", info.Name, info.Vendor, info.DeviceType, info.Backend)
	fmt.Printf("  driver:   %s %s\n", info.Driver, info.DriverInfo)
	fmt.Printf("  limits:   storageBinding=%d MiB bufferSize=%d MiB storageBufs/stage=%d\n",
		lim.MaxStorageBufferBindingSize>>20, lim.MaxBufferSize>>20, lim.MaxStorageBuffersPerShaderStage)

	fits := lim.MaxStorageBufferBindingSize >= batch512Binding &&
		uint64(lim.MaxStorageBuffersPerShaderStage) >= fusedStorageBuffers
	fmt.Printf("  batch512: %s\n", okStr(fits, "bindable", "NOT bindable"))

	t0 := time.Now()
	sg := ctx.SubgroupSize()
	fmt.Printf("  subgroup: %d %s (probe %v)\n", sg, okStr(sg >= 32, "(refine reachable)", "(portable only)"), time.Since(t0).Round(time.Millisecond))

	// Compilation only — never dispatched. A backend can compile a kernel it then
	// miscomputes (the software backend does exactly that), so this is a
	// necessary-not-sufficient check.
	for _, k := range []struct{ name, src string }{
		{"sa_refine.wgsl", gpu.SARefineWGSL},
		{"sa_gpu.wgsl", gpu.SAGpuWGSL},
		{"fronthalf.wgsl", gpu.U64WGSL + "\n" + gpu.FrontHalfWGSL},
	} {
		mod, err := ctx.Device.CreateShaderModule(&wgpu.ShaderModuleDescriptor{Label: k.name, WGSL: k.src})
		if err != nil {
			fmt.Printf("  compile %-16s FAIL: %v\n", k.name, err)
			continue
		}
		mod.Release()
		fmt.Printf("  compile %-16s ok\n", k.name)
	}

	// gles faults inside the driver on release; skip teardown there rather than
	// lose the report we just printed.
	if backend != "gl" {
		ctx.Close()
	}
}

func okStr(ok bool, yes, no string) string {
	if ok {
		return yes
	}
	return no
}
