// Command verifybackends measures STRATEGY.md's "Backends verified" metric:
// it runs the miner's --selftest across every backend x SA-path combination
// this machine can reach and appends one machine-readable record per combo to
// docs/backends.jsonl. Each combo runs in a SUBPROCESS because backends can
// take the whole process down (the gles backend faults in the driver on device
// release; DX12 removes devices) — one hostile driver must not void the matrix.
package main

import (
	"bufio"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"os/exec"
	"strings"
	"time"
)

type record struct {
	Name       string `json:"name"`
	Backend    string `json:"backend"`
	Subgroup   uint32 `json:"subgroup"`
	Path       string `json:"path"`
	Result     string `json:"result"`
	Error      string `json:"error,omitempty"`
	Ms         int64  `json:"ms"`
	Commit     string `json:"commit,omitempty"`
	MeasuredAt string `json:"measured_at"`
}

type combo struct {
	backend, power, sapath string
}

// The matrix: every backend registered on Windows, both adapter classes where
// they differ (high = discrete, low = integrated), both portable SA paths,
// plus the native SPIR-V combination it supports. Impossible combinations
// come back as "skip" from the miner itself.
var combos = []combo{
	{"vulkan", "high", "spirv"},
	{"vulkan", "high", "refine"},
	{"vulkan", "high", "portable"},
	{"vulkan", "low", "refine"},
	{"vulkan", "low", "portable"},
	{"dx12", "high", "refine"},
	{"dx12", "high", "portable"},
	{"gl", "high", "refine"},
	{"gl", "high", "portable"},
	{"software", "software", "refine"},
	{"software", "software", "portable"},
}

func main() {
	bin := flag.String("bin", "./go-gpu-miner.exe", "path to the built miner")
	out := flag.String("out", "docs/backends.jsonl", "verification log to append to (empty = stdout only)")
	timeout := flag.Duration("timeout", 3*time.Minute, "per-combo time budget")
	flag.Parse()

	var records []record
	for _, c := range combos {
		fmt.Fprintf(os.Stderr, "=== %s / %s / %s ===\n", c.backend, c.power, c.sapath)
		rec := runCombo(*bin, c, *timeout)
		fmt.Fprintf(os.Stderr, "    %-4s %s (%d ms) %s\n", rec.Result, rec.Name, rec.Ms, rec.Error)
		records = append(records, rec)
	}

	if *out != "" {
		f, err := os.OpenFile(*out, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
		if err != nil {
			fmt.Fprintf(os.Stderr, "open %s: %v\n", *out, err)
			os.Exit(1)
		}
		enc := json.NewEncoder(f)
		for _, r := range records {
			if err := enc.Encode(r); err != nil {
				fmt.Fprintf(os.Stderr, "append: %v\n", err)
				os.Exit(1)
			}
		}
		f.Close()
	}

	// Summary: the metric is distinct (device, backend) pairs with >= 1 pass.
	verified := map[string]bool{}
	for _, r := range records {
		if r.Result == "pass" {
			verified[r.Name+" / "+r.Backend] = true
		}
	}
	fmt.Fprintf(os.Stderr, "\nBackends verified on this machine: %d\n", len(verified))
	for k := range verified {
		fmt.Fprintf(os.Stderr, "  PASS %s\n", k)
	}
}

// runCombo runs one --selftest subprocess and turns whatever happened into a
// record. The miner prints exactly one '{'-prefixed line on stdout when -json
// is set; a missing line means the process crashed or hung, which is a FAIL of
// that combo, never of the sweep.
func runCombo(bin string, c combo, timeout time.Duration) record {
	fallback := record{
		Backend: c.backend, Path: c.sapath, Result: "fail",
		MeasuredAt: time.Now().Format("2006-01-02"),
	}

	cmd := exec.Command(bin, "-selftest", "-json",
		"-backend", c.backend, "-gpu", c.power, "-sapath", c.sapath)
	outPipe, err := cmd.StdoutPipe()
	if err != nil {
		fallback.Error = err.Error()
		return fallback
	}
	cmd.Stderr = nil // driver spam and panics are the subprocess's problem
	if err := cmd.Start(); err != nil {
		fallback.Error = err.Error()
		return fallback
	}

	// Scan stdout for the record while enforcing the budget.
	found := make(chan record, 1)
	go func() {
		sc := bufio.NewScanner(outPipe)
		for sc.Scan() {
			line := strings.TrimSpace(sc.Text())
			if !strings.HasPrefix(line, "{") {
				continue
			}
			var r record
			if json.Unmarshal([]byte(line), &r) == nil {
				found <- r
				return
			}
		}
	}()

	done := make(chan error, 1)
	go func() { done <- cmd.Wait() }()

	var rec record
	var got bool
	select {
	case rec = <-found:
		got = true
		select { // let the process finish (or crash in teardown) briefly
		case <-done:
		case <-time.After(10 * time.Second):
			_ = cmd.Process.Kill()
			<-done
		}
	case err := <-done:
		// Exited without a record: crash before the gate finished.
		select {
		case rec = <-found:
			got = true
		default:
			fallback.Error = fmt.Sprintf("no record: %v", err)
		}
	case <-time.After(timeout):
		_ = cmd.Process.Kill()
		<-done
		fallback.Error = fmt.Sprintf("timeout after %v", timeout)
	}
	if !got {
		return fallback
	}
	return rec
}
