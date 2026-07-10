// Package backends holds no code — only the doc-vs-artifact linter that keeps
// README/DEPLOY/STRATEGY from asserting backend support nobody verified.
package backends

import (
	"bufio"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// record mirrors the miner's -selftest -json output (one docs/backends.jsonl
// line per verified backend x path combo).
type record struct {
	Name    string `json:"name"`
	Backend string `json:"backend"`
	Path    string `json:"path"`
	Result  string `json:"result"`
}

const artifact = "docs/backends.jsonl"

var docs = []string{"README.md", "DEPLOY.md", "STRATEGY.md"}

// keywords maps a backend/vendor word a doc might claim to the predicate a
// pass record must satisfy to back the claim.
var keywords = map[string]func(r record) bool{
	"NVIDIA": func(r record) bool { return strings.Contains(r.Name, "NVIDIA") },
	"AMD":    func(r record) bool { return strings.Contains(r.Name, "AMD") || strings.Contains(r.Name, "Radeon") },
	"Intel":  func(r record) bool { return strings.Contains(r.Name, "Intel") },
	"Apple":  func(r record) bool { return strings.Contains(r.Name, "Apple") },
	"Metal":  func(r record) bool { return r.Backend == "Metal" },
	"DX12":   func(r record) bool { return r.Backend == "DX12" },
	"Vulkan": func(r record) bool { return r.Backend == "Vulkan" },
	"software": func(r record) bool {
		return strings.Contains(strings.ToLower(r.Name), "software")
	},
}

func repoRoot(t *testing.T) string {
	t.Helper()
	root, err := filepath.Abs(filepath.Join("..", ".."))
	if err != nil {
		t.Fatal(err)
	}
	return root
}

func loadPasses(t *testing.T) []record {
	t.Helper()
	f, err := os.Open(filepath.Join(repoRoot(t), artifact))
	if err != nil {
		t.Fatalf("no verification artifact: %v — run cmd/verifybackends to produce it; "+
			"docs may not claim verified backends without it", err)
	}
	defer f.Close()
	var passes []record
	sc := bufio.NewScanner(f)
	line := 0
	for sc.Scan() {
		line++
		txt := strings.TrimSpace(sc.Text())
		if txt == "" {
			continue
		}
		var r record
		if err := json.Unmarshal([]byte(txt), &r); err != nil {
			t.Fatalf("%s:%d: bad record: %v", artifact, line, err)
		}
		if r.Result == "pass" {
			passes = append(passes, r)
		}
	}
	if len(passes) == 0 {
		t.Fatalf("%s holds no pass records — nothing is verified", artifact)
	}
	return passes
}

// TestDocsClaimOnlyVerifiedBackends scans the user-facing docs for lines that
// assert a backend is verified/validated and requires a pass record backing
// every backend named on such a line. Honest negations ("not verified",
// "unverified", "no ... yet") and pointers at the artifact are exempt. This is
// what makes "Backends verified" an artifact instead of prose.
func TestDocsClaimOnlyVerifiedBackends(t *testing.T) {
	passes := loadPasses(t)
	backed := func(pred func(record) bool) bool {
		for _, r := range passes {
			if pred(r) {
				return true
			}
		}
		return false
	}

	for _, doc := range docs {
		body, err := os.ReadFile(filepath.Join(repoRoot(t), doc))
		if err != nil {
			t.Errorf("%s: %v", doc, err)
			continue
		}
		for i, line := range strings.Split(string(body), "\n") {
			low := strings.ToLower(line)
			if !strings.Contains(low, "verified") && !strings.Contains(low, "validated") {
				continue
			}
			// Honest phrasing is exempt: negations, failure statements, and
			// pointers at the artifact itself.
			if strings.Contains(low, "not verified") || strings.Contains(low, "unverified") ||
				strings.Contains(low, "not validated") || strings.Contains(low, "backends.jsonl") ||
				strings.Contains(low, "no ") || strings.Contains(low, "yet") ||
				strings.Contains(low, "fail") {
				continue
			}
			for kw, pred := range keywords {
				if !strings.Contains(line, kw) {
					continue
				}
				if !backed(pred) {
					t.Errorf("%s:%d claims %q is verified but %s has no pass record for it:\n  %s",
						doc, i+1, kw, artifact, strings.TrimSpace(line))
				}
			}
		}
	}
}

// TestDocsNoKnownFalseClaims forbids specific claims the code disproved:
// wgpu v0.30.10's software HAL runs invocations sequentially with no-op
// barriers, so no cooperative kernel is correct there.
func TestDocsNoKnownFalseClaims(t *testing.T) {
	for _, doc := range docs {
		body, err := os.ReadFile(filepath.Join(repoRoot(t), doc))
		if err != nil {
			t.Errorf("%s: %v", doc, err)
			continue
		}
		if strings.Contains(string(body), "correct on any GPU") {
			t.Errorf("%s claims %q — disproved: the software backend miscomputes every cooperative kernel", doc, "correct on any GPU")
		}
	}
}
