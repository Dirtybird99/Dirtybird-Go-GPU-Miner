package miner

import (
	"strings"
	"testing"

	"github.com/Dirtybird99/Dirtybird-Go-GPU-Miner/internal/getwork"
)

// TestSetJobMalformedBlob guards the getwork reader against a crafted or
// MITM-injected job: SetJob must reject an over-/under-length or non-hex
// blockhashing_blob with an error, never panic (hex.Decode does not bounds-check
// its destination, so an over-length blob once wrote past the fixed array).
func TestSetJobMalformedBlob(t *testing.T) {
	cases := []struct {
		name string
		blob string
	}{
		{"over-length", strings.Repeat("a", MiniblockSize*2+64)},
		{"way-over-length", strings.Repeat("a", 4096)},
		{"under-length", strings.Repeat("a", MiniblockSize*2-2)},
		{"empty", ""},
		{"odd-length", strings.Repeat("a", MiniblockSize*2+1)},
		{"non-hex", strings.Repeat("zz", MiniblockSize)},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			var s State
			s.Height.Store(100)
			s.Diff.Store(500)
			// Must return an error and must not panic.
			changed, err := s.SetJob(getwork.Job{Blockhashing_blob: tc.blob, Difficultyuint64: 1000, Height: 999})
			if err == nil {
				t.Fatalf("SetJob accepted malformed blob (len=%d), want error", len(tc.blob))
			}
			if changed {
				t.Fatalf("SetJob reported changed=true for a rejected job")
			}
			// A rejected job must not poison the status line: the daemon counters
			// mirror only jobs that validated.
			if h := s.Height.Load(); h != 100 {
				t.Errorf("rejected job overwrote Height: got %d, want 100", h)
			}
			if d := s.Diff.Load(); d != 500 {
				t.Errorf("rejected job overwrote Diff: got %d, want 500", d)
			}
		})
	}
}

// ValidJob builds a job that passes SetJob validation, for tests that need to
// drive the epoch. The blob is all zeros except the version nibble.
func validJob(jobid string) getwork.Job {
	return getwork.Job{
		Blockhashing_blob: "01" + strings.Repeat("00", MiniblockSize-1),
		Difficultyuint64:  1000,
		JobID:             jobid,
		Height:            42,
	}
}

// TestSetJobMirrorsCountersWhenUnchanged pins the mirror-always semantics: a
// re-pushed identical job must not bump the epoch but must still refresh the
// daemon accounting (Height climbs on every push even when the work doesn't).
func TestSetJobMirrorsCountersWhenUnchanged(t *testing.T) {
	var s State
	if changed, err := s.SetJob(validJob("j1")); err != nil || !changed {
		t.Fatalf("first SetJob: changed=%v err=%v", changed, err)
	}
	epoch := s.Epoch()

	j := validJob("j1")
	j.Height = 43
	changed, err := s.SetJob(j)
	if err != nil {
		t.Fatalf("re-push: %v", err)
	}
	if changed {
		t.Errorf("identical work reported changed=true")
	}
	if s.Epoch() != epoch {
		t.Errorf("identical work bumped the epoch")
	}
	if h := s.Height.Load(); h != 43 {
		t.Errorf("unchanged job did not mirror Height: got %d, want 43", h)
	}
}
