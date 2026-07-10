package miner

import (
	"strings"
	"testing"

	"github.com/Dirtybird99/Dirtybird-Go-GPU-Miner/internal/getwork"
)

// FuzzSetJob guards the miner's only untrusted-input parser. Jobs arrive over
// a websocket dialed with InsecureSkipVerify (the daemon uses a self-signed
// cert), so blockhashing_blob is attacker-reachable: an over-length blob once
// wrote past the fixed [48]byte through hex.Decode and panicked the getwork
// reader goroutine — a remote crash (fixed in 82cefbf). SetJob must never
// panic, and a rejected job must not leak state.
//
// Seeds run as ordinary tests on every `go test`; nightly CI explores with
// -fuzz=FuzzSetJob.
func FuzzSetJob(f *testing.F) {
	f.Add(strings.Repeat("a", MiniblockSize*2+64), uint64(1000))    // over-length (the crash)
	f.Add(strings.Repeat("a", 4096), uint64(1000))                  // way over
	f.Add(strings.Repeat("a", MiniblockSize*2-2), uint64(1000))     // under
	f.Add("", uint64(1000))                                         // empty
	f.Add(strings.Repeat("a", MiniblockSize*2+1), uint64(1000))     // odd length
	f.Add(strings.Repeat("zz", MiniblockSize), uint64(1000))        // non-hex
	f.Add("01"+strings.Repeat("00", MiniblockSize-1), uint64(1000)) // valid
	f.Add("01"+strings.Repeat("00", MiniblockSize-1), uint64(0))    // zero difficulty

	f.Fuzz(func(t *testing.T, blob string, diff uint64) {
		var s State
		s.Height.Store(7)
		changed, err := s.SetJob(getwork.Job{Blockhashing_blob: blob, Difficultyuint64: diff, Height: 999})
		if err != nil {
			if changed {
				t.Fatal("rejected job reported changed=true")
			}
			if s.Height.Load() != 7 {
				t.Fatal("rejected job mutated Height")
			}
			if s.Epoch() != 0 {
				t.Fatal("rejected job bumped the epoch")
			}
			return
		}
		// Accepted: the job must round-trip into a consistent snapshot.
		blobOut, _, _, epoch := s.Job()
		if epoch == 0 {
			t.Fatal("accepted job left epoch at 0")
		}
		_ = blobOut
	})
}
