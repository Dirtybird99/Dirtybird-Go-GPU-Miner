package main

import (
	"context"
	"strings"
	"testing"
	"time"

	"github.com/Dirtybird99/Dirtybird-Go-GPU-Miner/internal/getwork"
	"github.com/Dirtybird99/Dirtybird-Go-GPU-Miner/internal/miner"
)

// testJob builds a job that passes SetJob validation: zero blob except the
// version nibble, nonzero difficulty.
func testJob(jobid string) getwork.Job {
	return getwork.Job{
		Blockhashing_blob: "01" + strings.Repeat("00", miner.MiniblockSize-1),
		Difficultyuint64:  1000,
		JobID:             jobid,
	}
}

func testState(t *testing.T) *miner.State {
	t.Helper()
	st := &miner.State{}
	if _, err := st.SetJob(testJob("j1")); err != nil {
		t.Fatalf("SetJob: %v", err)
	}
	return st
}

// fullMailbox returns a submits channel with no free slot, so submitShare
// lands in its backpressure path immediately.
func fullMailbox() chan getwork.Submit {
	ch := make(chan getwork.Submit, 1)
	ch <- getwork.Submit{}
	return ch
}

// TestSubmitShareAbandonsOnRotation: once the job epoch rotates, a waiting
// share can no longer earn and must be abandoned promptly — not held for a
// fixed wall-clock timeout that ignores the job's lifetime.
func TestSubmitShareAbandonsOnRotation(t *testing.T) {
	st := testState(t)
	epoch := st.Epoch()
	submits := fullMailbox()
	env := &mineEnv{st: st, submits: submits, connected: func() bool { return true }}

	go func() {
		time.Sleep(50 * time.Millisecond)
		if _, err := st.SetJob(testJob("j2")); err != nil {
			panic(err)
		}
	}()

	start := time.Now()
	submitShare(context.Background(), env, make([]byte, miner.MiniblockSize), "j1", epoch)
	if el := time.Since(start); el > time.Second {
		t.Fatalf("submitShare held a dead share for %v", el)
	}
	if got := st.Stale.Load(); got != 1 {
		t.Errorf("Stale = %d, want 1", got)
	}
	if got := st.Submitted.Load(); got != 0 {
		t.Errorf("Submitted = %d, want 0", got)
	}
}

// TestSubmitShareAbandonsOnDisconnect: while the link is down the epoch never
// rotates (no jobs arrive), so the connected check is the only thing standing
// between an outage and the mine loop parking here forever.
func TestSubmitShareAbandonsOnDisconnect(t *testing.T) {
	st := testState(t)
	submits := fullMailbox()
	env := &mineEnv{st: st, submits: submits, connected: func() bool { return false }}

	start := time.Now()
	submitShare(context.Background(), env, make([]byte, miner.MiniblockSize), "j1", st.Epoch())
	if el := time.Since(start); el > time.Second {
		t.Fatalf("submitShare blocked %v with the link down", el)
	}
	if got := st.Stale.Load(); got != 1 {
		t.Errorf("Stale = %d, want 1", got)
	}
}

// TestSubmitShareWaitsWhileEarnable is the money test: with the job live and
// the link up, a share must survive a stall longer than the old 2-second
// deadline and be delivered when the mailbox frees up.
func TestSubmitShareWaitsWhileEarnable(t *testing.T) {
	st := testState(t)
	submits := fullMailbox()
	env := &mineEnv{st: st, submits: submits, connected: func() bool { return true }}

	go func() {
		time.Sleep(2500 * time.Millisecond) // past the old 2s drop deadline
		<-submits
	}()

	submitShare(context.Background(), env, make([]byte, miner.MiniblockSize), "j1", st.Epoch())
	if got := st.Submitted.Load(); got != 1 {
		t.Errorf("Submitted = %d, want 1 (share was dropped despite still being earnable)", got)
	}
	if got := st.Stale.Load(); got != 0 {
		t.Errorf("Stale = %d, want 0", got)
	}
}

// TestClassify pins the gate's three outcomes and its evaluation order.
func TestClassify(t *testing.T) {
	rehash := func(ok bool) func() bool { return func() bool { return ok } }

	if got := classify(1, 1, rehash(true)); got != gateSubmit {
		t.Errorf("fresh+match = %v, want gateSubmit", got)
	}
	if got := classify(1, 1, rehash(false)); got != gateMiscompute {
		t.Errorf("fresh+mismatch = %v, want gateMiscompute", got)
	}
	if got := classify(2, 1, rehash(true)); got != gateStale {
		t.Errorf("rotated = %v, want gateStale", got)
	}

	// Ordering: a rotated batch must never pay for the re-hash (a full
	// AstroBWTv3 hash) — the whole point of checking the atomic epoch first.
	called := false
	classify(2, 1, func() bool { called = true; return true })
	if called {
		t.Error("classify ran the re-hash for a stale candidate")
	}
}

// TestOnMiscomputeCounts: a miscompute with no pipe (unit-test env) logs and
// returns; the counter increment happens at the call site, so here we only
// pin that onMiscompute itself is safe without a GPU.
func TestOnMiscomputeNoPipe(t *testing.T) {
	env := &mineEnv{st: &miner.State{}, connected: func() bool { return true }}
	env.onMiscompute("j1", 3) // must not panic without a pipeline
	env.onMiscompute("j1", 4)
}

// TestSubmitShareFastPath: an open mailbox slot queues without waiting.
func TestSubmitShareFastPath(t *testing.T) {
	st := testState(t)
	submits := make(chan getwork.Submit, 1)
	env := &mineEnv{st: st, submits: submits, connected: func() bool { return true }}

	submitShare(context.Background(), env, make([]byte, miner.MiniblockSize), "j1", st.Epoch())
	if got := st.Submitted.Load(); got != 1 {
		t.Errorf("Submitted = %d, want 1", got)
	}
	select {
	case <-submits:
	default:
		t.Error("share not in the mailbox")
	}
}
