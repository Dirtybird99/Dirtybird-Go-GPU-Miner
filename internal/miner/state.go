package miner

import (
	"encoding/hex"
	"errors"
	"sync"
	"sync/atomic"

	"github.com/Dirtybird99/Dirtybird-Go-GPU-Miner/internal/getwork"
)

// MiniblockSize is derohe block.MINIBLOCK_SIZE.
const MiniblockSize = 48

var (
	ErrBadBlob    = errors.New("blockhashing_blob is not 48 bytes")
	ErrBadVersion = errors.New("unsupported miniblock version, check for a miner update")
	ErrBadDiff    = errors.New("job difficulty is zero")
)

// State is the shared job snapshot plus counters (the zig miner's state.zig
// model). Workers poll Epoch and re-snapshot the job when it changes.
type State struct {
	mu     sync.Mutex
	blob   [MiniblockSize]byte
	jobid  string
	target [32]byte
	epoch  atomic.Uint64 // bumped on every job change; 0 = no job yet

	Height atomic.Uint64
	Diff   atomic.Uint64

	// local counters
	TotalHashes atomic.Uint64
	Submitted   atomic.Uint64
	Stale       atomic.Uint64 // submits dropped because the mailbox was full

	// Miscompute counts target-meeting candidates the CPU re-hash gate rejected:
	// the GPU produced a pow the CPU could not reproduce. This is the
	// portability bet failing on this device, so it is a loud event. Note the
	// count only samples candidates that met target (rare), so as a rate it is a
	// biased lower bound on the true per-hash miscompute rate — measuring that
	// would mean re-hashing every candidate on the CPU, defeating the GPU
	// offload. Metric: Miscompute / (Miscompute + Submitted).
	Miscompute atomic.Uint64

	// GateStale counts target-meeting candidates dropped because the job rotated
	// between the batch epoch snapshot and the gate — benign, and near-zero in
	// practice because processBatch discards rotated batches wholesale first.
	GateStale atomic.Uint64

	// mirrored from job pushes (authoritative accept/reject accounting)
	Blocks     atomic.Uint64
	MiniBlocks atomic.Uint64
	Rejected   atomic.Uint64
}

// SetJob validates and installs a pushed job. It mirrors the daemon counters
// for every job that passes validation (even when the work itself is
// unchanged); it bumps the epoch only when the work changed.
func (s *State) SetJob(j getwork.Job) (changed bool, err error) {
	// Length MUST be checked before decoding: hex.Decode does not bounds-check
	// its destination, so an over-length blob would write past blob[47] and panic
	// the getwork reader goroutine (a malformed or MITM-injected job could crash
	// the whole miner).
	if len(j.Blockhashing_blob) != MiniblockSize*2 {
		return false, ErrBadBlob
	}
	var blob [MiniblockSize]byte
	n, err := hex.Decode(blob[:], []byte(j.Blockhashing_blob))
	if err != nil || n != MiniblockSize {
		return false, ErrBadBlob
	}
	if blob[0]&0xf != 1 { // derohe miner.go version-nibble check
		return false, ErrBadVersion
	}
	if j.Difficultyuint64 == 0 {
		return false, ErrBadDiff
	}

	// Mirror the daemon counters only now that the job validated: a malformed or
	// hostile push must not be able to poison the status line's Height/Diff.
	s.Height.Store(j.Height)
	s.Diff.Store(j.Difficultyuint64)
	s.Blocks.Store(j.Blocks)
	s.MiniBlocks.Store(j.MiniBlocks)
	s.Rejected.Store(j.Rejected)

	s.mu.Lock()
	defer s.mu.Unlock()
	if blob == s.blob && j.JobID == s.jobid {
		return false, nil
	}
	s.blob = blob
	s.jobid = j.JobID
	s.target = ComputeTarget(j.Difficultyuint64)
	s.epoch.Add(1)
	return true, nil
}

// Job returns a consistent snapshot of the current work.
func (s *State) Job() (blob [MiniblockSize]byte, jobid string, target [32]byte, epoch uint64) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.blob, s.jobid, s.target, s.epoch.Load()
}

// Epoch is the cheap per-hash staleness check.
func (s *State) Epoch() uint64 { return s.epoch.Load() }
