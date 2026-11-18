#!/usr/bin/env bash
# Delete training run directories ONLY when no run is actively writing to them.
#
# Why this exists (incident 2026-11-18):
#   A cleanup used a timestamp glob, `rm -rf logs/rsl_rl/<exp>/2026-11-18_02-*`,
#   which matched a run that had been started at 02:56 and was still training.
#   TensorBoard does not keep the event file open for its whole lifetime: every
#   flush calls `io.open(filename, "ab")` again (tensorboard/summary/writer/
#   io/gfile.py:171). With the directory gone that `open` raises
#   FileNotFoundError, the EventFileWriter worker thread stores it, and the main
#   thread re-raises it from `_check_worker_status()` inside `runner.learn()`.
#   The run dies minutes after the deletion, not immediately, so the mistake is
#   easy to miss. The checkpoints were in the deleted directory and were lost.
#
#   Rule encoded below: never infer "these runs are stale" from a timestamp
#   glob. Reason about live writers first, then delete exact directories.
#
# Usage:
#   bash scripts/safe_clean_runs.sh <run-dir> [<run-dir> ...]
#   bash scripts/safe_clean_runs.sh --before 2026-11-18_03:00 <exp-dir>
#   bash scripts/safe_clean_runs.sh --dry-run <run-dir>
#
# <run-dir> is one specific timestamped directory such as
#   logs/rsl_rl/xgoduck_velocity/2026-11-18_02-29-01_xgoduck
# Globs are refused on purpose; expanding a glob is how the incident happened.
# Absolute paths inside this repository and paths written with a "./" prefix are
# accepted and normalised to that repository-relative form. A path nested deeper
# than logs/rsl_rl/<exp>/<run> is refused: deleting a run's subdirectory would
# leave the run alive and broken.
#
# Exit codes: 0 deleted (or nothing to do), 1 refused, 2 bad usage.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# A run counts as live if any of its files was written this recently. Training
# flushes the event file every iteration, so the window only has to cover one
# slow iteration plus scheduler jitter, not a whole run.
ACTIVE_WINDOW_SECONDS="${ACTIVE_WINDOW_SECONDS:-1800}"
DRY_RUN=0
BEFORE=""
TARGETS=()

# The usage text is the header comment above; printing it from there keeps a
# single source of truth instead of a line range that drifts when the header
# grows (the previous `sed -n '18,36p'` printed code).
usage() {
  awk '
    /^# Usage:/ { printing = 1 }
    printing && /^#/ { sub(/^# ?/, ""); print; next }
    printing { exit }
  ' "${BASH_SOURCE[0]}"
}

require_value() {
  if [[ $# -lt 2 || -z "$2" ]]; then
    echo "error: $1 needs a value" >&2
    exit 2
  fi
}

# A junk window silently matched nothing, and "matched nothing" used to mean
# "no recent writes", so the liveness guard failed open. Validate it up front.
validate_active_window() {
  if [[ ! "$ACTIVE_WINDOW_SECONDS" =~ ^[0-9]+$ ]]; then
    echo "error: ACTIVE_WINDOW_SECONDS must be whole seconds, got" \
      "'$ACTIVE_WINDOW_SECONDS'" >&2
    exit 2
  fi
}

# The script runs from the repository root, so compare everything in that frame
# while still accepting what a human actually types.
normalize_target() {
  local target="$1"
  if [[ "$target" == "$ROOT"/* ]]; then
    target="${target#"$ROOT"/}"
  fi
  target="${target#./}"
  target="${target%/}"
  printf '%s\n' "$target"
}

parse_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --dry-run)
        DRY_RUN=1
        shift
        ;;
      --before)
        require_value "$@"
        BEFORE="$2"
        shift 2
        ;;
      -h|--help)
        usage
        exit 0
        ;;
      -*)
        echo "error: unknown option $1" >&2
        exit 2
        ;;
      *)
        TARGETS+=("$(normalize_target "$1")")
        shift
        ;;
    esac
  done
}

refuse() {
  echo "REFUSED: $*" >&2
  exit 1
}

# Glob characters mean the caller is guessing which runs are stale. That is the
# exact reasoning that destroyed a live run, so it is rejected outright.
check_no_glob() {
  local target
  for target in "${TARGETS[@]+"${TARGETS[@]}"}"; do
    case "$target" in
      *'*'*|*'?'*|*'['*)
        refuse "target '$target' contains a glob. Name the exact run directory."
        ;;
    esac
  done
}

check_training_not_running() {
  local pattern='uv run train|mjlab[./]scripts[./]train|bin/train '
  if ! command -v pgrep >/dev/null 2>&1; then
    echo "warning: pgrep not found; relying on file timestamps only." >&2
    return 0
  fi
  if pgrep -af "$pattern" >/dev/null 2>&1; then
    echo "Active training processes:" >&2
    pgrep -af "$pattern" >&2
    refuse "a training process is running; stop it before deleting any run dir."
  fi
}

# The incident's real signal: files touched moments ago inside the directory.
# A failed scan must refuse rather than delete: swallowing the error turned "I
# could not look" into "nothing was written recently".
check_not_recently_written() {
  local target="$1"
  local recent
  if ! recent="$(find "$target" -type f \
      -newermt "-${ACTIVE_WINDOW_SECONDS} seconds" -print -quit)"; then
    refuse "could not scan $target for recent writes; nothing was deleted."
  fi
  if [[ -n "$recent" ]]; then
    refuse "$target was written within ${ACTIVE_WINDOW_SECONDS}s ($recent);" \
      "it may still be live."
  fi
}

# Exactly logs/rsl_rl/<exp>/<run>: `case` globbing let `*` cross `/`, so a run's
# own subdirectory (a checkpoint dir, say) passed as a "run directory".
check_is_run_dir() {
  local target="$1"
  [[ -d "$target" ]] || refuse "$target is not a directory."
  if [[ ! "$target" =~ ^logs/rsl_rl/[^/]+/[^/]+$ ]]; then
    refuse "$target is not a run directory under logs/rsl_rl/<exp>/<run>."
  fi
}

# --before keeps the intent "drop runs older than T" but still requires an
# explicit experiment dir, so a typo cannot widen the blast radius to every
# task.
expand_before() {
  [[ -n "$BEFORE" ]] || return 0
  if [[ ${#TARGETS[@]} -ne 1 ]]; then
    refuse "--before takes exactly one experiment directory."
  fi
  local exp="${TARGETS[0]}"
  [[ -d "$exp" ]] || refuse "$exp is not a directory."
  local runs=()
  local run
  for run in "$exp"/*/; do
    # An unmatched glob leaves the pattern itself here, which is not a run.
    [[ -d "$run" ]] || continue
    run="${run%/}"
    [[ "$(basename "$run")" < "$BEFORE" ]] || continue
    runs+=("$run")
  done
  if [[ ${#runs[@]} -eq 0 ]]; then
    echo "nothing older than $BEFORE in $exp"
    exit 0
  fi
  TARGETS=("${runs[@]}")
}

delete_target() {
  local target="$1"
  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "would delete $target"
    return 0
  fi
  rm -rf -- "$target"
  echo "deleted $target"
}

main() {
  parse_args "$@"
  if [[ -z "$BEFORE" && ${#TARGETS[@]} -eq 0 ]]; then
    usage >&2
    exit 2
  fi
  validate_active_window
  check_no_glob
  check_training_not_running
  expand_before
  if [[ ${#TARGETS[@]} -eq 0 ]]; then
    echo "nothing to delete"
    return 0
  fi
  local target
  for target in "${TARGETS[@]}"; do
    check_is_run_dir "$target"
    check_not_recently_written "$target"
  done
  for target in "${TARGETS[@]}"; do
    delete_target "$target"
  done
}

main "$@"
