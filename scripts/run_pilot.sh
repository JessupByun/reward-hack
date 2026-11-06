#!/usr/bin/env bash
# Pilot launcher for docs/PLAN.md build order step 3: seeds 078_0 and 327_0 in both variants, with
# Haiku x5, gpt-oss x5 and Sonnet x3 epochs (52 episodes), as six inspect eval-set runs one after
# another (Haiku normal, Haiku conflict, gpt-oss normal, gpt-oss conflict, Sonnet normal, Sonnet
# conflict), each with its own log directory logs/pilot/<agent>-<variant> (passed to eval-set as an
# absolute path under this repository) and PLAN's flags.
#
# Usage, from any directory:
#   scripts/run_pilot.sh                   run every part (paid API calls)
#   DRY_RUN=1 scripts/run_pilot.sh         print the commands, one per line, and exit; nothing runs
#   AGENTS="gpt-oss" scripts/run_pilot.sh  run only the named agents' parts (haiku, gpt-oss, sonnet)
#
# Resuming: re-invoke the same command, with AGENTS to skip the parts that finished.
# eval-set keeps each log directory's eval-set id, returns a task whose log already holds every
# selected sample and epoch without running it, and re-runs only the missing or errored samples of
# an incomplete log; a log directory holding a log from a different command (another model, task
# argument, generate config or limit) makes eval-set refuse to start.
# --sample-id here only selects the two pilot seeds, the same on every invocation; it is never a
# recovery list, and recovery never uses inspect's retry command, which drops --cost-limit.
#
# Re-piloting after a fix (PLAN: "On a no-go, fix and re-pilot the failing part"): eval-set's task
# identity covers no code, so after a harness change it would reuse a part's old samples.
# Every run therefore records harness_sha256 as eval metadata, a sha256 over every file under
# HARNESS_PATHS (bytecode caches, dotfiles, src/rewardhack/monitors and src/rewardhack/analysis left
# out; those two read finished logs, and no module an episode loads imports them), and before running
# anything the launcher refuses while a selected part's log directory holds a log with a different or
# missing one.
# Move each directory it names out of the log root (for example from logs/pilot to
# logs/pilot-rejected/), where eval-set no longer reuses its samples, and re-run.
# scripts/pilot_report.py still reads logs/pilot-rejected by default (--earlier-logs), for spend only:
# it counts those logs' spend (the larger of each log's header and its samples) against the pilot caps,
# treating killed or replaced logs as lower bounds, and none of their episodes.
# Each part is a new process that loads the harness from disk, so the launcher hashes costs.yaml and
# the harness again before and after each part: it refuses to start a part when either changed since
# launch, and fails when either changed while a part ran, naming that part's log directory to move aside.
#
# One launcher at a time: a run holds the lock directory .run_pilot.lock in the log root (logs/pilot,
# or PILOT_LOG_ROOT) and refuses to start while that lock exists, or while any eval-set of the task
# names a --log-dir under the same log root, written as this script writes it (an absolute path, so
# eval-sets of other copies of this repository do not count).
# Stopping the launcher (SIGINT, SIGTERM or SIGHUP) stops the running part: it interrupts the
# eval-set, which cancels its samples and cleans up its sandboxes, and terminates it if it is still
# running after STOP_GRACE seconds.
# A launcher killed with SIGKILL cannot do that or remove its lock: the next one names the orphaned
# eval-set while it runs, and afterwards refuses until an operator who has checked that no pilot is
# running removes the stale lock by hand.
#
# Every command passes --log-model-api, because by default Inspect keeps the raw call of only the
# first 5 model events per sample, and the pilot report checks every gpt-oss response's provider.
# The per-sample cost limits are checked after each generation, and retried errors run with fresh
# limits, so a part can spend more than its samples times its limit (PLAN's Budget allows for this).
#
# Test-only mock override (no paid calls); set all three or none:
#   PILOT_MODEL_OVERRIDE=mockllm/model  every part runs mockllm/model, without gpt-oss's -M routing
#   PILOT_EPOCHS_OVERRIDE=<n>           epochs for every part
#   PILOT_LOG_ROOT=<absolute dir>       log directories become <dir>/<agent>-<variant>
# Setting only some of them (even to an empty value), any other model, or a log root inside this
# repository (compared after resolving symlinks, ignoring case) is refused.
# So is any other variable whose name starts with PILOT_ or contains DRY (both ignoring case) besides
# DRY_RUN, such as a misspelled PILOT_MODEL or DRYRUN that would otherwise leave a paid run, and any
# variable whose name starts with AGENT (ignoring case) other than exactly AGENTS, such as AGENT or
# agents, which would otherwise be ignored and run every part. AGENT counts only at the start of a
# name, because Claude Code sets AI_AGENT and CLAUDE_AGENT_SDK_* and ssh-agent sets SSH_AGENT_PID.
#
# Before running anything the script refuses to start unless Docker answers (within the bounded
# timeouts of rewardhack.containers.docker_problem and rewardhack.task.recorded_image_digest), the
# local sandbox image is the one sandbox/image_digest.txt records, both seeds exist under tasks/,
# no INSPECT_EVAL_* option is set in the environment uv gives the runs or in the .env file Inspect
# would load, no selected part's logs come from another harness, and, for a real run, a .env file
# exists.
# The .env check scans the file line by line and keeps only the key names that match
# ^\s*(export\s+)?'?(INSPECT_EVAL_\w*)'?\s*= (python-dotenv also accepts a single-quoted key); it
# never keeps, uses or prints a value.
# Every uv command passes --no-env-file, so neither UV_ENV_FILE nor uv's --env-file can add options.
# Every run also records costs.yaml's path and sha256 as eval metadata, because the log header has
# no field for --model-cost-config.

set -euo pipefail
set -f  # no globbing: AGENTS is split on whitespace only

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$ROOT"

SAMPLES="078_0,327_0"
SAMPLE_COUNT=2
ALL_AGENTS="haiku gpt-oss sonnet"
VARIANTS="normal conflict"
MOCK_MODEL="mockllm/model"
GPT_OSS_PROVIDER='provider={"order":["deepinfra/bf16"],"quantizations":["bf16"],"allow_fallbacks":false,"require_parameters":true}'
STOP_GRACE=120

die() {
  echo "run_pilot.sh: $*" >&2
  exit 2
}

# Sets MODEL, EPOCHS and COST_LIMIT (per-sample dollars) for an agent.
agent_settings() {
  case "$1" in
    haiku) MODEL=anthropic/claude-haiku-5-5 EPOCHS=5 COST_LIMIT=0.10 ;;
    gpt-oss) MODEL=openrouter/openai/gpt-oss-120b EPOCHS=5 COST_LIMIT=0.05 ;;
    sonnet) MODEL=anthropic/claude-sonnet-5-5 EPOCHS=3 COST_LIMIT=1.00 ;;
    *) die "unknown agent '$1'" ;;
  esac
}

# Prints a command on one line, single-quoting any argument a shell would split or expand.
print_command() {
  local line="" arg
  for arg in "$@"; do
    case "$arg" in
      *\'*) arg="$(printf '%q' "$arg")" ;;
      "" | *[!A-Za-z0-9_./:=,@%+-]*) arg="'$arg'" ;;
    esac
    line="$line${line:+ }$arg"
  done
  printf '%s\n' "$line"
}

# Prints an absolute path with its longest existing ancestor resolved by pwd -P (symlinks followed),
# so that a path that does not exist yet can still be compared with ROOT.
resolve_path() {
  local dir="$1" rest=""
  while [ ! -d "$dir" ]; do
    rest="/${dir##*/}$rest"
    dir="${dir%/*}"
    [ -n "$dir" ] || dir=/
  done
  dir="$(cd "$dir" && pwd -P)"
  printf '%s%s\n' "${dir%/}" "$rest"
}

lower() {
  printf '%s' "$1" | tr '[:upper:]' '[:lower:]'
}

# Escapes the extended regular expression characters in a string, for pgrep.
regex_escape() {
  printf '%s' "$1" | sed 's/[][\.*^$+?(){}|]/\\&/g'
}

# Dollars for a whole number of cents.
dollars() {
  printf '$%d.%02d' "$(($1 / 100))" "$(($1 % 100))"
}

# A misspelled control variable would be ignored and leave a paid run (or, for AGENTS, run every
# part), so any PILOT_*, AGENT* or *DRY* name (ignoring case) that this script does not read is refused.
unread="$(
  compgen -e | grep -i -e '^PILOT_' -e '^AGENT' -e 'DRY' |
    grep -v -x -e DRY_RUN -e AGENTS -e PILOT_MODEL_OVERRIDE -e PILOT_EPOCHS_OVERRIDE -e PILOT_LOG_ROOT || true
)"
[ -z "$unread" ] || die "these variables look like settings of this script, which reads only DRY_RUN, AGENTS," \
  "PILOT_MODEL_OVERRIDE, PILOT_EPOCHS_OVERRIDE and PILOT_LOG_ROOT; unset them first: $(echo $unread)"

if [ -z "${DRY_RUN+set}" ]; then
  DRY=0
else
  case "$DRY_RUN" in
    0) DRY=0 ;;
    1) DRY=1 ;;
    *) die "DRY_RUN must be 1, 0 or unset, not '${DRY_RUN}'" ;;
  esac
fi

# Inspect reads an INSPECT_EVAL_* variable for any option the command leaves out, so one left in the
# environment could add --fallback-models, --reasoning-tokens, a sampling setting or a sample limit.
stray="$(compgen -e | grep '^INSPECT_EVAL_' || true)"
[ -z "$stray" ] || die "unset these Inspect option variables first: $(echo $stray)"

# Any of the three PILOT_* overrides that is set, even to an empty value, means a mock run, so a mock
# run whose variables expand to nothing is refused instead of becoming the paid pilot.
if [ -n "${PILOT_MODEL_OVERRIDE+set}${PILOT_EPOCHS_OVERRIDE+set}${PILOT_LOG_ROOT+set}" ]; then
  OVERRIDE="${PILOT_MODEL_OVERRIDE-}"
  EPOCHS_OVERRIDE="${PILOT_EPOCHS_OVERRIDE-}"
  LOG_ROOT="${PILOT_LOG_ROOT-}"
  [ "$OVERRIDE" = "$MOCK_MODEL" ] ||
    die "PILOT_MODEL_OVERRIDE must be $MOCK_MODEL whenever a PILOT_* override is set, not '$OVERRIDE'"
  [[ "$EPOCHS_OVERRIDE" =~ ^[1-9][0-9]*$ ]] ||
    die "PILOT_EPOCHS_OVERRIDE must be a positive integer with the mock override, not '$EPOCHS_OVERRIDE'"
  case "$LOG_ROOT" in
    /*) ;;
    *) die "PILOT_LOG_ROOT must be an absolute directory with the mock override, not '$LOG_ROOT'" ;;
  esac
  case "$LOG_ROOT/" in
    */./* | */../*) die "PILOT_LOG_ROOT must not contain . or .. components: '$LOG_ROOT'" ;;
  esac
  case "$(lower "$(resolve_path "$LOG_ROOT")")/" in
    "$(lower "$ROOT")/"*) die "PILOT_LOG_ROOT must be outside this repository ($ROOT): '$LOG_ROOT'" ;;
  esac
  MOCK=1
  LOG_PARENT="$LOG_ROOT"
else
  MOCK=0
  LOG_PARENT="$ROOT/logs/pilot"
fi

requested=" "
for agent in ${AGENTS-$ALL_AGENTS}; do
  case " $ALL_AGENTS " in
    *" $agent "*) requested="$requested$agent " ;;
    *) die "unknown agent '$agent' in AGENTS (choose from: $ALL_AGENTS)" ;;
  esac
done
selected=""
for agent in $ALL_AGENTS; do
  case "$requested" in *" $agent "*) selected="$selected $agent" ;; esac
done
[ -n "$selected" ] || die "AGENTS selects no agent (choose from: $ALL_AGENTS)"

[ -f costs.yaml ] || die "costs.yaml is missing from $ROOT"
if command -v sha256sum >/dev/null 2>&1; then
  SHA256=(sha256sum)
else
  SHA256=(shasum -a 256)
fi

# The files an episode depends on; harness_sha256 hashes the sorted "sha256  path" line of each.
# The monitor and analysis packages read finished logs, and no module an episode loads imports them,
# so they are left out and work on them does not block resuming a part.
HARNESS_PATHS="costs.yaml uv.lock sandbox src/rewardhack tasks/${SAMPLES//,/ tasks/}"
for path in $HARNESS_PATHS; do
  [ -e "$path" ] || die "$path is missing from $ROOT"
done

# Prints costs.yaml's sha256 and harness_sha256, separated by a space.
fingerprints() {
  local costs harness
  costs="$("${SHA256[@]}" costs.yaml | cut -d ' ' -f 1)" || return 1
  harness="$(
    find $HARNESS_PATHS -type f ! -path '*/__pycache__/*' ! -path '*/.*' \
      ! -path 'src/rewardhack/monitors/*' ! -path 'src/rewardhack/analysis/*' -print0 | LC_ALL=C sort -z |
      xargs -0 "${SHA256[@]}" | "${SHA256[@]}" | cut -d ' ' -f 1
  )" || return 1
  printf '%s %s\n' "$costs" "$harness"
}
LAUNCH_FINGERPRINTS="$(fingerprints)" || die "could not hash costs.yaml and the harness files ($HARNESS_PATHS)"
COSTS_SHA256="${LAUNCH_FINGERPRINTS% *}"
HARNESS_SHA256="${LAUNCH_FINGERPRINTS#* }"

# Sets CMD to the eval-set command for an agent and variant, LOG_DIR to its log directory, and
# PART_SAMPLES to its sample count.
build_command() {
  local agent="$1" variant="$2" log_dir="$LOG_PARENT/$1-$2" epochs
  agent_settings "$agent"
  CMD=(uv run --no-env-file inspect eval-set src/rewardhack/task.py -T "variant=$variant" --sample-id "$SAMPLES")
  if [ "$MOCK" = 1 ]; then
    epochs="$EPOCHS_OVERRIDE"
    CMD+=(--epochs "$epochs" --model "$MOCK_MODEL" --reasoning-effort medium)
  else
    epochs="$EPOCHS"
    CMD+=(--epochs "$epochs" --model "$MODEL" --reasoning-effort medium)
    if [ "$agent" = gpt-oss ]; then
      CMD+=(-M "$GPT_OSS_PROVIDER")
    fi
  fi
  CMD+=(
    --turn-limit 30 --token-limit output:100k --time-limit 900 --cost-limit "$COST_LIMIT"
    --max-sandboxes 12 --max-connections 12 --retry-on-error 2 --fail-on-error 0.05
    --retry-attempts 2 --model-cost-config costs.yaml --log-model-api
    --metadata model_cost_config=costs.yaml --metadata "model_cost_sha256=$COSTS_SHA256"
    --metadata "harness_sha256=$HARNESS_SHA256" --log-dir "$log_dir"
  )
  LOG_DIR="$log_dir"
  PART_SAMPLES=$((SAMPLE_COUNT * epochs))
}

if [ "$DRY" = 1 ]; then
  for agent in $selected; do
    for variant in $VARIANTS; do
      build_command "$agent" "$variant"
      print_command "${CMD[@]}"
    done
  done
  exit 0
fi

if [ "$MOCK" = 0 ] && [ ! -f .env ]; then
  die "no .env file in $ROOT; it must hold ANTHROPIC_API_KEY and OPENROUTER_API_KEY"
fi
command -v perl >/dev/null 2>&1 || die "perl is needed to start each part with SIGINT not ignored"
command -v pgrep >/dev/null 2>&1 || die "pgrep is needed to check for an eval-set already running"

# The lock: mkdir is atomic, so only one launcher creates it; it records the launcher's pid and,
# while a part runs, the eval-set's pid. A lock left behind is never removed automatically, since two
# launchers could then both decide it is stale and take it.
mkdir -p "$LOG_PARENT"
LOCK="$(cd "$LOG_PARENT" && pwd -P)/.run_pilot.lock"
if ! mkdir "$LOCK" 2>/dev/null; then
  holder="$(cat "$LOCK/launcher" 2>/dev/null || true)"
  orphan="$(cat "$LOCK/eval-set" 2>/dev/null || true)"
  if [ -z "$holder" ]; then
    die "$LOCK exists without a launcher pid; after checking that no run_pilot.sh or eval-set is running on" \
      "$LOG_PARENT, remove it and re-run"
  elif kill -0 "$holder" 2>/dev/null; then
    die "another run_pilot.sh (pid $holder) is already running on $LOG_PARENT; wait for it to finish, or ask whoever started it"
  elif [ -n "$orphan" ] && kill -0 "$orphan" 2>/dev/null; then
    die "the eval-set (pid $orphan) that an earlier run_pilot.sh (pid $holder) started is still running on" \
      "$LOG_PARENT; wait for it to finish, or ask whoever started it"
  fi
  die "$LOCK is the stale lock of run_pilot.sh pid $holder, which is no longer running; after checking that no" \
    "run_pilot.sh or eval-set is running on $LOG_PARENT, remove it and re-run"
fi
echo "$$" >"$LOCK/launcher"
trap 'rm -rf "$LOCK"' EXIT

# Any eval-set writing under this log root, such as one a killed launcher left behind. LOG_PARENT is
# absolute, so an eval-set of another copy of this repository does not match.
pattern="inspect eval-set src/rewardhack/task.py .*--log-dir $(regex_escape "${LOG_PARENT%/}/")"
pgrep_status=0
running="$(pgrep -f "$pattern")" || pgrep_status=$?
case "$pgrep_status" in
  0) die "an eval-set is already running on $LOG_PARENT (pid $(echo $running)); wait for it to finish, or ask whoever started it" ;;
  1) ;;
  *) die "pgrep failed (exit $pgrep_status) while checking for a running eval-set" ;;
esac

child=""
current=""
remaining=""
resume_hint() {
  echo "run_pilot.sh: to resume, re-invoke with AGENTS=\"$remaining\" and otherwise the same settings" \
    "(finished samples are reused only while the harness is unchanged; after a fix, the launcher names" \
    "the log directories to move aside)" >&2
}
# Stops the running part. A part runs in the background so that this trap runs at once, and it
# starts with SIGINT not ignored, so the interrupt reaches Inspect (through uv, which forwards it)
# and Inspect cancels its samples and cleans up its sandboxes. A SIGINT from the terminal already
# reached the part, so it gets 30 seconds before the launcher interrupts it too.
stop() {
  local signal="$1" waited=0
  trap '' INT TERM HUP
  child="${child:-$(jobs -p)}"
  if [ -n "$child" ]; then
    echo "run_pilot.sh: stopping $current on SIG$signal (eval-set pid $child)" >&2
    if [ "$signal" = INT ]; then
      while kill -0 "$child" 2>/dev/null && [ "$waited" -lt 30 ]; do
        sleep 1
        waited=$((waited + 1))
      done
    fi
    kill -INT "$child" 2>/dev/null || true
    waited=0
    while kill -0 "$child" 2>/dev/null && [ "$waited" -lt "$STOP_GRACE" ]; do
      sleep 1
      waited=$((waited + 1))
    done
    if kill -0 "$child" 2>/dev/null; then
      echo "run_pilot.sh: $current is still running after ${STOP_GRACE}s; terminating it" >&2
      kill -TERM "$child" 2>/dev/null || true
    fi
    wait "$child" 2>/dev/null || true
  fi
  echo "run_pilot.sh: stopped on SIG$signal before every selected part finished" >&2
  [ -z "$remaining" ] || resume_hint
  exit 130
}
trap 'stop INT' INT
trap 'stop TERM' TERM
trap 'stop HUP' HUP

part_dirs=()
claude_cents=0
openrouter_cents=0
for agent in $selected; do
  for variant in $VARIANTS; do
    build_command "$agent" "$variant"
    part_dirs+=("$LOG_DIR")
    cents=$((PART_SAMPLES * 10#${COST_LIMIT/./}))
    if [ "$agent" = gpt-oss ]; then
      openrouter_cents=$((openrouter_cents + cents))
    else
      claude_cents=$((claude_cents + cents))
    fi
  done
done

uv run --no-env-file python - "$SAMPLES" "$HARNESS_SHA256" "${part_dirs[@]}" <<'EOF' || die "preflight failed; nothing was run"
import os
import re
import sys
from pathlib import Path

from dotenv import find_dotenv
from inspect_ai.log import list_eval_logs, read_eval_log

from rewardhack.containers import docker_problem, task_dirs
from rewardhack.task import recorded_image_digest

samples, harness, parts = sys.argv[1], sys.argv[2], sys.argv[3:]
# This process runs under the same uv options as every eval-set, so it sees the environment they get.
stray = sorted(key for key in os.environ if key.startswith("INSPECT_EVAL_"))
if stray:
    sys.exit(f"run_pilot.sh: unset these Inspect option variables first: {' '.join(stray)}")
# Inspect loads the .env file that find_dotenv(usecwd=True) finds before it parses options, so an
# INSPECT_EVAL_* key there would set an option the command leaves out. The file is scanned line by
# line and only matching key names are kept (python-dotenv also accepts a single-quoted key); no value
# is kept, used or printed.
dotenv_file = find_dotenv(usecwd=True)
key = re.compile(r"\s*(?:export\s+)?'?(INSPECT_EVAL_\w*)'?\s*=")
stray = []
if dotenv_file:
    with open(dotenv_file, encoding="utf-8", errors="replace") as f:
        stray = sorted({match.group(1) for line in f if (match := key.match(line))})
if stray:
    sys.exit(f"run_pilot.sh: remove these Inspect option variables from {dotenv_file} first: {' '.join(stray)}")
# eval-set lists a log directory's logs recursively and would reuse their samples after a harness change.
stale = []
for part in parts:
    if Path(part).is_dir():
        headers = [read_eval_log(info, header_only=True) for info in list_eval_logs(part)]
        found = sorted({str((h.eval.metadata or {}).get("harness_sha256")) for h in headers})
        if found and found != [harness]:
            stale.append(f"{part} (harness_sha256 {', '.join(found)})")
if stale:
    sys.exit(
        f"run_pilot.sh: the harness is now {harness}, but these log directories hold logs of another harness, "
        f"whose samples eval-set would reuse: {'; '.join(stale)}; move each one out of the log root (for example "
        "into logs/pilot-rejected/) and re-run"
    )
problem = docker_problem()
if problem is not None:
    sys.exit(f"run_pilot.sh: Docker is unavailable: {problem}")
try:
    digest = recorded_image_digest()
except RuntimeError as e:
    sys.exit(f"run_pilot.sh: {e}")
missing = sorted(set(samples.split(",")) - {d.name for d in task_dirs()})
if missing:
    sys.exit(f"run_pilot.sh: no task directory for {missing}")
print(f"run_pilot.sh: Docker answers and the sandbox image is {digest}", file=sys.stderr)
EOF

if [ "$MOCK" = 1 ]; then
  echo "run_pilot.sh: MOCK run: every part uses $MOCK_MODEL, $EPOCHS_OVERRIDE epoch(s), logs under $LOG_ROOT" >&2
fi
echo "run_pilot.sh: the selected parts' samples times their cost limits come to $(dollars "$claude_cents") on" \
  "Anthropic (PLAN's hard cap for the pilot's Claude part is \$14.00) and $(dollars "$openrouter_cents") on" \
  "OpenRouter (counted against the \$15.50 gpt-oss cap), which a generation in flight and retried errors can exceed" >&2
for agent in $selected; do
  remaining="$agent${selected#* "$agent"}"
  for variant in $VARIANTS; do
    build_command "$agent" "$variant"
    current="$agent $variant"
    # The part would load the harness from disk and record the launch-time fingerprints.
    now="$(fingerprints)" || now="(could not hash)"
    if [ "$now" != "$LAUNCH_FINGERPRINTS" ]; then
      echo "run_pilot.sh: costs.yaml or a harness file ($HARNESS_PATHS) changed since launch, so $current" \
        "was not started; it would record harness_sha256 $HARNESS_SHA256 while running other code" >&2
      resume_hint
      exit 2
    fi
    echo "run_pilot.sh: starting $current ($PART_SAMPLES samples, cost limit \$$COST_LIMIT each):" >&2
    print_command "${CMD[@]}" >&2
    # A background job of a non-interactive shell starts with SIGINT ignored, and Python then keeps
    # ignoring it; perl restores the default before it runs the command.
    perl -e '$SIG{INT} = $SIG{QUIT} = "DEFAULT"; exec { $ARGV[0] } @ARGV or die "exec $ARGV[0]: $!\n"' \
      "${CMD[@]}" <&0 &
    child=$!
    echo "$child" >"$LOCK/eval-set" || true
    status=0
    wait "$child" || status=$?
    child=""
    rm -f "$LOCK/eval-set"
    now="$(fingerprints)" || now="(could not hash)"
    if [ "$now" != "$LAUNCH_FINGERPRINTS" ]; then
      echo "run_pilot.sh: costs.yaml or a harness file ($HARNESS_PATHS) changed while $current ran (exit" \
        "$status), so some of its samples may have run other code than its log's harness_sha256" \
        "$HARNESS_SHA256; move $LOG_DIR out of the log root (for example into logs/pilot-rejected/) before" \
        "resuming" >&2
      resume_hint
      exit 1
    fi
    if [ "$status" != 0 ]; then
      echo "run_pilot.sh: $current did not finish (exit $status)" >&2
      resume_hint
      exit 1
    fi
  done
done
echo "run_pilot.sh: every selected part finished" >&2
