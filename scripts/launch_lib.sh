# The launch logic of scripts/run_pilot.sh (PLAN build order step 3) and scripts/run_main.sh (step 4).
# Each of them sets PHASE (pilot or main) and sources this file, which then does everything, so the two
# launchers share every flag and every safety check and cannot drift apart.
# The phases differ only in the table at the top: the tasks, the epochs per agent and variant, the log
# root, the prefix of the test-only override variables, and the cost line printed before the first part;
# besides, a real run of the full runs checks the spend so far against PLAN's caps in its preflight and,
# with the Inspect variables in .env, again before each later part.
# Each launcher's header documents what it does; this file is never run by itself.
#
# Only one launcher runs at a time across both phases, since they share Docker: a launcher takes its own
# lock directory (.run_<phase>.lock in its log root) and then refuses while the other phase's lock exists
# or while any eval-set of the task names a --log-dir under either phase's log root.
# Taking its own lock before looking for the other's means two launchers started together cannot both
# go on.
# Under a mock override (a test log root outside the repository) both locks and the eval-set check are in
# that log root alone, so a mock run never waits for, or blocks, a real one.

set -euo pipefail
set -f  # no globbing: AGENTS is split on whitespace only

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$ROOT"

ALL_AGENTS="haiku gpt-oss sonnet"
VARIANTS="normal conflict"
MOCK_MODEL="mockllm/model"
GPT_OSS_PROVIDER='provider={"order":["deepinfra/bf16"],"quantizations":["bf16"],"allow_fallbacks":false,"require_parameters":true}'
STOP_GRACE=120

# The phase table. EPOCHS_TABLE holds agent:variant:epochs entries.
case "${PHASE-}" in
  pilot)
    # PLAN build order step 3: seeds 078_0 and 327_0, Haiku x5, gpt-oss x5 and Sonnet x3 epochs (52 episodes).
    NAME=run_pilot.sh
    OTHER_PHASE=main
    SAMPLES="078_0,327_0"
    EPOCHS_TABLE="haiku:normal:5 haiku:conflict:5 gpt-oss:normal:5 gpt-oss:conflict:5 sonnet:normal:3 sonnet:conflict:3"
    ;;
  main)
    # PLAN's "Runs" table: all 8 tasks, Haiku and gpt-oss x12 normal and x24 conflict, Sonnet x5 and x5 (656 episodes).
    NAME=run_main.sh
    OTHER_PHASE=pilot
    SAMPLES="022_0,022_2,078_0,327_0,377_0,732_2,918_0,943_0"
    EPOCHS_TABLE="haiku:normal:12 haiku:conflict:24 gpt-oss:normal:12 gpt-oss:conflict:24 sonnet:normal:5 sonnet:conflict:5"
    ;;
  *)
    echo "launch_lib.sh: PHASE must be pilot or main, not '${PHASE-}'" >&2
    exit 2
    ;;
esac
OTHER_NAME="run_$OTHER_PHASE.sh"
# The override variables' prefix: PILOT or MAIN.
PREFIX="$(printf '%s' "$PHASE" | tr '[:lower:]' '[:upper:]')"
OTHER_PREFIX="$(printf '%s' "$OTHER_PHASE" | tr '[:lower:]' '[:upper:]')"
# Where an operator moves a part's log directory after a harness change (a mock run's goes anywhere out of
# its log root instead; MOVE_ASIDE, set with the log root, says which).
REJECTED="logs/$PHASE-rejected"
SAMPLE_COUNT=0
for sample in ${SAMPLES//,/ }; do
  SAMPLE_COUNT=$((SAMPLE_COUNT + 1))
done

die() {
  echo "$NAME: $*" >&2
  exit 2
}

# Sets MODEL and COST_LIMIT (per-sample dollars) for an agent, and EPOCHS for it and a variant.
agent_settings() {
  local entry
  case "$1" in
    haiku) MODEL=anthropic/claude-haiku-5-5 COST_LIMIT=0.10 ;;
    gpt-oss) MODEL=openrouter/openai/gpt-oss-120b COST_LIMIT=0.05 ;;
    sonnet) MODEL=anthropic/claude-sonnet-5-5 COST_LIMIT=1.00 ;;
    *) die "unknown agent '$1'" ;;
  esac
  EPOCHS=""
  for entry in $EPOCHS_TABLE; do
    case "$entry" in "$1:$2:"*) EPOCHS="${entry##*:}" ;; esac
  done
  [ -n "$EPOCHS" ] || die "no epochs for $1 $2"
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

# A variable meant for the other phase's launcher would be ignored here, so any name starting with its
# prefix (ignoring case) is refused; this comes first, so such a name is reported as the other phase's.
other="$(compgen -e | grep -i -e "^${OTHER_PREFIX}_" || true)"
[ -z "$other" ] || die "these variables look like settings of scripts/$OTHER_NAME, which this script never reads;" \
  "unset them first: $(echo $other)"
# A misspelled control variable would be ignored just as silently and leave a paid run (or, for AGENTS,
# run every part), so any PILOT_* (MAIN_* for the main runs), AGENT* or *DRY* name, and any name holding
# MODEL_OVERRIDE, EPOCHS_OVERRIDE or LOG_ROOT (a mistyped prefix, say), all ignoring case, that this
# script does not read is refused.
unread="$(
  compgen -e | grep -i -e "^${PREFIX}_" -e '^AGENT' -e 'DRY' -e MODEL_OVERRIDE -e EPOCHS_OVERRIDE -e LOG_ROOT |
    grep -v -x -e DRY_RUN -e AGENTS -e "${PREFIX}_MODEL_OVERRIDE" -e "${PREFIX}_EPOCHS_OVERRIDE" -e "${PREFIX}_LOG_ROOT" || true
)"
[ -z "$unread" ] || die "these variables look like settings of this script, which reads only DRY_RUN, AGENTS," \
  "${PREFIX}_MODEL_OVERRIDE, ${PREFIX}_EPOCHS_OVERRIDE and ${PREFIX}_LOG_ROOT; unset them first: $(echo $unread)"

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
# environment could add --fallback-models, --reasoning-tokens, a sampling setting or a sample limit; other
# INSPECT_* variables set the log format (INSPECT_LOG_FORMAT=json would write logs that scripts/main_status.py
# reports as bad input, which in a real run of the full runs stops the run only after its first part), sandbox
# output limits, HTTP retries and more. So every INSPECT_* variable but the two display settings is refused
# before anything runs.
stray="$(compgen -e | grep '^INSPECT_' | grep -v -x -e INSPECT_DISPLAY -e INSPECT_NO_ANSI || true)"
[ -z "$stray" ] || die "unset these Inspect option variables first: $(echo $stray)"
# PYTHONPATH or PYTHONHOME would change which modules (a sitecustomize.py, say) every uv command loads, while
# the logs record this repository's harness_sha256; PYTHONSTARTUP is refused with them.
stray="$(compgen -e | grep -x -e PYTHONPATH -e PYTHONHOME -e PYTHONSTARTUP || true)"
[ -z "$stray" ] || die "unset these Python variables first, which would change the code the runs load: $(echo $stray)"
# uv's own variables choose the project, working directory, environment, sync, lock and config file
# (UV_PROJECT, UV_WORKING_DIR, UV_PROJECT_ENVIRONMENT, UV_NO_SYNC, UV_FROZEN, UV_NO_EDITABLE, UV_CONFIG_FILE,
# UV_NO_PROJECT, UV_PYTHON and more), so one could make every uv command load another tree's task.py,
# costs.yaml or main_status.py, or another environment, under this repository's fingerprints. So every
# UV_* variable is refused but these, which change none of that: UV_RUN_RECURSION_DEPTH (uv run sets it
# for what it runs), UV_CACHE_DIR, UV_NO_PROGRESS, and UV_ENV_FILE and UV_NO_ENV_FILE (every uv command
# here passes --no-env-file).
stray="$(
  compgen -e | grep '^UV_' |
    grep -v -x -e UV_RUN_RECURSION_DEPTH -e UV_CACHE_DIR -e UV_NO_PROGRESS -e UV_ENV_FILE -e UV_NO_ENV_FILE || true
)"
[ -z "$stray" ] || die "unset these uv variables first, which would change the project, environment or settings every" \
  "uv command uses: $(echo $stray)"

# Any of the three overrides that is set, even to an empty value, means a mock run, so a mock run whose
# variables expand to nothing is refused instead of becoming the paid run.
MODEL_VAR="${PREFIX}_MODEL_OVERRIDE"
EPOCHS_VAR="${PREFIX}_EPOCHS_OVERRIDE"
ROOT_VAR="${PREFIX}_LOG_ROOT"
if [ -n "${!MODEL_VAR+set}${!EPOCHS_VAR+set}${!ROOT_VAR+set}" ]; then
  OVERRIDE="${!MODEL_VAR-}"
  EPOCHS_OVERRIDE="${!EPOCHS_VAR-}"
  LOG_ROOT="${!ROOT_VAR-}"
  [ "$OVERRIDE" = "$MOCK_MODEL" ] ||
    die "$MODEL_VAR must be $MOCK_MODEL whenever a ${PREFIX}_* override is set, not '$OVERRIDE'"
  [[ "$EPOCHS_OVERRIDE" =~ ^[1-9][0-9]*$ ]] ||
    die "$EPOCHS_VAR must be a positive integer with the mock override, not '$EPOCHS_OVERRIDE'"
  case "$LOG_ROOT" in
    /*) ;;
    *) die "$ROOT_VAR must be an absolute directory with the mock override, not '$LOG_ROOT'" ;;
  esac
  case "$LOG_ROOT/" in
    */./* | */../*) die "$ROOT_VAR must not contain . or .. components: '$LOG_ROOT'" ;;
  esac
  case "$(lower "$(resolve_path "$LOG_ROOT")")/" in
    "$(lower "$ROOT")/"*) die "$ROOT_VAR must be outside this repository ($ROOT): '$LOG_ROOT'" ;;
  esac
  MOCK=1
  LOG_PARENT="$LOG_ROOT"
  # A mock run's locks and eval-set check are in its own log root alone.
  OTHER_PARENT="$LOG_ROOT"
  MOVE_ASIDE="out of $LOG_ROOT"
else
  MOCK=0
  LOG_PARENT="$ROOT/logs/$PHASE"
  OTHER_PARENT="$ROOT/logs/$OTHER_PHASE"
  MOVE_ASIDE="into $REJECTED/"
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
  agent_settings "$agent" "$variant"
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

# Refuses because a launcher's lock directory exists: lock is the directory, root the log root it
# guards, script the launcher that takes it, and who "another" for this launcher's own lock or "a" for
# the other phase's. A lock left behind is never removed automatically, since two launchers could then
# both decide it is stale and take it.
refuse_held_lock() {
  local lock="$1" root="$2" script="$3" who="$4" holder orphan
  holder="$(cat "$lock/launcher" 2>/dev/null || true)"
  orphan="$(cat "$lock/eval-set" 2>/dev/null || true)"
  if [ -z "$holder" ]; then
    die "$lock exists without a launcher pid; after checking that no $script or eval-set is running on" \
      "$root, remove it and re-run"
  elif kill -0 "$holder" 2>/dev/null; then
    die "$who $script (pid $holder) is already running on $root; wait for it to finish, or ask whoever started it"
  elif [ -n "$orphan" ] && kill -0 "$orphan" 2>/dev/null; then
    die "the eval-set (pid $orphan) that an earlier $script (pid $holder) started is still running on" \
      "$root; wait for it to finish, or ask whoever started it"
  fi
  die "$lock is the stale lock of $script pid $holder, which is no longer running; after checking that no" \
    "$script or eval-set is running on $root, remove it and re-run"
}

# Refuses while any eval-set of the task names a --log-dir under root, such as one a killed launcher
# left behind. root is absolute, so an eval-set of another copy of this repository does not match.
refuse_running_eval_set() {
  local root="$1" pattern pgrep_status=0 running
  pattern="inspect eval-set src/rewardhack/task.py .*--log-dir $(regex_escape "${root%/}/")"
  running="$(pgrep -f "$pattern")" || pgrep_status=$?
  case "$pgrep_status" in
    0) die "an eval-set is already running on $root (pid $(echo $running)); wait for it to finish, or ask whoever started it" ;;
    1) ;;
    *) die "pgrep failed (exit $pgrep_status) while checking for a running eval-set" ;;
  esac
}

# The lock: mkdir is atomic, so only one launcher creates it; it records the launcher's pid and,
# while a part runs, the eval-set's pid.
mkdir -p "$LOG_PARENT"
LOCK="$(cd "$LOG_PARENT" && pwd -P)/.run_$PHASE.lock"
if ! mkdir "$LOCK" 2>/dev/null; then
  refuse_held_lock "$LOCK" "$LOG_PARENT" "$NAME" another
fi
echo "$$" >"$LOCK/launcher"
trap 'rm -rf "$LOCK"' EXIT

refuse_running_eval_set "$LOG_PARENT"
# The other phase: checked only now that this launcher holds its own lock, so of two launchers
# started together at least one sees the other's lock.
OTHER_LOCK="$(resolve_path "$OTHER_PARENT")/.run_$OTHER_PHASE.lock"
if [ -e "$OTHER_LOCK" ]; then
  refuse_held_lock "$OTHER_LOCK" "$OTHER_PARENT" "$OTHER_NAME" a
fi
if [ "$OTHER_PARENT" != "$LOG_PARENT" ]; then
  refuse_running_eval_set "$OTHER_PARENT"
fi

child=""
current=""
remaining=""
resume_hint() {
  echo "$NAME: to resume, re-invoke with AGENTS=\"$remaining\" and otherwise the same settings" \
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
    echo "$NAME: stopping $current on SIG$signal (eval-set pid $child)" >&2
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
      echo "$NAME: $current is still running after ${STOP_GRACE}s; terminating it" >&2
      kill -TERM "$child" 2>/dev/null || true
    fi
    wait "$child" 2>/dev/null || true
  fi
  echo "$NAME: stopped on SIG$signal before every selected part finished" >&2
  [ -z "$remaining" ] || resume_hint
  exit 130
}
trap 'stop INT' INT
trap 'stop TERM' TERM
trap 'stop HUP' HUP

# Each part's log directory, and each agent's samples times its cost limit in cents.
part_dirs=()
haiku_cents=0
gpt_oss_cents=0
sonnet_cents=0
for agent in $selected; do
  for variant in $VARIANTS; do
    build_command "$agent" "$variant"
    part_dirs+=("$LOG_DIR")
    cents=$((PART_SAMPLES * 10#${COST_LIMIT/./}))
    case "$agent" in
      haiku) haiku_cents=$((haiku_cents + cents)) ;;
      gpt-oss) gpt_oss_cents=$((gpt_oss_cents + cents)) ;;
      sonnet) sonnet_cents=$((sonnet_cents + cents)) ;;
    esac
  done
done

# A real run of the full runs checks each selected agent's spend so far against its cap in the preflight,
# and the next part's agent again before each later part (in the repository root, where
# scripts/main_status.py is); a mock run's model costs nothing, and the pilot launcher, whose pilot has
# run, has no such check.
spend_agents=""
if [ "$PHASE" = main ] && [ "$MOCK" = 0 ]; then
  for agent in $selected; do
    spend_agents="$spend_agents${spend_agents:+, }\"$agent\""
  done
fi

# The check of the Inspect variables, which the preflight and the check before each later part of a real
# run of the full runs make, under the same uv options as every eval-set. It needs LAUNCHER defined.
inspect_check_source() {
  cat <<'EOF'
import os
import sys

from dotenv import dotenv_values, find_dotenv

# Every INSPECT_* variable but the two display settings is refused, as in the launcher: INSPECT_EVAL_*
# sets an option the command leaves out, and others set the log format, sandbox limits and the like.
DISPLAY_ONLY = {"INSPECT_DISPLAY", "INSPECT_NO_ANSI"}
# This process runs under the same uv options as every eval-set, so it sees the environment they get.
stray = sorted(key for key in os.environ if key.startswith("INSPECT_") and key not in DISPLAY_ONLY)
if stray:
    sys.exit(f"{LAUNCHER}: unset these Inspect option variables first: {' '.join(stray)}")
# Each eval-set loads the .env file that find_dotenv(usecwd=True) finds, with python-dotenv's
# dotenv_values and load_dotenv, before it parses options, so an INSPECT_* key there would act like one
# in the environment. The file is read with that same parser (which strips a leading byte order mark,
# for one), and only the names of the keys it would set, those with a value, are kept; no value is used
# or printed.
dotenv_file = find_dotenv(usecwd=True)
stray = sorted(
    key for key, value in (dotenv_values(dotenv_file) if dotenv_file else {}).items()
    if key.startswith("INSPECT_") and key not in DISPLAY_ONLY and value is not None
)
if stray:
    sys.exit(f"{LAUNCHER}: remove these Inspect option variables from {dotenv_file} first: {' '.join(stray)}")
EOF
}

# The preflight, which runs under the same uv options as every eval-set. Its first lines name the
# launcher, the directory for rejected logs, whether this is a mock run and the agents whose spend it
# checks; the rest is the same for both phases.
preflight_source() {
  printf 'LAUNCHER = "%s"\nREJECTED = "%s"\nMOCK = %s\nSPEND_AGENTS = [%s]\n' "$NAME" "$REJECTED" "$MOCK" "$spend_agents"
  inspect_check_source
  cat <<'EOF'
from pathlib import Path

from inspect_ai.log import list_eval_logs, read_eval_log

from rewardhack.containers import docker_problem, task_dirs
from rewardhack.task import recorded_image_digest

samples, harness, parts = sys.argv[1], sys.argv[2], sys.argv[3:]
# Where a stale part's logs go: a mock run's anywhere out of its log root, not into the repository.
move_aside = f"out of {Path(parts[0]).parent}" if MOCK else f"into {REJECTED}/"
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
        f"{LAUNCHER}: the harness is now {harness}, but these log directories hold logs of another harness, "
        f"whose samples eval-set would reuse: {'; '.join(stale)}; move each one {move_aside} and re-run"
    )
# The spend so far and the most the selected parts can add, against each agent's cap, as scripts/main_status.py
# counts them; the parts are its six directories under one log root.
if SPEND_AGENTS:
    sys.path.insert(0, str(Path("scripts").resolve()))
    import main_status

    refusal = main_status.launch_refusal(["--logs", str(Path(parts[0]).parent)], SPEND_AGENTS)
    if refusal:
        sys.exit(f"{LAUNCHER}: {refusal}")
problem = docker_problem()
if problem is not None:
    sys.exit(f"{LAUNCHER}: Docker is unavailable: {problem}")
try:
    digest = recorded_image_digest()
except RuntimeError as e:
    sys.exit(f"{LAUNCHER}: {e}")
# Every selected task must be a task directory that the task loads, that is one with a task.json.
missing = sorted(set(samples.split(",")) - {d.name for d in task_dirs()})
if missing:
    sys.exit(f"{LAUNCHER}: no task directory for {missing}")
print(f"{LAUNCHER}: Docker answers and the sandbox image is {digest}", file=sys.stderr)
EOF
}
uv run --no-env-file python - "$SAMPLES" "$HARNESS_SHA256" "${part_dirs[@]}" <<<"$(preflight_source)" ||
  die "preflight failed; nothing was run"

# The check before each later part of a real run of the full runs: the preflight's Inspect variable check,
# since each part is a new eval-set that loads .env again, and its spend check for one agent, whose name
# follows the log root, since a part can spend more than its samples times its limit. (The pilot, which
# has run, and the free mock runs have no check between parts.)
spend_check_source() {
  printf 'LAUNCHER = "%s"\n' "$NAME"
  inspect_check_source
  cat <<'EOF'
from pathlib import Path

sys.path.insert(0, str(Path("scripts").resolve()))
import main_status

refusal = main_status.launch_refusal(["--logs", sys.argv[1]], sys.argv[2:])
if refusal:
    sys.exit(f"{LAUNCHER}: {refusal}")
EOF
}

if [ "$MOCK" = 1 ]; then
  echo "$NAME: MOCK run: every part uses $MOCK_MODEL, $EPOCHS_OVERRIDE epoch(s), logs under $LOG_ROOT" >&2
fi

# Joins its arguments as "a", "a and b" or "a, b and c".
join_and() {
  local out="" i=1 item
  for item in "$@"; do
    if [ "$i" = 1 ]; then
      out="$item"
    elif [ "$i" = "$#" ]; then
      out="$out and $item"
    else
      out="$out, $item"
    fi
    i=$((i + 1))
  done
  printf '%s' "$out"
}

if [ "$PHASE" = pilot ]; then
  echo "$NAME: the selected parts' samples times their cost limits come to $(dollars $((haiku_cents + sonnet_cents))) on" \
    "Anthropic (PLAN's hard cap for the pilot's Claude part is \$14.00) and $(dollars "$gpt_oss_cents") on" \
    "OpenRouter (counted against the \$15.50 gpt-oss cap), which a generation in flight and retried errors can exceed" >&2
else
  amounts=()
  for agent in $selected; do
    case "$agent" in
      haiku) amounts+=("$(dollars "$haiku_cents") for Haiku (PLAN's hard cap is \$28.80)") ;;
      gpt-oss) amounts+=("$(dollars "$gpt_oss_cents") for gpt-oss on OpenRouter (counted against the \$15.50 gpt-oss cap, which the pilot and the smoke test share)") ;;
      sonnet) amounts+=("$(dollars "$sonnet_cents") for Sonnet (PLAN's hard cap is \$80.00)") ;;
    esac
  done
  echo "$NAME: the selected parts' samples times their cost limits come to $(join_and "${amounts[@]}")," \
    "which a generation in flight and retried errors can exceed; scripts/main_status.py reports the spend so far" \
    "against these caps" >&2
fi
# A real run of the full runs checks the agent's spend so far, and .env, again before each part after the
# first, whose agents the preflight has just checked, since an earlier part can take its agent past its cap
# and each part loads .env anew.
spend_checked=1
for agent in $selected; do
  remaining="$agent${selected#* "$agent"}"
  for variant in $VARIANTS; do
    build_command "$agent" "$variant"
    current="$agent $variant"
    if [ -n "$spend_agents" ] && [ "$spend_checked" = 0 ]; then
      uv run --no-env-file python - "$LOG_PARENT" "$agent" <<<"$(spend_check_source)" || {
        echo "$NAME: the check before $current failed, so it was not started" >&2
        echo "$NAME: an agent refused for its cap stays out of AGENTS until Jessup approves more spend, and" \
          "uv run python scripts/main_status.py names the agents whose spend so far lets the launcher start them, or the" \
          "bad input that stops every part" >&2
        resume_hint
        exit 2
      }
    fi
    # The part would load the harness from disk and record the launch-time fingerprints.
    now="$(fingerprints)" || now="(could not hash)"
    if [ "$now" != "$LAUNCH_FINGERPRINTS" ]; then
      echo "$NAME: costs.yaml or a harness file ($HARNESS_PATHS) changed since launch, so $current" \
        "was not started; it would record harness_sha256 $HARNESS_SHA256 while running other code" >&2
      resume_hint
      exit 2
    fi
    echo "$NAME: starting $current ($PART_SAMPLES samples, cost limit \$$COST_LIMIT each):" >&2
    print_command "${CMD[@]}" >&2
    # A background job of a non-interactive shell starts with SIGINT ignored, and Python then keeps
    # ignoring it; perl restores the default before it runs the command.
    perl -e '$SIG{INT} = $SIG{QUIT} = "DEFAULT"; exec { $ARGV[0] } @ARGV or die "exec $ARGV[0]: $!\n"' \
      "${CMD[@]}" <&0 &
    child=$!
    spend_checked=0
    echo "$child" >"$LOCK/eval-set" || true
    status=0
    wait "$child" || status=$?
    child=""
    rm -f "$LOCK/eval-set"
    now="$(fingerprints)" || now="(could not hash)"
    if [ "$now" != "$LAUNCH_FINGERPRINTS" ]; then
      echo "$NAME: costs.yaml or a harness file ($HARNESS_PATHS) changed while $current ran (exit" \
        "$status), so some of its samples may have run other code than its log's harness_sha256" \
        "$HARNESS_SHA256; move $LOG_DIR $MOVE_ASIDE before resuming" >&2
      resume_hint
      exit 1
    fi
    if [ "$status" != 0 ]; then
      echo "$NAME: $current did not finish (exit $status)" >&2
      resume_hint
      exit 1
    fi
  done
done
echo "$NAME: every selected part finished" >&2
