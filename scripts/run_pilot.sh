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
# Move each directory it names from logs/pilot into logs/pilot-rejected/ (at any depth), where eval-set
# no longer reuses its samples, and re-run.
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
# So is any variable whose name holds MODEL_OVERRIDE, EPOCHS_OVERRIDE or LOG_ROOT (ignoring case) other than
# the three above, such as PILTO_MODEL_OVERRIDE or a bare LOG_ROOT, which would otherwise leave the paid pilot.
# So are PYTHONPATH, PYTHONHOME and PYTHONSTARTUP, which could load other code (a sitecustomize.py, say)
# into every run while its log records this repository's harness_sha256, and for the same reason every
# UV_* variable but UV_RUN_RECURSION_DEPTH (which uv run sets), UV_CACHE_DIR, UV_NO_PROGRESS, UV_ENV_FILE
# and UV_NO_ENV_FILE: UV_WORKING_DIR, UV_PROJECT, UV_PROJECT_ENVIRONMENT, UV_NO_SYNC, UV_FROZEN,
# UV_NO_EDITABLE, UV_CONFIG_FILE and the like would make every uv command use another tree's code or
# costs.yaml, or another environment.
#
# Before running anything the script refuses to start unless Docker answers (within the bounded
# timeouts of rewardhack.containers.docker_problem and rewardhack.task.recorded_image_digest), the
# local sandbox image is the one sandbox/image_digest.txt records, both seeds exist under tasks/,
# no INSPECT_* variable but INSPECT_DISPLAY and INSPECT_NO_ANSI is set in the environment uv gives the
# runs or in the .env file Inspect would load, no selected part's logs come from another harness, and,
# for a real run, a .env file exists.
# INSPECT_EVAL_* would set an option the command leaves out, and the other INSPECT_* variables set the log
# format (INSPECT_LOG_FORMAT=json would write logs scripts/main_status.py's spend check cannot count), sandbox
# limits and the like; this is stricter than the launcher the pilot ran with, which refused only INSPECT_EVAL_*.
# The .env check reads the file with python-dotenv's dotenv_values, the parser Inspect loads it with, and
# keeps only the names of the INSPECT_* keys that would be set; it never uses or prints a value.
# Each part loads .env anew, but this launcher, whose pilot has run, checks it only in the preflight.
# Every uv command passes --no-env-file, so neither UV_ENV_FILE nor uv's --env-file can add options.
# Every run also records costs.yaml's path and sha256 as eval metadata, because the log header has
# no field for --model-cost-config.
#
# The launch logic is scripts/launch_lib.sh, which scripts/run_main.sh (the full runs) sources too, so
# the two launchers share every flag and check above and cannot drift apart.
# The pilot and the full runs share Docker, so this script also refuses while run_main.sh's lock
# .run_main.lock exists in logs/main or an eval-set of the task names a --log-dir under logs/main
# (under the mock override, while .run_main.lock exists in PILOT_LOG_ROOT), and while any variable
# whose name starts with MAIN_ (ignoring case), a setting of run_main.sh, is set.

set -euo pipefail

PHASE=pilot
# shellcheck source=launch_lib.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)/launch_lib.sh"
