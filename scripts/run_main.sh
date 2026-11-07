#!/usr/bin/env bash
# Main launcher for docs/PLAN.md build order step 4, the full runs of PLAN's "Runs" table: all 8 tasks
# (022_0, 022_2, 078_0, 327_0, 377_0, 732_2, 918_0 and 943_0) in both variants, with Haiku x12 normal
# and x24 conflict, gpt-oss x12 and x24, and Sonnet x5 and x5 epochs (656 episodes), as six inspect
# eval-set runs one after another (Haiku normal, Haiku conflict, gpt-oss normal, gpt-oss conflict,
# Sonnet normal, Sonnet conflict), each with its own log directory logs/main/<agent>-<variant> (passed to
# eval-set as an absolute path under this repository) and PLAN's flags.
# The launch logic is scripts/launch_lib.sh, which scripts/run_pilot.sh sources too, so this launcher has
# every flag and every safety check of the pilot launcher, with the same per-agent model settings and
# per-sample cost limits, and the two cannot drift apart.
#
# Usage, from any directory:
#   scripts/run_main.sh                    run every part (paid API calls)
#   DRY_RUN=1 scripts/run_main.sh          print the commands, one per line, and exit; nothing runs
#   AGENTS="gpt-oss" scripts/run_main.sh   run only the named agents' parts (haiku, gpt-oss, sonnet)
#   uv run python scripts/main_status.py   at any time: each part's progress and the spend against the caps
#
# Resuming: re-invoke the same command, with AGENTS to skip the parts that finished.
# eval-set keeps each log directory's eval-set id, returns a task whose log already holds every
# selected sample and epoch without running it, and re-runs only the missing or errored samples of
# an incomplete log; a log directory holding a log from a different command (another model, task
# argument, generate config or limit) makes eval-set refuse to start.
# A sample that still errors after --retry-on-error in a log that finished stays in that log as an
# errored sample, which eval-set does not re-run and PLAN counts per agent and variant and excludes.
# --sample-id here only selects the eight tasks, the same on every invocation; it is never a
# recovery list, and recovery never uses inspect's retry command, which drops --cost-limit.
#
# A harness change between parts: eval-set's task identity covers no code, so after a harness change
# it would reuse a part's old samples.
# Every run therefore records harness_sha256 as eval metadata, a sha256 over every file under
# HARNESS_PATHS (costs.yaml, uv.lock, sandbox, src/rewardhack and all eight task directories, so a task
# edit counts; bytecode caches, dotfiles, src/rewardhack/monitors and src/rewardhack/analysis left out),
# and before running anything the launcher refuses while a selected part's log directory holds a log
# with a different or missing one.
# Move each directory it names from logs/main into logs/main-rejected/ (at any depth), where eval-set no
# longer reuses its samples, and re-run that part.
# scripts/main_status.py still reads logs/main-rejected by default (--rejected-logs), for spend only:
# it counts each of those logs' spend against the cap of the agent whose model the log ran, and treats a
# log anywhere else under logs/main, outside the six part directories, as bad input.
# Each part is a new process that loads the harness from disk, so the launcher hashes costs.yaml and
# the harness again before and after each part: it refuses to start a part when either changed since
# launch, and fails when either changed while a part ran, naming that part's log directory to move aside.
# So no file under HARNESS_PATHS may be edited while a full run is in progress, not even
# src/rewardhack/label.py or src/rewardhack/containers.py, which no episode imports: work on them (as for
# PLAN step 5) waits until the run ends or goes in another copy of the repository, since a part stopped this
# way becomes rejected logs, and any rejected Haiku or Sonnet spend leaves that agent no room under its cap.
#
# One launcher at a time across both phases, since the pilot and the full runs share Docker: a run holds
# the lock directory .run_main.lock in the log root (logs/main, or MAIN_LOG_ROOT) and refuses to start
# while that lock exists, while run_pilot.sh's lock .run_pilot.lock exists in logs/pilot, or while any
# eval-set of the task names a --log-dir under logs/main or logs/pilot, written as these scripts write it
# (an absolute path, so eval-sets of other copies of this repository do not count).
# run_pilot.sh refuses the same way while this launcher or an eval-set under logs/main runs.
# Under the mock override both locks and the eval-set check are in MAIN_LOG_ROOT alone.
# Stopping the launcher (SIGINT, SIGTERM or SIGHUP) stops the running part: it interrupts the
# eval-set, which cancels its samples and cleans up its sandboxes, and terminates it if it is still
# running after STOP_GRACE seconds.
# A launcher killed with SIGKILL cannot do that or remove its lock: the next one, of either phase, names
# the orphaned eval-set while it runs, and afterwards refuses until an operator who has checked that no
# run is going removes the stale lock by hand.
#
# Every command passes --log-model-api, because by default Inspect keeps the raw call of only the
# first 5 model events per sample, and each gpt-oss response's provider stays checkable as in the pilot.
# The per-sample cost limits (Haiku $0.10, gpt-oss $0.05, Sonnet $1.00) times each agent's samples
# come to $28.80, $14.40 and $80.00; the first and last are PLAN's Haiku and Sonnet hard caps, and the
# $15.50 gpt-oss cap is shared with the pilot and the smoke test.
# The limits are checked after each generation, and retried errors run with fresh limits, so a part can
# spend more than its samples times its limit (PLAN's Budget allows for this); any spend above a cap
# needs Jessup's approval first, and scripts/main_status.py reports each agent's spend against its cap.
# So before a real run, the preflight refuses to start any part while a selected agent's spend so far
# (scripts/main_status.py's count, rejected logs included), or that spend plus every sample its parts still
# have to run (missing or errored ones of an unfinished part, none of a finished one) times its cost limit,
# is above the agent's cap (for gpt-oss, $15.50 less the pilot's and smoke test's OpenRouter spend), or
# while scripts/main_status.py finds bad input; an approved higher cap goes into PLAN's Budget and
# AGENT_CAP in scripts/pilot_report.py.
# Before each later part the launcher checks that part's agent the same way again, and the Inspect
# variables in .env too (each part is a new eval-set, which loads .env anew); when either refuses, the part
# does not start and the launcher exits naming the remaining agents and saying that an agent refused for its
# cap stays out of AGENTS until Jessup approves more spend, so a part that took its agent past this stops the
# agent's next part.
#
# Test-only mock override (no paid calls); set all three or none:
#   MAIN_MODEL_OVERRIDE=mockllm/model  every part runs mockllm/model, without gpt-oss's -M routing
#   MAIN_EPOCHS_OVERRIDE=<n>           epochs for every part
#   MAIN_LOG_ROOT=<absolute dir>       log directories become <dir>/<agent>-<variant>
# Setting only some of them (even to an empty value), any other model, or a log root inside this
# repository (compared after resolving symlinks, ignoring case) is refused.
# So is any other variable whose name starts with MAIN_ or contains DRY (both ignoring case) besides
# DRY_RUN, such as a misspelled MAIN_MODEL or DRYRUN that would otherwise leave a paid run, and any
# variable whose name starts with AGENT (ignoring case) other than exactly AGENTS, such as AGENT or
# agents, which would otherwise be ignored and run every part. AGENT counts only at the start of a
# name, because Claude Code sets AI_AGENT and CLAUDE_AGENT_SDK_* and ssh-agent sets SSH_AGENT_PID.
# Any variable whose name starts with PILOT_ (ignoring case) is refused too: it is a setting of
# run_pilot.sh, which this script would otherwise ignore silently.
# So is any variable whose name holds MODEL_OVERRIDE, EPOCHS_OVERRIDE or LOG_ROOT (ignoring case) other than
# the three above, such as MIAN_MODEL_OVERRIDE or a bare LOG_ROOT, which would otherwise leave the paid runs.
# So are PYTHONPATH, PYTHONHOME and PYTHONSTARTUP, which could load other code (a sitecustomize.py, say)
# into every run while its log records this repository's harness_sha256, and for the same reason every
# UV_* variable but UV_RUN_RECURSION_DEPTH (which uv run sets), UV_CACHE_DIR, UV_NO_PROGRESS, UV_ENV_FILE
# and UV_NO_ENV_FILE: UV_WORKING_DIR, UV_PROJECT, UV_PROJECT_ENVIRONMENT, UV_NO_SYNC, UV_FROZEN,
# UV_NO_EDITABLE, UV_CONFIG_FILE and the like would make every uv command use another tree's code,
# costs.yaml or scripts/main_status.py, or another environment.
#
# Before running anything the script refuses to start unless Docker answers (within the bounded
# timeouts of rewardhack.containers.docker_problem and rewardhack.task.recorded_image_digest), the
# local sandbox image is the one sandbox/image_digest.txt records, each of the eight task directories
# exists under tasks/ with the task.json that makes the task load it, no INSPECT_* variable but
# INSPECT_DISPLAY and INSPECT_NO_ANSI is set in the environment uv gives the runs or in the .env file
# Inspect would load, no selected part's logs come from another harness, and, for a real run, a .env
# file exists.
# INSPECT_EVAL_* would set an option the command leaves out, and the other INSPECT_* variables set the log
# format (INSPECT_LOG_FORMAT=json would write logs scripts/main_status.py's spend check cannot count), sandbox
# limits and the like.
# The .env check reads the file with python-dotenv's dotenv_values, the parser Inspect loads it with, and
# keeps only the names of the INSPECT_* keys that would be set; it never uses or prints a value.
# Every uv command passes --no-env-file, so neither UV_ENV_FILE nor uv's --env-file can add options.
# Every run also records costs.yaml's path and sha256 as eval metadata, because the log header has
# no field for --model-cost-config.

set -euo pipefail

PHASE=main
# shellcheck source=launch_lib.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)/launch_lib.sh"
