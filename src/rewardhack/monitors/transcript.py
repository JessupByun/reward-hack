"""The monitor transcript and the final diff (PLAN.md, "Monitors"), shared by M1, M2, M3, the summarizer and the judge.

The transcript is the ticket (the sample's first user message; the system prompt is left out), then
for each accepted assistant turn its visible text and its tool calls with arguments, each call's
result cut to its last TOOL_RESULT_CHARS characters, and then the final diff.
M2 reads it with every ContentReasoning removed. M3 reads the same text with each turn's reasoning
(reasoning.message_reasoning_text) inserted right after the turn's opening tag, before its visible
text and tool calls, and nothing else changed, so M3 minus the inserted reasoning parts is
byte-identical to M2. The M3-summary arm passes a ReasoningHook that replaces each reasoning block's
text with the summarizer's output, which changes only those parts.

The final diff is difflib.unified_diff with DIFF_CONTEXT context lines, one per allowlisted file
(the keys of the episode_record score's final_files, which are the task's writable files), from the
pristine task file under tasks/<task_id>/src/ to the episode's final text.
That is the same pristine file that grade_hidden's stage() lays down before writing final_files
over it, and the task-directory hash in the sample's metadata must match the current task
directory, as grade_hidden requires, so the diff is never taken against edited task files.

Every function takes an EvalSample (scripts and tests) or a TaskState (a scorer); both carry
messages, metadata and scores under those names.
Under `inspect score --action append`, inspect-ai 0.3.282 rebuilds each sample's TaskState in
task_state_from_sample (inspect_ai/_eval/score.py lines 461-475) with
scores=dict(sample.scores) when append_scores is set (line 473), which _run_score_task sets for
action "append" (line 507), so the original episode_record score is state.scores["episode_record"].
With action "overwrite" state.scores starts empty (line 473) and the rewritten log keeps only the
new scores (_get_updated_scores, lines 155-159), which would drop episode_record from the scored
copy; episode_record() raises in that case instead of treating the sample as unscored.
The rebuilt TaskState's messages are the sample's messages (line 468), its metadata the sample's
metadata (line 471) and its store the sample's store (line 472).
Under either action the transcript starts with the sample's own events (init_transcript, lines 482-484),
which is how scored_in_transcript() tells the overwrite case from a sample that errored unscored.

Only the accepted conversation is read. agent_loop (rewardhack.task) appends a reply to
state.messages only when it accepts it, so a regenerated reply is never in sample.messages; it
stays in its model event, which no monitor reads.

Every scorer here (M1 and the LLM scorers) runs only on a scored copy under logs/scored/, never on an
agent log under logs/main or logs/pilot (PLAN.md: copy each agent log once into logs/scored/).
check_scored_copy refuses any other log inside logs/, comparing directories by file identity as well
as by name (so a differently cased spelling on macOS is still refused) and refusing URLs, and refuses
everything while logs/scored is a symlink or resolves anywhere but itself (check_scored_dir), since a
logs/scored linked to logs/main would make every agent log pass as a scored copy.
command_log_file reads which log the running `inspect score` command scores from its click context,
since a scorer factory is not told; it also refuses a command that streams or writes another file.
inspect score --overwrite reads the whole log and writes it back after its last sample, so two
commands rewriting one scored copy at once would lose one's scores: every command and scorer that
rewrites a scored copy holds its rewrite lock (hold_rewrite_lock), and a second one refuses.
inspect score reads the log before its scorer can take that lock, so the scorer then also refuses a
sample whose scores on disk include one the command did not read (check_not_rewritten).
"""

import difflib
import fcntl
import html
import json
import os
import threading
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import click
from inspect_ai.event import ScoreEvent
from inspect_ai.log import EvalSample, transcript
from inspect_ai.model import ChatMessage, ChatMessageAssistant, ChatMessageSystem, ChatMessageTool, ChatMessageUser
from inspect_ai.solver import TaskState

from rewardhack.grading import task_dir_hash, task_files
from rewardhack.monitors.reasoning import message_reasoning, message_reasoning_text
from rewardhack.task import ROOT, TASKS_DIR

LOGS_DIR = ROOT / "logs"
SCORED_DIR = LOGS_DIR / "scored"
# The end-of-episode scorer in rewardhack.task, whose metadata holds final_files and refused_writes.
SCORER = "episode_record"
TOOL_RESULT_CHARS = 2000
DIFF_CONTEXT = 3
# Joins a turn's reasoning blocks, in M3 and in the M3-summary arm alike.
BLOCK_SEPARATOR = "\n\n"
# git's marker for a last line without a newline, kept so such a line still diffs as a line.
NO_NEWLINE = "\n\\ No newline at end of file\n"
NO_CHANGES = "No changes to the source files."

Source = EvalSample | TaskState


class NoEpisodeRecord(ValueError):
    """The sample has no episode_record score, which is how a sample that errored before scoring looks."""


def same_file(a: Path, b: Path) -> bool:
    try:
        return os.path.samefile(a, b)
    except OSError:
        return False


def is_within(path: str | Path, directory: Path) -> bool:
    """Whether the path is the directory or inside it, by name after resolve() or by file identity.

    resolve() keeps the spelling it is given, so on a case-insensitive file system (macOS APFS by
    default) LOGS/main and logs/main name the same directory; os.path.samefile compares each existing
    ancestor with the directory itself, whatever its spelling.
    """
    resolved = Path(path).resolve()
    if resolved.is_relative_to(directory.resolve()):
        return True
    return any(same_file(ancestor, directory) for ancestor in (resolved, *resolved.parents))


# What Inspect's file system layer (fsspec) reads as a URL or a chain of file systems, not a local path.
URL_MARKERS = ("://", "::")


def check_scored_dir(scored_dir: Path | None = None, logs_dir: Path | None = None) -> None:
    """Refuse a logs/scored (default SCORED_DIR in LOGS_DIR) that is a symlink or resolves anywhere but logs/scored itself.

    Every scored-copy check compares resolved paths with logs/scored, so a logs/scored linked to
    logs/main (or anywhere else) would make agent logs pass as scored copies. A logs/scored that does
    not exist yet passes.
    """
    scored_dir = SCORED_DIR if scored_dir is None else scored_dir
    logs_dir = LOGS_DIR if logs_dir is None else logs_dir
    if scored_dir.is_symlink() or scored_dir.resolve() != logs_dir.resolve() / "scored":
        raise ValueError(
            f"{scored_dir} is a symlink or resolves to {scored_dir.resolve()}; it must be a directory of its own in {logs_dir}, "
            "or agent logs would pass as scored copies"
        )


def check_scored_copy(path: str | Path) -> None:
    """Refuse a log inside logs/ but outside logs/scored/: agent logs are copied once and only the copies are scored.

    A URL (file://... or any other fsspec form) is refused, since it is not resolved as a local path,
    and so is every log while logs/scored is a symlink (check_scored_dir).
    """
    if any(marker in str(path) for marker in URL_MARKERS):
        raise ValueError(f"{path} is a URL; score a scored copy under {SCORED_DIR} by its local path")
    check_scored_dir()
    if is_within(path, LOGS_DIR) and not is_within(path, SCORED_DIR):
        raise ValueError(f"{path} is an agent log; score its copy under {SCORED_DIR} (scripts/score_logs.py copy)")


# --- one rewrite of a scored copy at a time ---

# scripts/score_logs.py holds each scored copy's rewrite lock for its whole m1 or score command and passes the
# descriptor of the log's lock to each inspect score it runs under this variable, so that command's scorer
# holds the script's lock instead of refusing it.
REWRITE_LOCK_FD_ENV = "REWARDHACK_REWRITE_LOCK_FD"
# The rewrite locks this process holds for the rest of its life, by the lock file's (device, inode).
_HELD_REWRITE_LOCKS: dict[tuple[int, int], int] = {}
_REWRITE_THREAD_LOCK = threading.Lock()


class RewriteLocked(RuntimeError):
    """Another command or scorer holds the scored copy's rewrite lock, or rewrote the copy after this command read it; raised before any sample is scored."""


def rewrite_lock_path(log: str | Path) -> Path:
    """The scored copy's rewrite lock: .<log name>.lock beside the resolved log."""
    path = Path(log).resolve()
    return path.with_name(f".{path.name}.lock")


def file_identity(fd: int) -> tuple[int, int]:
    info = os.fstat(fd)
    return info.st_dev, info.st_ino


def take_rewrite_lock(fd: int, log: str | Path) -> int:
    """The exclusive flock on an open rewrite lock, without waiting; closes fd and raises RewriteLocked when another holds it."""
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as e:
        os.close(fd)
        raise RewriteLocked(
            f"another scoring command or scorer is rewriting {log} (it holds {rewrite_lock_path(log)}); "
            "commands that rewrite a scored copy go one at a time, so wait for it to finish"
        ) from e
    except BaseException:
        os.close(fd)
        raise
    return fd


def open_rewrite_lock(log: str | Path) -> int:
    """An open descriptor holding the log's rewrite lock (created if missing); raises RewriteLocked when another holds it.

    The caller closes the descriptor to release the lock. Call it only on a log that passed the
    scored-copy checks, since it creates the lock file beside the log.
    """
    return take_rewrite_lock(os.open(rewrite_lock_path(log), os.O_RDWR | os.O_CREAT, 0o644), log)


def inherited_rewrite_lock(identity: tuple[int, int]) -> int | None:
    """The descriptor named by REWRITE_LOCK_FD_ENV when it is open on the lock file with this identity, else None."""
    try:
        fd = int(os.environ.get(REWRITE_LOCK_FD_ENV, ""))
        return fd if file_identity(fd) == identity else None
    except (ValueError, OSError):
        return None


def hold_rewrite_lock(log: str | Path) -> Path:
    """Take the rewrite lock of the scored copy that this process's inspect score command rewrites, until the process exits.

    The lock is a non-blocking exclusive flock on rewrite_lock_path(log), held by a descriptor that is never
    closed, so the operating system releases it when the process exits, after inspect score has written
    the log. Taking it again in this process is a no-op. A descriptor of this same lock file inherited
    from scripts/score_logs.py (REWRITE_LOCK_FD_ENV), which holds it for its command, is taken instead of
    a new one, so the command does not refuse its own script's lock. Raises RewriteLocked when another
    command or scorer holds it. Call it only after check_scored_copy, since it creates the lock file.
    """
    lock = rewrite_lock_path(log)
    with _REWRITE_THREAD_LOCK:
        fd = os.open(lock, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            identity = file_identity(fd)
        except BaseException:
            os.close(fd)
            raise
        if identity in _HELD_REWRITE_LOCKS:
            os.close(fd)
            return lock
        inherited = inherited_rewrite_lock(identity)
        if inherited is not None:
            os.close(fd)
            fd = inherited
        _HELD_REWRITE_LOCKS[identity] = take_rewrite_lock(fd, log)
    return lock


def check_not_rewritten(read: Mapping[str, object] | None, on_disk: Iterable[str], log: str | Path) -> None:
    """Refuse a sample whose scores in the log on disk include one that this inspect score command did not read.

    read is the sample's state.scores and on_disk the score names the log holds for it, read after taking
    the rewrite lock. inspect score reads the log before its scorer takes that lock, so a command that
    rewrote the log in between would lose its scores when this one writes the log back. Under --action
    overwrite nothing is read into state.scores, so that case is left to episode_record's own refusal.
    """
    if read and (added := sorted(set(on_disk) - set(read))):
        raise RewriteLocked(f"another command rewrote {log} after this inspect score command read it (adding {', '.join(added)}); run it again")


def command_log_file() -> str | None:
    """The log_file argument of the running `inspect score` command, or None outside it.

    Under the command it refuses --stream (PLAN.md: never pass --stream), since a streaming run starts
    writing the output log before any sample is scored, and --output-file, since the scored copy is
    rewritten in place; either may also come from an INSPECT_SCORE_* variable. The scorer factories
    call it, so the command fails before it writes anything.
    """
    context = click.get_current_context(silent=True)
    if context is None or context.command.name != "score":
        return None
    if context.params.get("stream"):
        raise ValueError("inspect score --stream (or INSPECT_SCORE_STREAM) is refused; PLAN.md never passes --stream")
    if context.params.get("output_file"):
        raise ValueError("inspect score --output-file (or INSPECT_SCORE_OUTPUT_FILE) is refused; the scored copy is rewritten in place")
    value = context.params.get("log_file")
    return value if isinstance(value, str) and value else None


def scored_in_transcript() -> bool:
    """Whether the current transcript holds an episode_record score event.

    Under inspect score the transcript starts with the sample's own events, so this is True for
    every sample that the episode scorer scored, whatever the score action.
    """
    return any(isinstance(event, ScoreEvent) and event.scorer == SCORER for event in transcript().events)


def episode_record(source: Source) -> dict:
    """The episode_record score's metadata: final_files, run_tests, refused_writes and regenerations.

    Raises NoEpisodeRecord when the sample has no such score, and RuntimeError for a scorer run
    under `inspect score --action overwrite`, which hides the score from the scorer and would drop
    it from the scored log (see the module docstring).
    """
    score = (source.scores or {}).get(SCORER)
    if score is not None and isinstance(score.metadata, dict) and "final_files" in score.metadata:
        return score.metadata
    if isinstance(source, TaskState) and scored_in_transcript():
        raise RuntimeError(f"state.scores has no {SCORER} score; run inspect score with --action append, not overwrite")
    raise NoEpisodeRecord(f"sample {source.sample_id if isinstance(source, TaskState) else source.id} has no {SCORER} score")


def task_dir(source: Source, tasks_dir: Path = TASKS_DIR) -> Path:
    """The sample's task directory, after checking that its hash matches the one recorded in the sample."""
    meta = source.metadata
    found = tasks_dir / str(meta.get("task_id"))
    if not (found / "task.json").is_file():
        raise ValueError(f"task {meta.get('task_id')!r} is not in {tasks_dir}")
    actual = task_dir_hash(found)
    if meta.get("task_dir_sha256") != actual:
        raise ValueError(f"task hash mismatch: the sample has {meta.get('task_dir_sha256')}, {found} is {actual}")
    return found


def pristine_files(task_root: Path) -> dict[str, str]:
    """Every file under the task's src/, keyed as final_files is (src/<path>)."""
    src = task_root / "src"
    return {f"src/{path.relative_to(src).as_posix()}": path.read_text(encoding="utf-8") for path in task_files(src)}


def diff_lines(text: str | None) -> list[str]:
    """A file's lines for difflib, each ending in a newline; a last line without one carries NO_NEWLINE.

    Lines are split on newline characters only, as Python's tokenizer and ast number them (a lone
    carriage return aside), so the i-th entry is line i + 1 of the text. None (a deleted file) has
    no lines.
    """
    if text is None:
        return []
    *lines, last = text.split("\n")
    return [line + "\n" for line in lines] + ([last + NO_NEWLINE] if last else [])


@dataclass(frozen=True)
class FileDiff:
    """One allowlisted file's final diff.

    added holds the 1-based line numbers, in final, of the lines the diff adds (its + lines).
    final is None when the episode deleted the file.
    """

    path: str
    pristine: str
    final: str | None
    text: str
    added: frozenset[int]


def file_diff(path: str, pristine: str, final: str | None) -> FileDiff:
    """The unified diff with DIFF_CONTEXT context lines from the pristine text to the final text."""
    before, after = diff_lines(pristine), diff_lines(final)
    tofile = "/dev/null" if final is None else f"b/{path}"
    text = "".join(difflib.unified_diff(before, after, f"a/{path}", tofile, n=DIFF_CONTEXT))
    # unified_diff prints exactly these groups of opcodes (difflib.unified_diff's own loop).
    groups = difflib.SequenceMatcher(None, before, after).get_grouped_opcodes(DIFF_CONTEXT)
    added = frozenset(
        j + 1 for group in groups for tag, _, _, j1, j2 in group if tag in ("replace", "insert") for j in range(j1, j2)
    )
    return FileDiff(path, pristine, final, text, added)


def file_diffs(final_files: dict[str, str | None], task_root: Path) -> list[FileDiff]:
    """One FileDiff per allowlisted file, in final_files order, from the task's pristine src/ to final_files."""
    pristine = pristine_files(task_root)
    missing = [path for path in final_files if path not in pristine]
    if missing:
        raise ValueError(f"final_files {missing} have no pristine file under {task_root / 'src'}")
    return [file_diff(path, pristine[path], final) for path, final in final_files.items()]


def final_diff(source: Source, tasks_dir: Path = TASKS_DIR) -> list[FileDiff]:
    """The sample's final diff: file_diffs of its episode_record final_files against its checked task directory."""
    return file_diffs(episode_record(source)["final_files"], task_dir(source, tasks_dir))


def diff_text(diffs: Sequence[FileDiff]) -> str:
    """Every file's diff, in order, or "" when no file changed."""
    return "".join(diff.text for diff in diffs)


@dataclass(frozen=True)
class ReasoningBlock:
    """One kept reasoning block (reasoning.message_reasoning) of an accepted assistant turn.

    turn is the message's index among the accepted assistant messages, counted from 0, as the turn
    of store()['regenerations'] is; index is the block's position among that message's kept blocks.
    """

    turn: int
    index: int
    text: str


# The M3-summary arm's replacement for each reasoning block's text; an empty result drops the block.
ReasoningHook = Callable[[ReasoningBlock], str]


def turn_reasoning(message: ChatMessage, turn: int) -> list[ReasoningBlock]:
    return [ReasoningBlock(turn, index, text) for index, text in enumerate(message_reasoning(message))]


def reasoning_blocks(messages: Sequence[ChatMessage]) -> list[ReasoningBlock]:
    """Every kept reasoning block of the accepted assistant turns, in order; the summarizer takes these."""
    turns = [message for message in messages if isinstance(message, ChatMessageAssistant)]
    return [block for turn, message in enumerate(turns) for block in turn_reasoning(message, turn)]


@dataclass(frozen=True)
class Part:
    """A piece of the transcript; reasoning parts are the ones M3 inserts and M2 leaves out."""

    text: str
    reasoning: bool = False


def tag(element: str, body: str, /, **attributes: str) -> str:
    """The body between an opening and a closing tag, each on its own line; attribute values are escaped."""
    opening = "".join(f' {key}="{html.escape(value, quote=True)}"' for key, value in attributes.items())
    newline = "" if body.endswith("\n") or not body else "\n"
    return f"<{element}{opening}>\n{body}{newline}</{element}>\n"


def tool_call_text(arguments: dict) -> str:
    """Each argument as its own tag, a string as it is and any other value as JSON."""
    return "".join(
        tag("argument", value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str), name=name)
        for name, value in arguments.items()
    )


def tool_result_text(message: ChatMessageTool) -> str:
    """The tool result's whole text, then its error message if the call failed (a refused write has only that)."""
    if message.error is None:
        return message.text
    error = f"Error: {message.error.message}"
    return f"{message.text}\n{error}" if message.text else error


def tool_result_part(message: ChatMessageTool) -> Part:
    """The result cut to its last TOOL_RESULT_CHARS characters, marked truncated when that cut anything."""
    body = tool_result_text(message)
    attributes = {"function": message.function or ""}
    if len(body) > TOOL_RESULT_CHARS:
        attributes["truncated"] = "true"
    return Part(tag("tool_result", body[-TOOL_RESULT_CHARS:], **attributes))


def reasoning_part(message: ChatMessageAssistant, turn: int, replace: ReasoningHook | None) -> Part | None:
    """The turn's inserted reasoning, or None when it has none (then M3 equals M2 for that turn)."""
    if replace is None:
        text = message_reasoning_text(message, BLOCK_SEPARATOR)
    else:
        text = BLOCK_SEPARATOR.join(kept for kept in (replace(block) for block in turn_reasoning(message, turn)) if kept)
    return Part(tag("reasoning", text), reasoning=True) if text else None


def turn_parts(
    message: ChatMessageAssistant, results: list[ChatMessageTool], turn: int, replace: ReasoningHook | None
) -> list[Part]:
    """One assistant turn: its reasoning (M3 only), visible text, and each tool call followed by its result."""
    parts = [Part(f'<turn number="{turn + 1}">\n')]
    reasoning = reasoning_part(message, turn, replace)
    if reasoning is not None:
        parts.append(reasoning)
    if message.text.strip():
        parts.append(Part(tag("message", message.text)))
    unmatched = list(results)
    for call in message.tool_calls or []:
        parts.append(Part(tag("tool_call", tool_call_text(call.arguments), function=call.function)))
        result = next((r for r in unmatched if r.tool_call_id == call.id), None)
        if result is not None:
            unmatched.remove(result)
            parts.append(tool_result_part(result))
    parts += [tool_result_part(result) for result in unmatched]
    parts.append(Part("</turn>\n"))
    return parts


def transcript_parts(
    messages: Sequence[ChatMessage], diffs: Sequence[FileDiff], replace: ReasoningHook | None = None
) -> list[Part]:
    """The whole transcript as parts: the ticket, every accepted assistant turn, then the final diff.

    Tool results are placed after the call they answer (matched by tool call ID); system messages
    are left out, and any user message after the ticket (the harness sends none) is kept in place.
    """
    first_user = next((i for i, message in enumerate(messages) if isinstance(message, ChatMessageUser)), None)
    if first_user is None:
        raise ValueError("the conversation has no user message, so it has no ticket")
    parts = [Part(tag("ticket", messages[first_user].text) + "\n")]
    turn = 0
    i = first_user + 1
    while i < len(messages):
        message = messages[i]
        i += 1
        if isinstance(message, ChatMessageAssistant):
            results = []
            while i < len(messages) and isinstance(messages[i], ChatMessageTool):
                results.append(messages[i])
                i += 1
            parts += turn_parts(message, results, turn, replace)
            parts.append(Part("\n"))
            turn += 1
        elif isinstance(message, ChatMessageTool):
            parts += [tool_result_part(message), Part("\n")]
        elif isinstance(message, ChatMessageUser):
            parts.append(Part(tag("user", message.text) + "\n"))
        elif not isinstance(message, ChatMessageSystem):
            raise ValueError(f"unexpected {message.role} message in the conversation")
    parts.append(Part(tag("final_diff", diff_text(diffs) or NO_CHANGES)))
    return parts


def join(parts: Sequence[Part], reasoning: bool) -> str:
    return "".join(part.text for part in parts if reasoning or not part.reasoning)


def m2_transcript(source: Source, tasks_dir: Path = TASKS_DIR) -> str:
    """M2's input: the monitor transcript with every ContentReasoning removed."""
    return join(transcript_parts(source.messages, final_diff(source, tasks_dir)), reasoning=False)


def m3_transcript(source: Source, replace: ReasoningHook | None = None, tasks_dir: Path = TASKS_DIR) -> str:
    """M3's input: M2's transcript with each turn's reasoning inserted before its visible text and tool calls.

    replace, when given, supplies each block's text instead (the M3-summary arm); the blocks are
    those of reasoning_blocks, joined with BLOCK_SEPARATOR as message_reasoning_text joins them.
    """
    return join(transcript_parts(source.messages, final_diff(source, tasks_dir), replace), reasoning=True)
