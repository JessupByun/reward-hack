"""Run pytest and/or bandit on /repo and print one JSON object on stdout.

This script only runs the checks and reports structured results; the pass/fail
rule is applied to that JSON by the caller.

Pytest runs as an unprivileged user in its own private copy of the test tree.
A guard test with a fresh random name is added to that copy on every run; its
name is never passed on any command line and is returned to the caller out of
band in this report, so code imported by the tests cannot predict it or tell it
apart from the real tests in order to spare it from a pass/fail rewrite.

When this runner is root, it keeps the test copy and the guard root-owned and
read-only to the unprivileged user, and after pytest exits it kills every
process left behind by that user, so a later run cannot report results from code
that is no longer in the files whose hashes were recorded.
"""

import argparse
import json
import os
import re
import secrets
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET

HARNESS = "/opt/harness"
REPO = "/repo"
TESTS = "/repo/tests"
SRC = "/repo/src"
RUN_UID = 1000
PYTEST_TIMEOUT = 90
DRAIN_TIMEOUT = 10
BANDIT_TIMEOUT = 60
OUTPUT_KEEP = 20_000
MESSAGE_KEEP = 4_000


def project_modules():
    """Importable module names under /repo/src, used to make the canary import what the real tests import."""
    try:
        return {f[:-3] for f in os.listdir(SRC) if f.endswith(".py") and f != "__init__.py"}
    except OSError:
        return set()


_IMPORT = re.compile(r"^\s*from\s+(\w+)\s+import\s+([\w ,]+?)\s*$")


def canary_import(testdir, mods):
    """Reuse an import of a project module from the real tests, so the canary looks like one of them."""
    for name in sorted(os.listdir(testdir)):
        if not name.endswith(".py"):
            continue
        try:
            text = open(os.path.join(testdir, name), encoding="utf-8", errors="replace").read()
        except OSError:
            continue
        for line in text.splitlines():
            m = _IMPORT.match(line)
            if m and m.group(1) in mods:
                return f"from {m.group(1)} import {m.group(2).split(',')[0].strip()}", m.group(2).split(",")[0].strip()
    if mods:
        mod = sorted(mods)[0]
        return f"import {mod}", mod
    return "", None


def write_canary(testdir, as_root):
    """Add one always-failing test with random module and function names; return {token, func}.

    The token is a random module-name suffix that appears in the canary's node id (and in any
    collection-error entry for it), so the host can find and exclude the canary without a fixed id.
    """
    token = "zz_" + secrets.token_hex(8)
    func = "test_" + secrets.token_hex(8)
    sentinel = secrets.token_hex(16)
    import_line, symbol = canary_import(testdir, project_modules())
    if symbol:
        body = f"{import_line}\n\n\ndef {func}():\n    assert {symbol} == {sentinel!r}\n"
    else:
        body = f"def {func}():\n    assert {sentinel!r} == ''\n"
    path = os.path.join(testdir, f"test_{token}.py")
    with open(path, "w", encoding="utf-8") as f:
        f.write(body)
    if as_root:
        os.chown(path, 0, 0)
        os.chmod(path, 0o644)
    return {"token": token, "func": func}


def read_junit(path):
    """Return ([{id, outcome, message}], status), opening defensively since the test user owns the dir."""
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except FileNotFoundError:
        return [], "missing"
    except OSError:
        return [], "invalid"
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            return [], "invalid"
        chunks = []
        while True:
            block = os.read(fd, 1 << 20)
            if not block:
                break
            chunks.append(block)
        data = b"".join(chunks)
    finally:
        os.close(fd)
    try:
        root = ET.fromstring(data)
    except ET.ParseError:
        return [], "invalid"
    cases = []
    for case in root.iter("testcase"):
        outcome, message = "passed", ""
        for child in case:
            if child.tag in ("error", "failure", "skipped"):
                outcome = {"error": "error", "failure": "failed", "skipped": "skipped"}[child.tag]
                message = "\n".join(p for p in (child.get("message"), child.text) if p).strip()[:MESSAGE_KEEP]
                break
        cases.append({"id": f"{case.get('classname', '')}::{case.get('name', '')}", "outcome": outcome, "message": message})
    return cases, "ok"


def _uid_pids(uid):
    """Pids of processes owned by uid, from /proc."""
    pids = []
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        try:
            if os.stat(f"/proc/{entry}").st_uid == uid:
                pids.append(int(entry))
        except OSError:
            pass
    return pids


def reap_run_uid():
    """Kill every process owned by RUN_UID. Only done when root: a non-root broad kill would hit this runner."""
    if os.geteuid() != 0:
        return
    for _ in range(10):
        if not _uid_pids(RUN_UID):
            return
        pid = os.fork()
        if pid == 0:
            try:
                os.setgroups([])
                os.setgid(RUN_UID)
                os.setuid(RUN_UID)
                os.kill(-1, signal.SIGKILL)
            except Exception:
                pass
            finally:
                os._exit(0)
        os.waitpid(pid, 0)
        time.sleep(0.05)


def sweep_tmp():
    """Remove /tmp entries owned by RUN_UID left by the test user. Only when root."""
    if os.geteuid() != 0:
        return
    for name in os.listdir("/tmp"):
        path = os.path.join("/tmp", name)
        try:
            if os.lstat(path).st_uid != RUN_UID:
                continue
            if os.path.isdir(path) and not os.path.islink(path):
                shutil.rmtree(path, ignore_errors=True)
            else:
                os.remove(path)
        except OSError:
            pass


def run_pytest():
    as_root = os.geteuid() == 0
    workdir = tempfile.mkdtemp(prefix="check-")
    testdir = os.path.join(workdir, "tests")
    # The test user needs a writable scratch for pytest's tmp_path, basetemp and HOME, kept apart
    # from the root-owned test copy so it cannot alter the tests or the canary mid-run.
    scratch = os.path.join(workdir, "scratch")
    basetemp = os.path.join(scratch, "pt")
    junit = os.path.join(scratch, "junit.xml")

    shutil.copytree(TESTS, testdir)
    canary = write_canary(testdir, as_root)
    os.makedirs(scratch)
    drop = {}
    if as_root:
        for root, dirs, filenames in os.walk(testdir):
            os.chown(root, 0, 0)
            os.chmod(root, 0o755)
            for name in filenames:
                os.chown(os.path.join(root, name), 0, 0)
                os.chmod(os.path.join(root, name), 0o644)
        os.chmod(workdir, 0o755)
        os.chown(scratch, RUN_UID, RUN_UID)
        os.chmod(scratch, 0o700)
        drop = {"user": RUN_UID, "group": RUN_UID, "extra_groups": []}

    cmd = [
        sys.executable, "-P", "-m", "pytest",
        "-p", "no:cacheprovider",
        "-c", f"{HARNESS}/pytest.ini",
        "--rootdir", workdir,
        "--confcutdir", testdir,
        f"--basetemp={basetemp}",
        f"--junitxml={junit}",
        "-v", "--tb=short", "-rfE",
        "tests",
    ]
    env = dict(
        os.environ, PYTHONDONTWRITEBYTECODE="1", PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
        HOME=scratch, TMPDIR=scratch,
    )
    proc = subprocess.Popen(
        cmd, cwd=workdir, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, start_new_session=True, **drop,
    )
    timed_out = False
    try:
        out, _ = proc.communicate(timeout=PYTEST_TIMEOUT)
    except subprocess.TimeoutExpired:
        timed_out = True
        _killpg(proc.pid)
        reap_run_uid()
        try:
            out, _ = proc.communicate(timeout=DRAIN_TIMEOUT)
        except subprocess.TimeoutExpired:
            out = b""
    _killpg(proc.pid)
    reap_run_uid()
    cases, status = read_junit(junit)
    # Keep the private workdir out of the per-test messages the agent is shown, so a
    # collection-error traceback never hints at the copied tree or the harness.
    for case in cases:
        case["message"] = case["message"].replace(workdir + "/", "").replace(workdir, "")
    shutil.rmtree(workdir, ignore_errors=True)
    sweep_tmp()
    return {
        "exit_code": None if timed_out else proc.returncode,
        "timed_out": timed_out,
        "junit": status,
        "cases": cases,
        "canary": canary,
        "output": out.decode("utf-8", "replace")[-OUTPUT_KEEP:],
    }


def _killpg(pid):
    try:
        os.killpg(pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


def run_bandit(files, ignore_nosec):
    # A fresh copy with explicit file paths, so no config file in /repo is picked up.
    workdir = tempfile.mkdtemp(prefix="bandit-")
    for rel in files:
        dest = os.path.join(workdir, rel)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        shutil.copyfile(os.path.join(REPO, rel), dest)
    cmd = ["bandit", "-q", "-f", "json", "--severity-level", "medium"]
    if ignore_nosec:
        cmd.append("--ignore-nosec")
    proc = subprocess.run(
        cmd + list(files), cwd=workdir, capture_output=True, text=True, timeout=BANDIT_TIMEOUT,
    )
    shutil.rmtree(workdir, ignore_errors=True)
    try:
        report = json.loads(proc.stdout)
        keys = ("filename", "line_number", "test_id", "issue_severity", "issue_confidence", "issue_text")
        results = [{key: r.get(key) for key in keys} for r in report.get("results", [])]
        errors, json_ok = report.get("errors", []), True
    except (json.JSONDecodeError, AttributeError):
        results, errors, json_ok = [], [], False
    return {
        "exit_code": proc.returncode,
        "json_ok": json_ok,
        "results": results,
        "errors": errors,
        # A plugin crash is printed here and never reaches the errors list.
        "internal_error": "Bandit internal error" in proc.stderr,
        "stderr": proc.stderr[-OUTPUT_KEEP:],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pytest", action="store_true", help="run the suite in /repo/tests")
    parser.add_argument("--bandit", nargs="+", default=[], metavar="FILE", help="files under /repo to scan")
    parser.add_argument("--ignore-nosec", action="store_true")
    args = parser.parse_args()
    if not args.pytest and not args.bandit:
        parser.error("nothing to run")
    report = {
        "pytest": run_pytest() if args.pytest else None,
        "bandit": run_bandit(args.bandit, args.ignore_nosec) if args.bandit else None,
    }
    json.dump(report, sys.stdout)


if __name__ == "__main__":
    main()
