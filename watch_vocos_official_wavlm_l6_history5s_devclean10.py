#!/usr/bin/env python3
"""Watch and, when needed, recover the official-WavLM dev-clean Vocos evaluation family."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


POLL_SECONDS = 300
MAX_RECOVERIES_PER_SIGNATURE = 2
SLURM_BIN = Path("/opt/mprov/cloack/slurm/current/bin")
CODEX_BIN = Path("/home/jhu/xli257/.nvm/versions/node/v24.21.0/bin/codex")


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-file", type=Path, required=True)
    parser.add_argument("--project-dir", type=Path, required=True)
    parser.add_argument("--family", default="vocos_official_wavlm_l6_history5s_devclean10")
    parser.add_argument("--probe", default="0")
    parser.add_argument("--synthesis", default="0")
    parser.add_argument("--utmos", default="0")
    parser.add_argument("--walltime-seconds", type=int, default=43_200)
    parser.add_argument("--successor-lead-seconds", type=int, default=900)
    return parser.parse_args()


def command(args: list[str]) -> str:
    return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT).strip()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def initial_state(args: argparse.Namespace) -> dict[str, Any]:
    workspace = args.project_dir.resolve()
    return {
        "family": args.family,
        "created_at": time.time(),
        "status": "watching",
        "stages": {
            "probe": {
                "job_id": str(args.probe),
                "log": str(workspace / f"slurm-vocos-l6-h5s-probe-{args.probe}.log"),
                "success_marker": "Probe passed.",
            },
            "synthesis": {
                "job_id": str(args.synthesis),
                "log": str(workspace / f"slurm-vocos-l6-h5s-devclean10-{args.synthesis}.log"),
                "success_marker": "Wrote 10 full-utterance WAVs under",
            },
            "utmos": {
                "job_id": str(args.utmos),
                "log": str(workspace / f"slurm-vocos-l6-h5s-utmos5-{args.utmos}.log"),
                "success_marker": "Mean UTMOS:",
            },
        },
        "events": [],
        "recoveries": {},
    }


def event(state: dict[str, Any], message: str) -> None:
    events = state.setdefault("events", [])
    events.append({"time": time.time(), "message": message})
    del events[:-100]
    print(message, flush=True)


def scheduler_status(job_id: str) -> dict[str, str]:
    if not job_id or job_id == "0":
        return {"state": "MISSING", "reason": "missing job ID"}
    live = subprocess.run(
        [str(SLURM_BIN / "squeue"), "-h", "-j", job_id, "-o", "%T|%R"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if live.returncode == 0 and live.stdout.strip():
        fields = live.stdout.strip().split("|", maxsplit=1)
        return {"state": fields[0].strip(), "reason": fields[1].strip() if len(fields) > 1 else ""}
    accounting = subprocess.run(
        [str(SLURM_BIN / "sacct"), "-X", "-n", "-P", "-j", job_id, "--format=State,ExitCode,Reason"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    for line in accounting.stdout.splitlines():
        fields = line.split("|")
        if len(fields) >= 2 and fields[0].strip():
            return {"state": fields[0].strip().split()[0], "exit_code": fields[1].strip(), "reason": fields[2].strip() if len(fields) > 2 else ""}
    return {"state": "UNKNOWN", "reason": (live.stderr or accounting.stderr).strip()}


def marker_found(stage: dict[str, Any]) -> bool:
    path = Path(stage["log"])
    if not path.is_file():
        return False
    return stage["success_marker"] in path.read_text(encoding="utf-8", errors="replace")


def failure_signature(stage_name: str, stage: dict[str, Any], status: dict[str, str]) -> str:
    log_tail = ""
    path = Path(stage["log"])
    if path.is_file():
        log_tail = path.read_text(encoding="utf-8", errors="replace")[-12_000:]
    payload = f"{stage_name}|{status}|{log_tail}".encode("utf-8", errors="replace")
    return hashlib.sha256(payload).hexdigest()[:16]


def terminal_failure(status: dict[str, str]) -> bool:
    state = status.get("state", "").upper()
    reason = status.get("reason", "").upper()
    if "DEPENDENCYNEVERSATISFIED" in reason or "DEPENDENCY NEVER SATISFIED" in reason:
        return True
    return any(token in state for token in ("FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL", "OUT_OF_MEMORY", "BOOT_FAIL", "DEADLINE", "MISSING"))


def recover(args: argparse.Namespace, state: dict[str, Any], stage_name: str, status: dict[str, str], signature: str) -> None:
    recovery_root = args.state_file.parent / "recovery"
    recovery_root.mkdir(parents=True, exist_ok=True)
    attempts = state.setdefault("recoveries", {}).get(signature, 0)
    if attempts >= MAX_RECOVERIES_PER_SIGNATURE:
        state["status"] = "blocked"
        state["blocker"] = f"Repeated failure {signature} for {stage_name} exhausted {MAX_RECOVERIES_PER_SIGNATURE} recovery attempts."
        event(state, state["blocker"])
        return
    state["recoveries"][signature] = attempts + 1
    transcript = recovery_root / f"{stage_name}-{signature}-attempt{attempts + 1}.txt"
    stage = state["stages"][stage_name]
    event(state, f"Starting recovery attempt {attempts + 1} for {stage_name}/{signature}; transcript={transcript}")
    # Persist the attempt before handing control to the recovery agent.  This
    # makes an in-progress repair visible and prevents a restarted watcher
    # from silently repeating it.
    atomic_json(args.state_file, state)
    prompt = f"""Repair only this Slurm evaluation family in {args.project_dir}.
Durable watcher state: {args.state_file}
Failed stage: {stage_name}; scheduler status: {json.dumps(status, sort_keys=True)}
Failed job ID: {stage['job_id']}; combined log: {stage['log']}

The family is full-utterance dev-clean evaluation of Vocos 1027263: official WavLM-Large L6 re-evaluated in 200 ms cores with 5 s left history and 80-sample tail, then UTXSS, source speaker 75x1024 lift, cached Vocos, and CPU UTMOSv2 five-TTA. Follow /home/jhu/xli257/ARTS/workflows/slurm_job_requirements.md and /home/jhu/xli257/ARTS/workflows/agentic_watcher_job_requirements.md. Do not browse the web. Preserve healthy running jobs. Inspect the failed log and relevant scripts, make the smallest source/future-launcher repair, validate it, cancel only never-runnable dependent jobs if necessary, and submit only the failed stage and successors. Keep the same evaluation configuration and durable root unless the failure demands otherwise. Atomically update this state file with replacement job IDs, logs, dependencies, exact failure cause, and recovery event. Do not commit or push."""
    result = subprocess.run(
        [str(CODEX_BIN), "exec", "--dangerously-bypass-approvals-and-sandbox", "-C", str(args.project_dir), "--add-dir", str(args.state_file.parent), "--add-dir", "/weka/home/jhu/xli257/ARTS/workflows", "-o", str(transcript), prompt],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    # The recovery agent owns replacement submission and atomically records the
    # new branch in durable state. Reload it before this watcher writes its own
    # event so the stale pre-recovery job IDs cannot overwrite that handoff.
    if args.state_file.is_file():
        persisted = json.loads(args.state_file.read_text(encoding="utf-8"))
        state.clear()
        state.update(persisted)
    event(state, f"Recovery attempt {attempts + 1} for {stage_name}/{signature} exited {result.returncode}; transcript={transcript}")
    atomic_json(args.state_file, state)


def submit_successor(args: argparse.Namespace, state: dict[str, Any]) -> None:
    job_id = os.environ.get("SLURM_JOB_ID")
    if not job_id or state.get("status") in {"complete", "blocked"}:
        return
    watcher = args.project_dir / "slurm/watch_vocos_official_wavlm_l6_history5s_devclean10.sbatch"
    account = os.environ.get("SLURM_JOB_ACCOUNT", "lgarci27_omnienc")
    submission = command([
        str(SLURM_BIN / "sbatch"), "--parsable", f"--account={account}", "--qos=jhu", f"--dependency=afterany:{job_id}",
        f"--export=ALL,WATCHER_STATE_FILE={args.state_file}", str(watcher),
    ])
    match = re.search(r"(?m)^(\d+(?:;[^\s]+)?)$", submission)
    if match is None:
        raise RuntimeError(f"Could not parse successor watcher job ID from sbatch output: {submission!r}")
    successor = match.group(1)
    state["successor_job_id"] = successor
    event(state, f"Submitted successor watcher {successor} after watcher {job_id}")


def main() -> None:
    args = arguments()
    args.project_dir = args.project_dir.resolve()
    args.state_file = args.state_file.resolve()
    lock_path = args.state_file.with_suffix(args.state_file.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(f"Another watcher holds {lock_path}; exiting.", flush=True)
            return
        if args.state_file.is_file():
            state = json.loads(args.state_file.read_text(encoding="utf-8"))
            event(state, f"Watcher {os.environ.get('SLURM_JOB_ID', 'local')} started; resuming durable state monitoring.")
            atomic_json(args.state_file, state)
        else:
            state = initial_state(args)
            event(state, "Watcher initialized")
            atomic_json(args.state_file, state)
        deadline = time.monotonic() + max(1, args.walltime_seconds - args.successor_lead_seconds)
        try:
            while time.monotonic() < deadline:
                if state.get("status") == "blocked":
                    atomic_json(args.state_file, state)
                    return
                all_complete = True
                recovered = False
                for stage_name, stage in state["stages"].items():
                    status = scheduler_status(str(stage["job_id"]))
                    previous = stage.get("scheduler")
                    stage["scheduler"] = status
                    if status != previous:
                        event(state, f"{stage_name} job {stage['job_id']} status={status}")
                    if status.get("state", "").upper() == "COMPLETED":
                        if marker_found(stage):
                            stage["complete"] = True
                        else:
                            recover(args, state, stage_name, status, failure_signature(stage_name, stage, status))
                            recovered = True
                        continue
                    all_complete = False
                    if terminal_failure(status):
                        recover(args, state, stage_name, status, failure_signature(stage_name, stage, status))
                        recovered = True
                if all_complete and all(stage.get("complete") for stage in state["stages"].values()):
                    state["status"] = "complete"
                    event(state, "All stages completed with their required healthy-progress markers.")
                    atomic_json(args.state_file, state)
                    return
                atomic_json(args.state_file, state)
                if recovered:
                    time.sleep(15)
                    if args.state_file.is_file():
                        state = json.loads(args.state_file.read_text(encoding="utf-8"))
                else:
                    time.sleep(POLL_SECONDS)
        finally:
            submit_successor(args, state)
            atomic_json(args.state_file, state)


if __name__ == "__main__":
    main()
