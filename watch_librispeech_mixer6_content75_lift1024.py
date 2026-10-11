#!/usr/bin/env python3
"""Durably watch the BigVGAN-matched mixed-corpus Vocos launch through completion."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


POLL_SECONDS = 300
SUCCESS_STATES = {"COMPLETED"}
FAILURE_STATES = {
    "BOOT_FAIL", "CANCELLED", "DEADLINE", "FAILED", "NODE_FAIL", "OUT_OF_MEMORY", "PREEMPTED", "REVOKED", "TIMEOUT",
}
PROGRESS_MARKER = re.compile(r"Epoch [0-9]+:.*(loss=|[0-9]+/[0-9]+)")
MAX_RECOVERY_ATTEMPTS = 2
SLURM_BIN = Path("/opt/mprov/cloack/slurm/current/bin")
CODEX_BIN = Path("/home/jhu/xli257/.nvm/versions/node/v24.21.0/bin/codex")


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-file", type=Path, required=True)
    parser.add_argument("--project-dir", type=Path, required=True)
    parser.add_argument("--preflight", required=True)
    parser.add_argument("--probe", required=True)
    parser.add_argument("--training", required=True)
    parser.add_argument("--walltime-seconds", type=int, required=True)
    parser.add_argument("--successor-lead-seconds", type=int, default=900)
    return parser.parse_args()


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def command(args: list[str]) -> str:
    return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT).strip()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def event(state: dict[str, Any], kind: str, detail: str) -> None:
    state.setdefault("events", []).append({"timestamp_utc": timestamp(), "kind": kind, "detail": detail})
    state["events"] = state["events"][-200:]


def initial_state(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "version": 1,
        "scope": "completion",
        "created_at_utc": timestamp(),
        "project_dir": str(args.project_dir),
        "state_file": str(args.state_file),
        "poll_seconds": POLL_SECONDS,
        "healthy_progress_marker": PROGRESS_MARKER.pattern,
        "stages": {
            "preflight": {
                "job_id": str(args.preflight),
                "dependency": "initial mixed-corpus preflight",
                "log": str(args.project_dir / f"slurm-vocos-libri-mixer-lift-preflight-{args.preflight}.log"),
            },
            "probe": {
                "job_id": str(args.probe),
                "dependency": f"afterok:{args.preflight}",
                "log": str(args.project_dir / f"slurm-vocos-libri-mixer-lift-probe-{args.probe}.log"),
            },
            "training": {
                "job_id": str(args.training),
                "dependency": f"afterok:{args.probe}",
                "log": str(args.project_dir / f"slurm-vocos-libri-mixer-lift-{args.training}.log"),
            },
        },
        "recovery_attempts": {},
        "events": [],
    }


def job_status(job_id: str) -> dict[str, str]:
    try:
        output = command([str(SLURM_BIN / "squeue"), "--noheader", "--jobs", job_id, "--format=%T|%R"])
    except subprocess.CalledProcessError:
        output = ""
    if output:
        state, _, reason = output.partition("|")
        return {"state": state.strip().upper(), "reason": reason.strip()}
    try:
        output = command([
            str(SLURM_BIN / "sacct"), "--noheader", "--parsable2", "--allocations", "--jobs", job_id,
            "--format=JobIDRaw,State,ExitCode",
        ])
    except subprocess.CalledProcessError as exc:
        return {"state": "UNKNOWN", "reason": f"sacct error: {exc.output.strip()}"}
    for line in output.splitlines():
        fields = line.split("|")
        if len(fields) >= 3 and fields[0] == str(job_id):
            return {"state": fields[1].split()[0].upper(), "reason": f"exit_code={fields[2]}"}
    return {"state": "UNKNOWN", "reason": "not yet visible in squeue or sacct"}


def training_has_progress(stage: dict[str, Any]) -> bool:
    log = Path(stage["log"])
    if not log.is_file():
        return False
    return PROGRESS_MARKER.search(log.read_text(errors="replace")) is not None


def failure_signature(stage_name: str, status: dict[str, str]) -> str:
    return f"{stage_name}:{status['state']}:{status['reason'][:240]}"


def recovery(args: argparse.Namespace, state: dict[str, Any], stage_name: str, status: dict[str, str]) -> bool:
    signature = failure_signature(stage_name, status)
    attempts = int(state.setdefault("recovery_attempts", {}).get(signature, 0))
    if attempts >= MAX_RECOVERY_ATTEMPTS:
        state["blocker"] = f"Recovery limit reached for {signature!r}; inspect its failed log and watcher state."
        event(state, "blocked", state["blocker"])
        return False
    state["recovery_attempts"][signature] = attempts + 1
    recovery_dir = args.state_file.parent / "recovery"
    recovery_dir.mkdir(parents=True, exist_ok=True)
    transcript = recovery_dir / f"{stage_name}-{int(time.time())}-attempt{attempts + 1}.txt"
    prompt = f"""You are the narrowly scoped recovery agent for exactly one Vocos Slurm family. Do not use web search.

Workspace: {args.project_dir}
Durable watcher state: {args.state_file}
Failed stage: {stage_name}
Failed status: {json.dumps(status, sort_keys=True)}
The family trains causal Vocos on the BigVGAN-matched LibriSpeech + Mixer6 75-D causal-content / corpus-specific 75x1024-lift setup. Its launcher is slurm/submit_vocos_librispeech_mixer6_content75_lift1024.sh.

Follow /home/jhu/xli257/ARTS/workflows/agentic_watcher_job_requirements.md and /home/jhu/xli257/ARTS/workflows/slurm_job_requirements.md. Inspect only this family’s state, scripts, and combined logs. Preserve every healthy running job and never cancel, requeue, or modify one. Make the minimal necessary source or future-launcher repair, validate it, and submit only the failed stage and its dependent replacement stages. Preserve the data setup, configuration, artifact root, and W&B lineage unless the failure itself requires a documented change. Atomically update the existing JSON state’s stages.preflight/stages.probe/stages.training job_id, log, and dependency fields for replacements; retain events; record the concrete cause and recovery event. If no safe repair is possible, atomically set a concrete blocker. Keep durable artifacts in scratch. Do not commit or push.
"""
    result = subprocess.run(
        [
            str(CODEX_BIN), "exec", "--dangerously-bypass-approvals-and-sandbox", "-C", str(args.project_dir),
            "--add-dir", str(args.state_file.parent.parent), "--add-dir", "/home/jhu/xli257/ARTS/workflows",
            "-o", str(transcript), prompt,
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=3600,
    )
    transcript.with_suffix(".stdout.txt").write_text(result.stdout, encoding="utf-8")
    event(state, "recovery", f"stage={stage_name} attempt={attempts + 1} returncode={result.returncode} transcript={transcript}")
    return result.returncode == 0


def submit_successor(args: argparse.Namespace, state: dict[str, Any]) -> None:
    job_id = os.environ.get("SLURM_JOB_ID")
    if not job_id:
        raise RuntimeError("SLURM_JOB_ID is required to submit a watcher successor")
    account = os.environ.get("SLURM_JOB_ACCOUNT", "nandrew9_arts")
    script = args.project_dir / "slurm/watch_vocos_librispeech_mixer6_content75_lift1024.sbatch"
    output = command([
        str(SLURM_BIN / "sbatch"), "--parsable", f"--account={account}", "--qos=jhu",
        f"--dependency=afterany:{job_id}", f"--export=ALL,WATCHER_STATE_FILE={args.state_file}", str(script),
    ])
    successor = output.splitlines()[-1].split(";", 1)[0]
    state["successor_job_id"] = successor
    event(state, "successor_submitted", f"job_id={successor} dependency=afterany:{job_id}")


def main() -> int:
    args = arguments()
    args.project_dir = args.project_dir.resolve()
    args.state_file = args.state_file.resolve()
    if not CODEX_BIN.is_file():
        raise RuntimeError(f"Approved Codex CLI is unavailable: {CODEX_BIN}")
    lock_path = args.state_file.with_suffix(args.state_file.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with lock_path.open("w") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(f"Another watcher holds {lock_path}; exiting.", flush=True)
            return 0
        state = json.loads(args.state_file.read_text()) if args.state_file.is_file() else initial_state(args)
        state["active_watcher_job_id"] = os.environ.get("SLURM_JOB_ID", "manual")
        event(state, "watcher_started", f"job_id={state['active_watcher_job_id']}")
        atomic_json(args.state_file, state)
        while True:
            state = json.loads(args.state_file.read_text())
            if state.get("blocker"):
                event(state, "watcher_stopped", f"blocked: {state['blocker']}")
                atomic_json(args.state_file, state)
                return 1
            stages = state["stages"]
            statuses = {name: job_status(stage["job_id"]) for name, stage in stages.items()}
            for name, status in statuses.items():
                if stages[name].get("last_status") != status:
                    stages[name]["last_status"] = status
                    event(state, "status", f"{name} job={stages[name]['job_id']} state={status['state']} reason={status['reason']}")
            progress = training_has_progress(stages["training"])
            if progress and not stages["training"].get("healthy_progress"):
                stages["training"]["healthy_progress"] = True
                event(state, "healthy_progress", f"training job {stages['training']['job_id']} emitted a completed optimizer step")
            failed: tuple[str, dict[str, str]] | None = None
            for name, status in statuses.items():
                if status["state"] in FAILURE_STATES or "DEPENDENCYNEVERSATISFIED" in status["reason"].upper():
                    failed = (name, status)
                    break
                if name == "training" and status["state"] in SUCCESS_STATES and not progress:
                    failed = (name, {"state": "COMPLETED_WITHOUT_PROGRESS", "reason": "training exited without an optimizer-step marker"})
                    break
            if failed is not None:
                name, status = failed
                event(state, "failure_detected", f"{name}: {status}")
                atomic_json(args.state_file, state)
                if not recovery(args, state, name, status):
                    atomic_json(args.state_file, state)
                    return 1
                time.sleep(30)
                continue
            if all(status["state"] in SUCCESS_STATES for status in statuses.values()) and progress:
                event(state, "completed", "all monitored stages completed after healthy training progress")
                atomic_json(args.state_file, state)
                print("All monitored stages completed successfully.", flush=True)
                return 0
            if args.walltime_seconds - int(time.monotonic() - started) <= args.successor_lead_seconds:
                submit_successor(args, state)
                atomic_json(args.state_file, state)
                return 0
            atomic_json(args.state_file, state)
            print("Watcher poll: " + " ".join(f"{name}={status['state']}" for name, status in statuses.items()), flush=True)
            time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    sys.exit(main())
