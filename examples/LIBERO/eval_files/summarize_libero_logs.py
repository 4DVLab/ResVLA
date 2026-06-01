#!/usr/bin/env python3
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional


STEP_RE = re.compile(r"steps_(\d+)_pytorch_model\.pt")
TOTAL_RATE_RE = re.compile(r"Total success rate:\s*([0-9]*\.?[0-9]+)")
TOTAL_EP_RE = re.compile(r"Total episodes:\s*(\d+)")
SUCCESS_LINE_RE = re.compile(r"# successes:\s*(\d+)\s*\(([0-9]*\.?[0-9]+)%\)")
EPISODE_INLINE_RE = re.compile(r"# episodes completed so far:\s*(\d+)\b")
PURE_INT_LINE_RE = re.compile(r"^\s*(\d+)\s*$")
PORT_RE = re.compile(r'"port"\s*:\s*(\d+)')
WS_PORT_RE = re.compile(r"ws://[^\s:]+:(\d+)")


def extract_episode_progress_counts(text: str) -> list[int]:
    episodes = []
    lines = text.splitlines()

    for i, line in enumerate(lines):
        if "# episodes completed so far:" not in line:
            continue

        # Most lines contain the number inline, e.g. "... so far: 7 eval_libero.py:230".
        inline_match = EPISODE_INLINE_RE.search(line)
        if inline_match:
            episodes.append(int(inline_match.group(1)))
            continue

        # When rich logging wraps, the count is printed on the next line by itself.
        for next_line in lines[i + 1 : i + 4]:
            wrapped_match = PURE_INT_LINE_RE.match(next_line)
            if wrapped_match:
                episodes.append(int(wrapped_match.group(1)))
                break

    return episodes


def parse_log_file(path: Path) -> Optional[Dict]:
    text = path.read_text(encoding="utf-8", errors="ignore")

    step_match = STEP_RE.search(path.name)
    if not step_match:
        return None
    step = int(step_match.group(1))

    suite = path.parent.name

    total_rate_matches = TOTAL_RATE_RE.findall(text)
    total_ep_matches = TOTAL_EP_RE.findall(text)
    success_matches = SUCCESS_LINE_RE.findall(text)
    ep_progress_matches = extract_episode_progress_counts(text)
    port_matches = PORT_RE.findall(text)
    ws_port_matches = WS_PORT_RE.findall(text)

    rate = float(total_rate_matches[-1]) if total_rate_matches else None
    episodes = int(total_ep_matches[-1]) if total_ep_matches else None

    successes = None
    success_pct = None
    if success_matches:
        successes = int(success_matches[-1][0])
        success_pct = float(success_matches[-1][1])

    if episodes is None and ep_progress_matches:
        episodes = int(ep_progress_matches[-1])

    if rate is None:
        if successes is not None and episodes:
            rate = successes / episodes
        elif success_pct is not None:
            rate = success_pct / 100.0

    if successes is None and rate is not None and episodes:
        successes = int(round(rate * episodes))

    port = None
    if port_matches:
        port = int(port_matches[0])
    elif ws_port_matches:
        port = int(ws_port_matches[0])

    if rate is None:
        return None

    return {
        "file": str(path),
        "step": step,
        "suite": suite,
        "port": port,
        "success_rate": rate,
        "successes": successes,
        "episodes": episodes,
    }


def summarize(logs_dir: Path) -> Dict:
    records = []
    for p in sorted(logs_dir.rglob("*.log")):
        rec = parse_log_file(p)
        if rec is not None:
            records.append(rec)

    by_step = defaultdict(list)
    for r in records:
        by_step[r["step"]].append(r)

    summary = {}
    for step in sorted(by_step.keys()):
        rows = by_step[step]
        suite_rates = [r["success_rate"] for r in rows if r["success_rate"] is not None]
        mean_rate = sum(suite_rates) / len(suite_rates) if suite_rates else None
        ports = sorted(set(r["port"] for r in rows if r["port"] is not None))

        valid_weighted = [
            r for r in rows if (r["successes"] is not None and r["episodes"] is not None and r["episodes"] > 0)
        ]
        if valid_weighted:
            total_succ = sum(r["successes"] for r in valid_weighted)
            total_eps = sum(r["episodes"] for r in valid_weighted)
            weighted_rate = total_succ / total_eps
        else:
            total_succ = None
            total_eps = None
            weighted_rate = None

        summary[step] = {
            "num_suites": len(rows),
            "ports": ports,
            "mean_suite_success_rate": mean_rate,
            "weighted_success_rate": weighted_rate,
            "total_successes": total_succ,
            "total_episodes": total_eps,
            "suite_details": sorted(rows, key=lambda x: x["suite"]),
        }

    return {
        "logs_dir": str(logs_dir),
        "num_files_parsed": len(records),
        "steps": summary,
    }


def print_table(result: Dict) -> None:
    steps = result["steps"]
    print(f"Logs dir: {result['logs_dir']}")
    print(f"Parsed files: {result['num_files_parsed']}")
    print("")
    print(
        f"{'Step':>8}  {'Suites':>6}  {'Ports':>18}  {'MeanSuiteRate(%)':>16}  {'WeightedRate(%)':>16}  {'Succ/Eps':>14}"
    )
    print("-" * 98)
    for step in sorted(steps.keys()):
        s = steps[step]
        mean_rate = s["mean_suite_success_rate"]
        weighted_rate = s["weighted_success_rate"]
        mean_str = f"{mean_rate * 100:.2f}" if mean_rate is not None else "N/A"
        weighted_str = f"{weighted_rate * 100:.2f}" if weighted_rate is not None else "N/A"
        port_str = ",".join(str(p) for p in s["ports"]) if s["ports"] else "N/A"
        if s["total_successes"] is not None and s["total_episodes"] is not None:
            se_str = f"{s['total_successes']}/{s['total_episodes']}"
        else:
            se_str = "N/A"
        print(
            f"{step:8d}  {s['num_suites']:6d}  {port_str:18}  {mean_str:16}  {weighted_str:16}  {se_str:14}"
        )

    print("")
    print("Per-step suite details:")
    for step in sorted(steps.keys()):
        print(f"  Step {step}:")
        for r in steps[step]["suite_details"]:
            rate = r["success_rate"] * 100 if r["success_rate"] is not None else None
            se = (
                f"{r['successes']}/{r['episodes']}"
                if (r["successes"] is not None and r["episodes"] is not None)
                else "N/A"
            )
            rate_str = f"{rate:.2f}%" if rate is not None else "N/A"
            port_str = str(r["port"]) if r["port"] is not None else "N/A"
            print(f"    - {r['suite']:14s}  port={port_str:>5s}  rate={rate_str:>8s}  succ/eps={se}")


def main():
    parser = argparse.ArgumentParser(description="Summarize LIBERO eval logs by checkpoint step.")
    parser.add_argument("logs_dir", type=str, help="Path to logs dir, e.g. .../results/.../logs")
    parser.add_argument(
        "--json_out",
        type=str,
        default="",
        help="Optional output json path. If set, full parsed summary will be saved.",
    )
    args = parser.parse_args()

    logs_dir = Path(args.logs_dir)
    if not logs_dir.exists():
        raise FileNotFoundError(f"Logs dir not found: {logs_dir}")

    result = summarize(logs_dir)
    print_table(result)

    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nSaved JSON summary to: {out}")


if __name__ == "__main__":
    main()
