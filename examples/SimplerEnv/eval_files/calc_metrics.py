#!/usr/bin/env python3
"""Summarize success rates from ResVLA eval logs.

Usage:
  python examples/SimplerEnv/eval_files/auto_eval_scripts/calc_metrics.py \
    --eval_dir /path/to/bridge_eval_YYYYMMDD_HHMMSS

The script expects an eval directory that contains `eval_*.log` files.
It will ignore `server_*.log` files and subdirectories.

Bridge:
  - Typically 4 tests, one log per test: `eval_env_<TaskName>_gpuX.log`
Fractal:
  - Many tests, potentially multiple log files per test (e.g., different b/gpu).
    - Metrics are merged by the common prefix before "__".
        For example, "drawer_va__base__CloseBottom..." and "drawer_va__light_darker__OpenTop..."
        are considered the same merged test "drawer_va".

It parses success rate from the *last non-empty line* of each log.
Common formats supported:
  - "Average success 0.0833"
  - "success rate: 0.0833" / "success_rate=0.0833"

If parsing fails, the error is recorded and reported at the end.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional


_SUCCESS_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"Average\s+success\s+([0-9]*\.?[0-9]+(?:[eE][-+]?\d+)?)"),
    re.compile(r"success\s*rate\s*[:=]\s*([0-9]*\.?[0-9]+(?:[eE][-+]?\d+)?)", re.IGNORECASE),
    re.compile(r"success_rate\s*[:=]\s*([0-9]*\.?[0-9]+(?:[eE][-+]?\d+)?)", re.IGNORECASE),
]


@dataclass(frozen=True)
class ParsedLog:
    log_path: str
    test_id: str
    variant: str
    success: float


def _infer_benchmark(eval_dir: Path) -> Optional[str]:
    name = eval_dir.name.lower()
    if name.startswith("bridge"):
        return "bridge"
    if name.startswith("fractal"):
        return "fractal"

    # Fallback based on files present
    for p in eval_dir.iterdir():
        if p.is_file() and p.name.startswith("eval_env_") and p.suffix == ".log":
            return "bridge"
        if p.is_file() and p.name.startswith("eval_") and p.suffix == ".log":
            return "fractal"
    return None


def _iter_eval_logs(eval_dir: Path) -> list[Path]:
    if not eval_dir.exists() or not eval_dir.is_dir():
        return []

    logs: list[Path] = []
    for p in sorted(eval_dir.iterdir()):
        if not p.is_file():
            continue
        if not (p.name.startswith("eval_") and p.suffix == ".log"):
            continue
        if p.name.startswith("server_"):
            continue
        logs.append(p)
    return logs


def _tail_nonempty_lines(path: Path, max_lines: int = 80) -> list[str]:
    """Return up to the last `max_lines` non-empty lines.

    This avoids loading very large files fully.
    """

    try:
        with path.open("rb") as f:
            f.seek(0, os.SEEK_END)
            file_size = f.tell()
            if file_size == 0:
                return []

            # Read chunks from end until we have enough lines.
            chunk_size = 8192
            data = b""
            offset = 0
            while True:
                offset = min(file_size, offset + chunk_size)
                f.seek(file_size - offset)
                data = f.read(offset) + data
                lines = data.splitlines()
                if len(lines) >= max_lines + 5 or offset >= file_size:
                    break

        decoded: list[str] = []
        for raw in lines[-(max_lines + 5) :]:
            try:
                s = raw.decode("utf-8", errors="replace").strip()
            except Exception:
                continue
            if s:
                decoded.append(s)
        return decoded[-max_lines:]
    except Exception:
        return []


def _parse_success_from_lines(lines: list[str]) -> Optional[float]:
    # Search from the end (most likely summary near the bottom)
    for line in reversed(lines):
        for pat in _SUCCESS_PATTERNS:
            m = pat.search(line)
            if m:
                try:
                    return float(m.group(1))
                except ValueError:
                    continue
    return None


def _extract_variant(text: str) -> str:
    t = text.lower()
    if "_va__" in t or "__va__" in t or re.search(r"(^|_)va(_|$)", t):
        return "va"
    if "_vm__" in t or "__vm__" in t or re.search(r"(^|_)vm(_|$)", t):
        return "vm"
    return "base"


def _bridge_test_id_from_filename(filename: str) -> Optional[str]:
    m = re.match(r"^eval_env_(.+?)_gpu\d+\.log$", filename)
    if not m:
        return None
    return m.group(1)


def _fractal_test_id_from_filename(filename: str) -> str:
    # Strip prefix/suffix
    s = filename
    if s.endswith(".log"):
        s = s[: -len(".log")]
    if s.startswith("eval_"):
        s = s[len("eval_") :]

    # Remove common trailing shards like __b0_gpu2, _gpu2, __gpu2
    s = re.sub(r"__b\d+_gpu\d+$", "", s)
    s = re.sub(r"__gpu\d+$", "", s)
    s = re.sub(r"_gpu\d+$", "", s)
    return s


def parse_eval_dir(eval_dir: Path) -> tuple[str, list[ParsedLog], list[str]]:
    errors: list[str] = []

    benchmark = _infer_benchmark(eval_dir)
    if benchmark is None:
        return (
            "unknown",
            [],
            [
                f"Could not infer the benchmark from directory name: {eval_dir}",
                "Please pass a directory like bridge_eval_* or fractal_eval_*.",
            ],
        )

    log_files = _iter_eval_logs(eval_dir)
    if not log_files:
        return (
            benchmark,
            [],
            [f"No eval_*.log files were found under: {eval_dir}", "(Ignored server_*.log files and subdirectories)"],
        )

    parsed: list[ParsedLog] = []
    for log_path in log_files:
        lines = _tail_nonempty_lines(log_path, max_lines=80)
        if not lines:
            errors.append(f"Empty file or unreadable log: {log_path}")
            continue

        success = _parse_success_from_lines(lines)
        if success is None:
            errors.append(f"Could not parse success rate (no match in the last 80 lines): {log_path}")
            continue

        if benchmark == "bridge":
            test_id = _bridge_test_id_from_filename(log_path.name)
            if test_id is None:
                errors.append(
                    f"Unexpected bridge log naming pattern (expected eval_env_<task>_gpuX.log): {log_path.name}"
                )
                test_id = log_path.stem
        else:
            test_id = _fractal_test_id_from_filename(log_path.name)

        variant = _extract_variant(test_id)
        parsed.append(
            ParsedLog(
                log_path=str(log_path),
                test_id=test_id,
                variant=variant,
                success=float(success),
            )
        )

    return benchmark, parsed, errors


def _mean(values: list[float]) -> Optional[float]:
    if not values:
        return None
    return sum(values) / len(values)


def _fractal_merge_key(test_id: str) -> str:
    # e.g. drawer_va__base__CloseBottom... -> drawer_va
    return test_id.split("__", 1)[0]


def summarize(parsed_logs: list[ParsedLog], benchmark: str) -> dict:
    # 1) First, group by the original test_id to merge multiple log shards for the same test.
    by_test_id: dict[str, dict] = {}
    for r in parsed_logs:
        entry = by_test_id.setdefault(
            r.test_id,
            {
                "test_id": r.test_id,
                "variant": r.variant,
                "successes": [],
                "log_paths": [],
            },
        )
        entry["successes"].append(r.success)
        entry["log_paths"].append(r.log_path)

    per_test: list[dict] = []
    for test_id, entry in sorted(by_test_id.items(), key=lambda kv: kv[0]):
        successes = entry["successes"]
        per_test.append(
            {
                "test_id": test_id,
                "variant": entry["variant"],
                "num_logs": len(successes),
                "success": _mean(successes),
                "log_paths": entry["log_paths"],
            }
        )

    # 2) Then, merge tests for fractal by the common prefix.
    if benchmark == "fractal":
        merged: dict[str, dict] = {}
        for t in per_test:
            merged_id = _fractal_merge_key(str(t["test_id"]))
            m = merged.setdefault(
                merged_id,
                {
                    "test_id": merged_id,
                    "variant": _extract_variant(merged_id),
                    "successes": [],
                    "log_paths": [],
                    "num_logs": 0,
                    "num_subtests": 0,
                },
            )
            m["num_subtests"] += 1
            m["num_logs"] += int(t.get("num_logs", 0) or 0)
            m["log_paths"].extend(list(t.get("log_paths", [])))
            if t.get("success") is not None:
                m["successes"].append(float(t["success"]))

        tests: list[dict] = []
        for merged_id, m in sorted(merged.items(), key=lambda kv: kv[0]):
            tests.append(
                {
                    "test_id": merged_id,
                    "variant": m["variant"],
                    "num_logs": m["num_logs"],
                    "num_subtests": m["num_subtests"],
                    # Unweighted mean over subtests (each subtest already merged over its log shards)
                    "success": _mean(m["successes"]),
                    "log_paths": m["log_paths"],
                }
            )
    else:
        tests = per_test

    overall = _mean([t["success"] for t in tests if t.get("success") is not None])

    by_variant: dict[str, list[float]] = {}
    for t in tests:
        if t.get("success") is None:
            continue
        by_variant.setdefault(str(t["variant"]), []).append(float(t["success"]))

    variant_summary = {
        k: {
            "variant": k,
            "num_tests": len(v),
            "avg_success": _mean(v),
        }
        for k, v in sorted(by_variant.items(), key=lambda kv: kv[0])
    }

    return {
        "num_logs": len(parsed_logs),
        "num_tests": len(tests),
        "overall_avg_success": overall,
        "tests": tests,
        "variants": variant_summary,
    }


def _print_text(benchmark: str, eval_dir: Path, summary: dict, errors: list[str]) -> None:
    print(f"Benchmark: {benchmark}")
    print(f"Eval dir: {eval_dir}")
    print(f"Found logs: {summary.get('num_logs', 0)}")
    print(f"Tests: {summary.get('num_tests', 0)}")
    print("-")

    if errors:
        print("-")
        print("Problems:")
        for e in errors:
            print(f"  - {e}")

    tests = summary.get("tests", [])
    if tests:
        print("Merged success:")
        for t in tests:
            succ = t.get("success")
            succ_str = "nan" if succ is None else f"{succ:.6f}".rstrip("0").rstrip(".")
            extra = ""
            if benchmark == "fractal":
                extra = f"\tsubtests={t.get('num_subtests')}"
            print(
                f"  {t.get('test_id')}\tvariant={t.get('variant')}\tlogs={t.get('num_logs')}{extra}\tsuccess={succ_str}"
            )

    overall = summary.get("overall_avg_success")
    if overall is not None:
        print("-")
        print(f"Overall avg (mean over tests): {overall:.6f}".rstrip("0").rstrip("."))

    variants = summary.get("variants", {})
    if variants:
        print("-")
        print("By variant (mean over tests):")
        for k, v in variants.items():
            avg = v.get("avg_success")
            avg_str = "nan" if avg is None else f"{avg:.6f}".rstrip("0").rstrip(".")
            print(f"  {k}\ttests={v.get('num_tests')}\tavg_success={avg_str}")

def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize eval success rates from eval_*.log")
    parser.add_argument(
        "--eval_dir",
        type=str,
        required=True,
        help="Path to the eval log directory (e.g., .../bridge_eval_*/ or .../fractal_eval_*/)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON summary to stdout.",
    )

    args = parser.parse_args()
    eval_dir = Path(args.eval_dir).expanduser().resolve()

    benchmark, parsed_logs, errors = parse_eval_dir(eval_dir)
    summary = summarize(parsed_logs, benchmark)

    if args.json:
        payload = {
            "benchmark": benchmark,
            "eval_dir": str(eval_dir),
            "summary": summary,
            "errors": errors,
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        _print_text(benchmark, eval_dir, summary, errors)


if __name__ == "__main__":
    main()
