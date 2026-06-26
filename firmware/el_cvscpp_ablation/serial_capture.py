#!/usr/bin/env python3
"""Capture EL_C_vsCpp benchmark UART output until DONE or timeout."""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", required=True)
    parser.add_argument("--baudrate", type=int, default=12000000)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--done-prefix", default="DONE test=EL_C_vsCpp")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        import serial
    except ImportError:
        print("pyserial is required for serial capture", file=sys.stderr)
        return 3

    args.output.parent.mkdir(parents=True, exist_ok=True)
    done_line: str | None = None
    status = None
    start = time.monotonic()
    buffer = bytearray()

    with args.output.open("w", encoding="utf-8") as log:
        with serial.Serial(args.port, args.baudrate, timeout=0.2) as ser:
            ser.reset_input_buffer()
            while time.monotonic() - start < args.timeout:
                chunk = ser.read(256)
                if not chunk:
                    continue
                buffer.extend(chunk)
                while b"\n" in buffer:
                    raw_line, _, rest = buffer.partition(b"\n")
                    buffer = bytearray(rest)
                    line = raw_line.decode("utf-8", errors="replace").rstrip("\r")
                    print(line, flush=True)
                    log.write(line + "\n")
                    log.flush()
                    if line.startswith(args.done_prefix):
                        done_line = line
                        match = re.search(r"\bstatus=(-?\d+)\b", line)
                        if match:
                            status = int(match.group(1))
                        break
                if done_line is not None:
                    break

    if done_line is None:
        print(f"timeout waiting for {args.done_prefix}", file=sys.stderr)
        return 124
    if status is None:
        print(f"DONE line missing status: {done_line}", file=sys.stderr)
        return 4
    return 0 if status == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())

