#!/usr/bin/env python3
"""Read-only readiness report; never manufactures live acceptance evidence."""

import argparse
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from companion.codex_compat import probe


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = await probe(ROOT / ".runtime/codex/acceptance.json")
    content = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content)
    print(content)
    return 0 if result["available"] else 2


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
