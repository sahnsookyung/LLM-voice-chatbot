#!/usr/bin/env python3
"""Install the verified official CPU release in the project runtime."""

import hashlib
import tarfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
URL = "https://github.com/NVIDIA/NeMo-Speech.cpp/releases/download/v0.2.0/nemo-speech-0.2.0-macos-aarch64-cpu.tar.gz"
SHA256 = "6a91355e15d22db6a319144d389d2e1b02f233c1c6808c0da85b6d3283e52296"


def main():
    archive = ROOT / ".runtime/downloads/nemo.tar.gz"
    archive.parent.mkdir(parents=True, exist_ok=True)
    if not archive.exists():
        urllib.request.urlretrieve(URL, archive)
    if hashlib.sha256(archive.read_bytes()).hexdigest() != SHA256:
        raise SystemExit(
            "NeMo archive checksum mismatch; remove .runtime/downloads/nemo.tar.gz and retry."
        )
    with tarfile.open(archive) as bundle:
        for member in bundle.getmembers():
            member.name = "/".join(member.name.split("/")[1:])
            if member.name:
                bundle.extract(member, ROOT / ".runtime/nemo", filter="data")
    print("NeMo 0.2.0 CPU runtime verified and installed.")


if __name__ == "__main__":
    main()
