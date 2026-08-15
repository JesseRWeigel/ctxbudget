"""The old batch uploader. Superseded, still on disk, and nothing in this project imports it.

Present so the fixture contains a file that is neither small nor referenced nor duplicated,
which is the case a naive ranking gets right by accident and a good one gets right on purpose.
"""

from __future__ import annotations

import base64
import hashlib
import os
import shutil
import tempfile
from dataclasses import dataclass


@dataclass
class UploadSlice:
    ordinal: int
    payload: bytes
    checksum: str

    @classmethod
    def of(cls, ordinal: int, payload: bytes) -> "UploadSlice":
        return cls(ordinal, payload, hashlib.sha256(payload).hexdigest())


class LegacyUploader:
    def __init__(self, staging_dir: str | None = None, slice_bytes: int = 4 * 1024 * 1024):
        self.staging_dir = staging_dir or tempfile.mkdtemp(prefix="legacy-upload-")
        self.slice_bytes = slice_bytes
        self.manifest: list[UploadSlice] = []

    def cut(self, blob: bytes) -> list[UploadSlice]:
        pieces = []
        for ordinal, start in enumerate(range(0, len(blob), self.slice_bytes)):
            pieces.append(UploadSlice.of(ordinal, blob[start:start + self.slice_bytes]))
        self.manifest = pieces
        return pieces

    def stage(self) -> list[str]:
        written = []
        for piece in self.manifest:
            target = os.path.join(self.staging_dir, f"part-{piece.ordinal:05d}.bin")
            with open(target, "wb") as handle:
                handle.write(piece.payload)
            written.append(target)
        return written

    def manifest_blob(self) -> bytes:
        rows = [f"{piece.ordinal}\t{piece.checksum}\t{len(piece.payload)}"
                for piece in self.manifest]
        return base64.b64encode("\n".join(rows).encode("utf-8"))

    def verify_staged(self) -> bool:
        for piece in self.manifest:
            target = os.path.join(self.staging_dir, f"part-{piece.ordinal:05d}.bin")
            if not os.path.exists(target):
                return False
            with open(target, "rb") as handle:
                if hashlib.sha256(handle.read()).hexdigest() != piece.checksum:
                    return False
        return True

    def discard(self) -> None:
        shutil.rmtree(self.staging_dir, ignore_errors=True)
        self.manifest = []

    def summary(self) -> str:
        total = sum(len(piece.payload) for piece in self.manifest)
        return (f"{len(self.manifest)} slices, {total} bytes staged under "
                f"{os.path.basename(self.staging_dir)}")
