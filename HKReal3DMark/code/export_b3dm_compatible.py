#!/usr/bin/env python3
"""Export a B3DM's embedded GLB as glTF with ordinary PNG textures.

Open3Dhk tiles use KTX2 textures that current MeshLab builds may not decode.
This utility keeps the original geometry/materials, decodes each embedded KTX2
image with Basis Universal, and writes a portable .gltf + .bin + PNG directory.
"""

from __future__ import annotations

import argparse
import shutil
import struct
import subprocess
import tempfile
from pathlib import Path

from pygltflib import GLTF2


def embedded_glb(data: bytes) -> bytes:
    if len(data) < 28:
        raise ValueError("B3DM header is truncated")
    magic, version, byte_length, ft_json, ft_bin, bt_json, bt_bin = struct.unpack_from(
        "<4s6I", data, 0
    )
    if magic != b"b3dm" or version != 1 or byte_length > len(data):
        raise ValueError("Unsupported or incomplete B3DM")
    offset = 28 + ft_json + ft_bin + bt_json + bt_bin
    glb = data[offset:byte_length]
    if glb[:4] != b"glTF":
        raise ValueError("B3DM does not contain a GLB payload")
    return glb


def decode_ktx2(payload: bytes, target: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="hkreal3d_ktx2_") as tmp_name:
        tmp = Path(tmp_name)
        source = tmp / "texture.ktx2"
        decoded = tmp / "decoded"
        decoded.mkdir()
        source.write_bytes(payload)
        subprocess.run(
            [
                "basisu",
                str(source),
                "-unpack",
                "-no_ktx",
                "-etc1_only",
                "-output_path",
                str(decoded),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
        )
        candidates = sorted(decoded.glob("*.png"))
        if not candidates:
            raise RuntimeError("Basis Universal produced no PNG")
        shutil.copy2(candidates[0], target)


def export(source: Path, output: Path) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    temp_glb = output / "embedded_original.glb"
    temp_glb.write_bytes(embedded_glb(source.read_bytes()))
    model = GLTF2().load(str(temp_glb))
    blob = model.binary_blob()
    if blob is None:
        raise ValueError("Embedded GLB has no binary buffer")

    for index, image in enumerate(model.images):
        if image.bufferView is None:
            continue
        view = model.bufferViews[image.bufferView]
        start = view.byteOffset or 0
        payload = blob[start : start + view.byteLength]
        suffix = ".png" if image.mimeType == "image/ktx2" else ".bin"
        name = f"texture_{index:02d}{suffix}"
        target = output / name
        if image.mimeType == "image/ktx2":
            decode_ktx2(payload, target)
            image.mimeType = "image/png"
        else:
            target.write_bytes(payload)
        image.uri = name
        image.bufferView = None

    target_model = output / "model.gltf"
    model.save(str(target_model))
    temp_glb.unlink()
    return target_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = export(args.source, args.output)
    print(result)


if __name__ == "__main__":
    main()
