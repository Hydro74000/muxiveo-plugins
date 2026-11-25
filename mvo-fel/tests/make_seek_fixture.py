"""Mire Matroska aux images clés BL et EL décalées, et son extrait coupé en milieu de GOP EL.

BL 256×144 (IRAP toutes les 16 images), EL 128×72 (toutes les 10), GOP ouverts à trois B-frames
(CRA suivies de RASL dans les deux couches), contenu animé (testsrc2) : chaque rang a sa propre image
reconstruite. RPU FEL : celui de ``synthetic-fel.hevc`` (``assets/tests/fel_orig.bin`` de dovi_tool, L5
remis à zéro), répété. Extrait : à partir de la CRA BL de l'image 16 (EL : IRAP suivante à l'image 20).

Outils : FFmpeg/libx265, dovi_tool et le muxer Matroska natif de Muxiveo (``--muxiveo`` : racine du
dépôt) ; FFmpeg ne sait pas horodater un HEVC brut à images B, le muxer reprend les horodatages de la BL
encodée en Matroska.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

FRAMES = 48
RATE = "24000/1001"


def run(*command: str) -> None:
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path, help="mire complète (.mkv)")
    parser.add_argument("cut", type=Path, help="extrait commençant à l'image 16 (.mkv)")
    parser.add_argument("--muxiveo", type=Path, required=True, help="racine du dépôt Muxiveo")
    args = parser.parse_args()
    sys.path.insert(0, str(args.muxiveo.resolve()))
    from core.matroska.native_muxer import MatroskaNativeMuxer

    fixture = Path(__file__).parent / "fixtures" / "synthetic-fel.hevc"
    with tempfile.TemporaryDirectory(prefix="fel-seek-fixture-") as temporary:
        work = Path(temporary)
        twelve = work / "twelve.bin"
        run("dovi_tool", "extract-rpu", "-i", str(fixture), "-o", str(twelve))
        rpu = work / "rpu.bin"
        rpu.write_bytes(twelve.read_bytes() * (FRAMES // 12))
        for name, size, keyint, hue, extension in (("bl", "256x144", 16, 0, "mkv"), ("el", "128x72", 10, 90, "hevc")):
            run("ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"testsrc2=s={size}:r={RATE},hue=h={hue}",
                "-frames:v", str(FRAMES), "-pix_fmt", "yuv420p10le", "-c:v", "libx265", "-x265-params",
                f"log-level=0:pools=1:frame-threads=1:keyint={keyint}:min-keyint={keyint}:scenecut=0:bframes=3"
                ":open-gop=1:chromaloc=2", str(work / f"{name}.{extension}"))
        run("ffmpeg", "-v", "error", "-y", "-i", str(work / "bl.mkv"), "-c", "copy", "-bsf:v", "hevc_mp4toannexb",
            "-f", "hevc", str(work / "bl.hevc"))
        run("dovi_tool", "inject-rpu", "-i", str(work / "el.hevc"), "-r", str(rpu), "-o", str(work / "el-rpu.hevc"))
        muxed = work / "muxed.hevc"
        run("dovi_tool", "mux", "--bl", str(work / "bl.hevc"), "--el", str(work / "el-rpu.hevc"), "-o", str(muxed))
        timed = work / "timed.mkv"
        MatroskaNativeMuxer().mux(hevc_input=muxed, source_for_timestamps=work / "bl.mkv", output=timed,
                                  pixel_width=256, pixel_height=144, timestamp_order="packet")
        # Recopie FFmpeg : un cluster indexé par image clé (positionnement au milieu de la mire).
        run("ffmpeg", "-v", "error", "-y", "-i", str(timed), "-map", "0:v:0", "-c", "copy", str(args.output.resolve()))
        cut_from_keyframe(args.output.resolve(), args.cut.resolve(), keyframe=1)
        # Extrait précédé d'un paquet sans image (NAL invalide) et un second entre la CRA BL et l'IRAP EL.
        with_invalid_blocks(args.cut.resolve(), args.cut.resolve().with_name(args.cut.stem + "-invalid.mkv"),
                            before=(0, 2))


def cut_from_keyframe(source: Path, output: Path, *, keyframe: int) -> None:
    """Blocs recopiés à partir de la n-ième image clé (0 : la première), ordre de décodage conservé :
    la CRA BL et ses RASL ouvrent l'extrait (FFmpeg ne coupe pas en copie un flux sans DTS)."""
    from core.matroska.mux_plan import MatroskaMuxPacket, MatroskaMuxPlan, MatroskaMuxTrack
    from core.matroska.reader import MatroskaReader
    from core.matroska.writer import MatroskaWriter

    reader = MatroskaReader(source)
    reader.segment()
    (track,) = [item for item in reader.tracks() if item.track_type == 1]
    blocks = list(reader.blocks(track_numbers={track.number}))
    starts = [index for index, block in enumerate(blocks) if block.is_keyframe]
    kept = blocks[starts[keyframe]:]
    MatroskaWriter().write(MatroskaMuxPlan(
        output=output,
        tracks=(MatroskaMuxTrack(source=source, source_track=track, output_number=1, output_uid=1),),
        packets=tuple(MatroskaMuxPacket(output_track_number=1, block=block, source_sequence=index)
                      for index, block in enumerate(kept)),
    ))


#: NAL à en-tête invalide (tranche, forbidden_zero_bit à 1, first_slice_segment_in_pic_flag à 1), ignorée par
#: le décodeur ; préfixée par sa longueur (hvcC, 4 octets).
INVALID_NAL = bytes([0x80 | (1 << 1), 0x01, 0x80, 0x55, 0xAA])


def with_invalid_blocks(source: Path, output: Path, *, before: tuple[int, ...]) -> None:
    """Copie de ``source`` avec, devant chacun des blocs ``before`` (ordre de décodage), un bloc ne contenant
    qu'une NAL invalide (même horodatage que le bloc suivant, sans image clé) : paquet lisible sans image."""
    import dataclasses

    from core.matroska.mux_plan import MatroskaMuxPacket, MatroskaMuxPlan, MatroskaMuxTrack
    from core.matroska.reader import MatroskaReader
    from core.matroska.writer import MatroskaWriter

    reader = MatroskaReader(source)
    reader.segment()
    (track,) = [item for item in reader.tracks() if item.track_type == 1]
    junk = len(INVALID_NAL).to_bytes(4, "big") + INVALID_NAL
    blocks = []
    for index, block in enumerate(reader.blocks(track_numbers={track.number})):
        if index in before:
            blocks.append(dataclasses.replace(block, payload=junk, payload_bytes=len(junk), flags=block.flags & ~0x80,
                                              is_keyframe=False, references=(), references_ns=()))
        blocks.append(block)
    MatroskaWriter().write(MatroskaMuxPlan(
        output=output,
        tracks=(MatroskaMuxTrack(source=source, source_track=track, output_number=1, output_uid=1),),
        packets=tuple(MatroskaMuxPacket(output_track_number=1, block=block, source_sequence=index)
                      for index, block in enumerate(blocks)),
    ))


if __name__ == "__main__":
    main()
