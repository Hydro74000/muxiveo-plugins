"""Vérifie l'ABI et une reconstruction synthétique sur la bibliothèque livrée."""
from __future__ import annotations

import argparse
import ctypes as c
import json
import os
import re
import tempfile
from pathlib import Path

WRITE = c.CFUNCTYPE(c.c_int, c.c_void_p, c.c_void_p, c.c_size_t)
PROGRESS = c.CFUNCTYPE(None, c.c_void_p, c.c_int64)


def load(path: Path):
    lib = c.CDLL(str(path.resolve()))
    lib.mvo_fel_abi_version.restype = c.c_uint32
    assert lib.mvo_fel_abi_version() == 1
    lib.mvo_fel_capabilities.restype = c.c_char_p
    assert json.loads(lib.mvo_fel_capabilities())["pixel_format"] == "gbrp16le"
    lib.mvo_fel_create.argtypes = [c.c_char_p, c.c_int, c.c_int, c.c_int, c.c_int]
    lib.mvo_fel_create.restype = c.c_void_p
    lib.mvo_fel_run.argtypes = [c.c_void_p, WRITE, PROGRESS, c.c_void_p]
    lib.mvo_fel_run.restype = c.c_int
    lib.mvo_fel_error.argtypes = [c.c_void_p]
    lib.mvo_fel_error.restype = c.c_char_p
    lib.mvo_fel_run_fd.argtypes = [c.c_void_p, c.c_ssize_t, PROGRESS, c.c_void_p]
    lib.mvo_fel_run_fd.restype = c.c_int
    lib.mvo_fel_set_output.argtypes = [c.c_void_p, c.c_char_p, c.c_char_p, c.c_char_p]
    lib.mvo_fel_set_output.restype = c.c_int
    lib.mvo_fel_set_crop.argtypes = [c.c_void_p, *([c.c_int] * 4)]
    lib.mvo_fel_set_crop.restype = c.c_int
    lib.mvo_fel_set_timestamps.argtypes = [c.c_void_p, c.c_char_p]
    lib.mvo_fel_set_timestamps.restype = c.c_int
    lib.mvo_fel_set_start.argtypes = [c.c_void_p, c.c_int64]
    lib.mvo_fel_set_start.restype = c.c_int
    lib.mvo_fel_set_precision.argtypes = [c.c_void_p, c.c_char_p]
    lib.mvo_fel_set_precision.restype = c.c_int
    lib.mvo_fel_set_device.argtypes = [c.c_void_p, c.c_char_p]
    lib.mvo_fel_set_device.restype = c.c_int
    lib.mvo_fel_backend.argtypes = [c.c_void_p]
    lib.mvo_fel_backend.restype = c.c_char_p
    for name in ("mvo_fel_cancel", "mvo_fel_destroy"):
        getattr(lib, name).argtypes = [c.c_void_p]
        getattr(lib, name).restype = None
    return lib


def smoke(library: Path, fixture: Path, output: Path | None = None, device: str = "auto") -> None:
    lib = load(library)
    assert json.loads(lib.mvo_fel_capabilities()).get("seek") == 1
    for mode, expected in (("normal", 0), ("seek", 0), ("closed", 2), ("cancel", 3), ("bad_stream", 1),
                           ("missing_source", 4)):
        path = fixture.with_name("missing-source.hevc") if mode == "missing_source" else fixture
        handle = lib.mvo_fel_create(str(path.resolve()).encode(), 999 if mode == "bad_stream" else 0,
                                    2, 24000, 1001)
        assert handle
        if mode == "seek":
            # Cinq images à 24000/1001 : la sixième est la première produite.
            assert lib.mvo_fel_set_start(handle, 5*1001*1000000//24000) == 0
            assert lib.mvo_fel_set_start(handle, -1) == -1
        if device != "auto":
            assert lib.mvo_fel_set_device(handle, device.encode("ascii")) == 0
        received, frames = 0, 0
        header = bytearray()
        sink = output.open("wb") if output and mode == "normal" else None

        def write(_opaque, pointer, size):
            nonlocal received
            if mode == "closed":
                return 1
            data = c.string_at(pointer, size)
            if len(header) < 32:
                header.extend(data[:32-len(header)])
            received += size
            if sink:
                sink.write(data)
            return 0

        def progress(_opaque, count):
            nonlocal frames
            frames = count
            if mode == "cancel":
                lib.mvo_fel_cancel(handle)

        try:
            result = lib.mvo_fel_run(handle, WRITE(write), PROGRESS(progress), None)
            assert result == expected, (mode, result, lib.mvo_fel_error(handle))
            if mode == "seek":
                assert frames == 7, frames
                assert lib.mvo_fel_set_start(handle, 0) == -1
            if mode == "normal":
                assert frames == 12, frames
                assert header.startswith(b"nut/multimedia container\x00"), header
                assert received > 12*256*144*6, received
        finally:
            lib.mvo_fel_destroy(handle)
            if sink:
                sink.close()
    fd_output(lib, fixture)
    yuv_output(lib, fixture)
    precision_modes(lib, fixture, device)
    missing_layers(lib, fixture, device)
    frame_statistics(lib, fixture, device)
    measure(lib, fixture, device)
    capture(lib, fixture, device)
    print("ABI, NUT, 12 images/B-frames, seek, descripteur, Y'CbCr, précisions, AU sans RPU ni EL, statistiques, mesure, "
          "capture, fermeture aval, annulation, piste invalide et erreur de lecture : OK")


def capture(lib, fixture: Path, device: str) -> None:
    """Session de capture (capacité capture) : refus, CPU = GPU au bit près, positionnements répétés.

    Positionnement : source en conteneur (MVO_FEL_TEST_CLIP) ; le flux brut de test est refusé.
    """
    assert json.loads(lib.mvo_fel_capabilities()).get("capture") == ["pair"]
    lib.mvo_fel_capture_frame_size.argtypes = [c.c_void_p]
    lib.mvo_fel_capture_frame_size.restype = c.c_size_t
    lib.mvo_fel_capture_pair.argtypes = [c.c_void_p, c.c_int64, c.c_int, c.c_void_p, c.c_size_t]
    lib.mvo_fel_capture_pair.restype = c.c_int

    def session(path: Path, engine: str, output: bytes = b"yuv420p10le"):
        handle = lib.mvo_fel_create(str(path).encode(), 0, 4, 24000, 1001)
        assert handle
        assert lib.mvo_fel_set_output(handle, output, b"topleft", b"none") == 0
        assert lib.mvo_fel_set_device(handle, engine.encode("ascii")) == 0
        assert lib.mvo_fel_set_precision(handle, b"exact") == 0
        return handle

    # Sortie RGB ou flux brut (pas de positionnement) : refusés, erreur lisible.
    for path, output in ((fixture.resolve(), b"gbrp16le"), (fixture.resolve(), b"yuv420p10le")):
        handle = session(path, "cpu", output)
        try:
            assert lib.mvo_fel_capture_frame_size(handle) == 0 and lib.mvo_fel_error(handle)
        finally:
            lib.mvo_fel_destroy(handle)
    clip = os.environ.get("MVO_FEL_TEST_CLIP")
    if not clip:
        return
    container = Path(clip).resolve()
    captured = {}
    for engine in ("cpu", device):
        handle = session(container, engine)
        try:
            size = lib.mvo_fel_capture_frame_size(handle)
            assert size > 0, lib.mvo_fel_error(handle)
            small = c.create_string_buffer(size)
            assert lib.mvo_fel_capture_pair(handle, 0, 1, small, size) != 0  # deux couches : 2 × size requis
            runs = []
            for start in (200000, 600000, 200000):
                buffer = c.create_string_buffer(4 * size)
                result = lib.mvo_fel_capture_pair(handle, start, 2, buffer, 4 * size)
                assert result == 0, (engine, lib.mvo_fel_error(handle))
                runs.append(buffer.raw)
            assert runs[0] == runs[2], engine  # même position : mêmes octets après un autre positionnement
            frames = [runs[0][k*size:(k + 1)*size] for k in range(4)]
            assert frames[0] != frames[1]  # reconstruite ≠ mapped (FEL)
            captured[engine] = runs
        finally:
            lib.mvo_fel_destroy(handle)
    assert captured["cpu"] == captured[device]


def frame_statistics(lib, fixture: Path, device: str) -> None:
    """Statistiques de l'image : format, plans, fenêtre recadrée ; fichier CPU identique au GPU."""
    assert json.loads(lib.mvo_fel_capabilities()).get("frame_statistics") == 1
    lib.mvo_fel_set_frame_statistics.argtypes = [c.c_void_p, c.c_char_p]
    lib.mvo_fel_set_frame_statistics.restype = c.c_int

    def run(engine: str, path: Path, yuv: bool) -> list[dict]:
        handle = lib.mvo_fel_create(str(fixture.resolve()).encode(), 0, 2, 24000, 1001)
        assert handle
        try:
            assert lib.mvo_fel_set_device(handle, engine.encode("ascii")) == 0
            if yuv:
                assert lib.mvo_fel_set_output(handle, b"yuv420p10le", b"topleft", b"none") == 0
                assert lib.mvo_fel_set_crop(handle, 0, 8, 2, 8) == 0
            assert lib.mvo_fel_set_frame_statistics(handle, b"") == -1
            assert lib.mvo_fel_set_frame_statistics(handle, str(path).encode()) == 0
            result = lib.mvo_fel_run(handle, WRITE(lambda *_: 0), PROGRESS(lambda *_: None), None)
            assert result == 0, lib.mvo_fel_error(handle)
            assert lib.mvo_fel_set_frame_statistics(handle, str(path).encode()) == -1
        finally:
            lib.mvo_fel_destroy(handle)
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

    with tempfile.TemporaryDirectory(prefix="fel-stats-") as temporary:
        for yuv, (width, height) in ((False, (256, 144)), (True, (254, 128))):
            cpu_path, gpu_path = Path(temporary) / f"cpu-{yuv}.jsonl", Path(temporary) / f"gpu-{yuv}.jsonl"
            lines = run("cpu", cpu_path, yuv)
            run(device, gpu_path, yuv)
            assert cpu_path.read_bytes() == gpu_path.read_bytes(), yuv
            header, frames = lines[0], [line for line in lines if "frame" in line]
            scenes = [line for line in lines if "scene_first" in line]
            assert header["format"] == "mvo-fel-frame-statistics" and header["bins"] == 1024
            assert (header["width"], header["height"]) == (width, height), header
            assert [f["frame"] for f in frames] == list(range(12))
            # Plans contigus couvrant toutes les images ; histogrammes complets.
            assert scenes[0]["scene_first"] == 0 and sum(scene["frames"] for scene in scenes) == 12
            for scene, following in zip(scenes, scenes[1:] + [None]):
                if following:
                    assert following["scene_first"] == scene["scene_first"] + scene["frames"]
                # Fenêtre mesurée : fenêtre encodée (zone active L5 entière sur la séquence de test).
                windows = [f["window"] for f in frames[scene["scene_first"]:scene["scene_first"] + scene["frames"]]]
                assert all(w[2:] == [width, height] for w in windows), windows
                assert sum(count for _, count in scene["maxrgb_hist"]) == sum(w[2]*w[3] for w in windows)
            for f in frames:
                assert 0 <= f["y100"] <= 1 and f["avg_nits"] <= max(f["max_nits"]) + 1e-3 and f["y9999_nits"] > 0, f
                percentiles = f["maxrgb_percentiles"]
                assert len(percentiles) == 7 and percentiles == sorted(percentiles), f


def measure(lib, fixture: Path, device: str) -> None:
    """Mesure seule : aucune image produite, statistiques identiques au CPU et au GPU.

    « reconstructed » : mêmes statistiques que la production ; « base » : BL lue comme un signal HDR10,
    RPU et EL ignorés pour les pixels (flux sans RPU ni EL : mêmes mesures, sans plans).
    """
    capabilities = json.loads(lib.mvo_fel_capabilities())
    assert capabilities.get("measure") == capabilities.get("layers") == ["base", "mapped", "reconstructed"]
    assert capabilities.get("measure_sampling") == ["all", "keyframes"]
    for name in ("mvo_fel_set_measure", "mvo_fel_set_layer", "mvo_fel_set_measure_sampling"):
        getattr(lib, name).argtypes = [c.c_void_p, c.c_char_p]
        getattr(lib, name).restype = c.c_int
    lib.mvo_fel_set_frame_statistics.argtypes = [c.c_void_p, c.c_char_p]
    lib.mvo_fel_set_frame_statistics.restype = c.c_int
    run_measure = lib.mvo_fel_run
    no_write = c.cast(None, WRITE)

    def measured(path: Path, layer: str | None, engine: str, stats: Path | None, produce: bool = False,
                 sampling: str | None = None):
        handle = lib.mvo_fel_create(str(path).encode(), 0, 2, 24000, 1001)
        assert handle
        try:
            assert lib.mvo_fel_set_device(handle, engine.encode("ascii")) == 0
            if sampling is not None:
                assert lib.mvo_fel_set_measure_sampling(handle, sampling.encode()) == 0
            if stats is not None:
                assert lib.mvo_fel_set_frame_statistics(handle, str(stats).encode()) == 0
            if layer and produce:
                assert lib.mvo_fel_set_layer(handle, layer.encode()) == 0
            elif layer:
                assert lib.mvo_fel_set_measure(handle, layer.encode()) == 0
            written = []
            sink = WRITE(lambda _o, _p, n: written.append(n) or 0) if produce else no_write
            result = run_measure(handle, sink, PROGRESS(lambda *_: None), None)
            return result, sum(written), (lib.mvo_fel_error(handle) or b"").decode()
        finally:
            lib.mvo_fel_destroy(handle)

    handle = lib.mvo_fel_create(str(fixture.resolve()).encode(), 0, 2, 24000, 1001)
    try:
        assert lib.mvo_fel_set_measure(handle, b"bl") == -1 and lib.mvo_fel_set_measure(handle, None) == -1
        assert lib.mvo_fel_set_measure_sampling(handle, b"rapide") == -1
        assert lib.mvo_fel_set_layer(handle, b"el") == -1
        assert lib.mvo_fel_set_measure(handle, b"base") == 0
        # Mesure : RGB pleine image imposé.
        assert lib.mvo_fel_set_output(handle, b"yuv420p10le", b"topleft", b"none") == -1
        assert lib.mvo_fel_set_crop(handle, 0, 8, 0, 8) == -1
        assert lib.mvo_fel_set_output(handle, b"gbrp16le", b"left", b"none") == 0
    finally:
        lib.mvo_fel_destroy(handle)
    with tempfile.TemporaryDirectory(prefix="fel-measure-") as temporary:
        folder = Path(temporary)
        # Sans fichier de statistiques : refusé ; callback nul sans mesure : refusé.
        assert measured(fixture.resolve(), "base", "cpu", None)[0] == 1
        assert measured(fixture.resolve(), None, "cpu", folder / "none.jsonl")[0] == 1
        # Reconstruite : mêmes statistiques que la production, rien d'écrit.
        produced = folder / "produced.jsonl"
        result, size, error = measured(fixture.resolve(), None, "cpu", produced, produce=True)
        assert result == 0 and size > 0, error
        for engine in ("cpu", device):
            path = folder / f"reconstructed-{engine}.jsonl"
            result, size, error = measured(fixture.resolve(), "reconstructed", engine, path)
            assert result == 0 and size == 0, (engine, error)
            assert path.read_bytes() == produced.read_bytes(), engine
        base = {}
        for engine in ("cpu", device):
            path = folder / f"base-{engine}.jsonl"
            result, _, error = measured(fixture.resolve(), "base", engine, path)
            assert result == 0, (engine, error)
            base[engine] = path.read_bytes()
        assert base["cpu"] == base[device]
        assert base["cpu"] != produced.read_bytes()  # le résiduel FEL change l'image
        lines = [json.loads(line) for line in base["cpu"].decode().splitlines()]
        frames = [line for line in lines if "frame" in line]
        assert len(frames) == 12 and frames[0]["window"] == [0, 0, 256, 144]
        # Flux sans RPU ni EL (HDR10 simple) : mêmes pixels mesurés, aucun plan signalé.
        plain = folder / "plain.hevc"
        strip_layers(fixture, plain, {unit: {62, 63} for unit in range(12)})
        path = folder / "plain.jsonl"
        result, _, error = measured(plain, "base", device, path)
        assert result == 0, error
        plain_lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        plain_frames = [line for line in plain_lines if "frame" in line]
        assert [line for line in plain_lines if "scene_first" in line][0]["frames"] == 12
        strip = lambda f: {k: v for k, v in f.items() if k != "scene_start"}  # noqa: E731
        assert [strip(f) for f in plain_frames] == [strip(f) for f in frames]
        # Reconstruite ou mapped sur un flux sans EL ni RPU : refusées.
        assert measured(plain, "reconstructed", "cpu", folder / "refused.jsonl")[0] != 0
        assert measured(plain, "mapped", "cpu", folder / "refused-mapped.jsonl")[0] != 0
        # Mapped : courbes du RPU sans résiduel, CPU = GPU, différente de l'image reconstruite (FEL).
        mapped = {}
        for engine in ("cpu", device):
            path = folder / f"mapped-{engine}.jsonl"
            result, _, error = measured(fixture.resolve(), "mapped", engine, path)
            assert result == 0, (engine, error)
            mapped[engine] = path.read_bytes()
        assert mapped["cpu"] == mapped[device] and mapped["cpu"] != produced.read_bytes()
        # Couches produites (pas de mesure seule) : images écrites, mêmes statistiques que la mesure.
        for layer, reference in (("base", base["cpu"]), ("mapped", mapped["cpu"])):
            path = folder / f"produced-{layer}.jsonl"
            result, size, error = measured(fixture.resolve(), layer, "cpu", path, produce=True)
            assert result == 0 and size > 12 * 256 * 144 * 6, (layer, error)
            assert path.read_bytes() == reference, layer
        # Mesure par images clés (0.5.0) : flux brut refusé (horodatages), reconstruite refusée.
        assert measured(fixture.resolve(), "base", "cpu", folder / "kf-raw.jsonl", sampling="keyframes")[0] != 0
        # Source en conteneur (clip réel, comme les tests natifs GPU) : MVO_FEL_TEST_CLIP.
        clip = os.environ.get("MVO_FEL_TEST_CLIP")
        if clip:
            container = Path(clip)
            assert measured(container, "reconstructed", "cpu", folder / "kf-rec.jsonl", sampling="keyframes")[0] != 0
            full_path, sampled_path = folder / "kf-all.jsonl", folder / "kf.jsonl"
            assert measured(container, "base", "cpu", full_path, sampling="all")[0] == 0
            result, _, error = measured(container, "base", device, sampled_path, sampling="keyframes")
            assert result == 0, error
            full = [json.loads(line) for line in full_path.read_text(encoding="utf-8").splitlines()]
            sampled = [json.loads(line) for line in sampled_path.read_text(encoding="utf-8").splitlines()]
            assert sampled[0].get("sampling") == "keyframes" and "sampling" not in full[0]
            full_frames = [line for line in full if "frame" in line]
            sampled_frames = [line for line in sampled if "frame" in line]
            assert 1 <= len(sampled_frames) < len(full_frames)
            times = [line["time_s"] for line in sampled_frames]
            assert times == sorted(times) and times[0] == full_frames[0]["time_s"]
            # Image clé : mêmes statistiques que dans la mesure complète (même image ; GPU rapide ≈ CPU).
            same = {line["time_s"]: line for line in full_frames}
            assert all(f["time_s"] in same for f in sampled_frames)
            for frame in sampled_frames:
                reference = same[frame["time_s"]]
                assert abs(frame["avg_nits"] - reference["avg_nits"]) <= 0.01 * max(1.0, reference["avg_nits"])


def strip_layers(fixture: Path, output: Path, units: dict[int, set[int]]) -> None:
    """Copie du flux sans les NAL indiqués (62 RPU, 63 EL) des access units choisies (ordre de décodage)."""
    data = fixture.read_bytes()
    starts = [m.start() - (m.start() > 0 and data[m.start() - 1] == 0) for m in re.finditer(b"\x00\x00\x01", data)]
    kept, unit = bytearray(), -1
    for begin, end in zip(starts, starts[1:] + [len(data)]):
        header = data.index(b"\x00\x00\x01", begin) + 3
        kind = (data[header] >> 1) & 0x3F
        if kind < 32 and data[header + 2] & 0x80:  # première tranche d'une image
            unit += 1
        if kind not in units.get(unit, ()):
            kept += data[begin:end]
    output.write_bytes(bytes(kept))


def missing_layers(lib, fixture: Path, device: str) -> None:
    """AU sans RPU ni EL : recomposées sans résiduel (CPU = GPU exact) ; RPU ou EL seul : refusé.

    BL et EL de structures de GOP différentes : appariées par rang de présentation.
    """
    lib.mvo_fel_statistics.argtypes = [c.c_void_p]
    lib.mvo_fel_statistics.restype = c.c_char_p

    def render(path: Path, engine: str) -> tuple[int, bytes, dict]:
        handle = lib.mvo_fel_create(str(path).encode(), 0, 2, 24000, 1001)
        assert handle
        data = bytearray()
        try:
            assert lib.mvo_fel_set_precision(handle, b"exact") == 0
            assert lib.mvo_fel_set_device(handle, engine.encode("ascii")) == 0
            result = lib.mvo_fel_run(handle, WRITE(lambda _o, p, n: data.extend(c.string_at(p, n)) or 0),
                                     PROGRESS(lambda *_: None), None)
            statistics = json.loads(lib.mvo_fel_statistics(handle) or b"{}")
            statistics["gpu"] = (lib.mvo_fel_backend(handle) or b"").startswith(b"vulkan")
            return result, bytes(data), statistics
        finally:
            lib.mvo_fel_destroy(handle)

    with tempfile.TemporaryDirectory(prefix="fel-missing-") as temporary:
        stripped = Path(temporary) / "stripped.hevc"
        _, complete, statistics = render(fixture.resolve(), "cpu")
        assert statistics["bl_only_frames"] == 0
        # AU non référencées (TRAIL_N) : deux en fin de flux comme certains disques, une en milieu de flux.
        for units, count in (({10: {62, 63}, 11: {62, 63}}, 2), ({3: {62, 63}}, 1)):
            strip_layers(fixture, stripped, units)
            result, cpu, statistics = render(stripped, "cpu")
            assert result == 0 and statistics["frames"] == 12 and statistics["bl_only_frames"] == count, (units, statistics)
            assert len(cpu) == len(complete) and cpu != complete
            result, gpu, statistics = render(stripped, device)
            assert result == 0 and gpu == cpu and statistics["bl_only_frames"] == count, (units, result, statistics)
        # RPU sans EL, EL sans RPU (couche orpheline en fin de flux), IDR sans couche : refusés.
        for units in ({10: {63}}, {10: {62}}, {0: {62, 63}}):
            strip_layers(fixture, stripped, units)
            assert render(stripped, "cpu")[0] != 0, units
    # Liste des indécises du GPU saturée (simulée) : chaque image refaite sur le CPU certifié, mêmes octets.
    os.environ["MVO_FEL_TEST_OVERFLOW"] = "1"
    try:
        result, data, statistics = render(fixture.resolve(), device)
    finally:
        del os.environ["MVO_FEL_TEST_OVERFLOW"]
    assert result == 0 and data == complete, (result, statistics)
    assert statistics["cpu_rescue_frames"] == (12 if statistics["gpu"] else 0), statistics
    # GOP différents (BL sans B-frames, EL à trois B-frames) : appariement par rang de présentation,
    # même image que la séquence conforme (couches grises identiques une fois décodées).
    mixed = fixture.with_name("synthetic-fel-mixed-gop.hevc")
    for engine in ("cpu", device):
        result, data, statistics = render(mixed, engine)
        assert result == 0 and statistics["frames"] == 12 and data == complete, (engine, result, statistics)


def precision_modes(lib, fixture: Path, device: str) -> None:
    """Modes de précision : valeurs refusées ; exact identique au CPU octet pour octet."""
    capabilities = json.loads(lib.mvo_fel_capabilities())
    assert capabilities.get("precision") == ["exact", "standard", "fast"]
    assert capabilities.get("device_features") == ["float64"]
    # Liste historique sans clé nouvelle (versions publiées de Muxiveo) ; liste étendue avec float64.
    for name, extended in (("mvo_fel_devices", False), ("mvo_fel_devices_features", True)):
        function = getattr(lib, name)
        function.argtypes = [c.c_void_p, c.c_size_t]
        function.restype = c.c_size_t
        size = function(None, 0)
        buffer = c.create_string_buffer(size)
        assert function(buffer, size) == size
        assert all(("float64" in item) == extended for item in json.loads(buffer.value))

    def render(engine: str, precision: str | None) -> tuple[bytes, str]:
        handle = lib.mvo_fel_create(str(fixture.resolve()).encode(), 0, 2, 24000, 1001)
        assert handle
        data = bytearray()
        try:
            assert lib.mvo_fel_set_precision(handle, b"double") == -1
            if precision:
                assert lib.mvo_fel_set_precision(handle, precision.encode()) == 0
            assert lib.mvo_fel_set_device(handle, engine.encode("ascii")) == 0

            def write(_opaque, pointer, size):
                data.extend(c.string_at(pointer, size))
                return 0

            assert lib.mvo_fel_run(handle, WRITE(write), PROGRESS(lambda *_: None), None) == 0
            assert lib.mvo_fel_set_precision(handle, b"exact") == -1
            return bytes(data), (lib.mvo_fel_backend(handle) or b"").decode("utf-8", "replace")
        finally:
            lib.mvo_fel_destroy(handle)

    cpu, _ = render("cpu", "exact")
    exact, backend = render(device, "exact")
    standard, _ = render(device, "standard")
    fast, _ = render(device, None)
    # CPU certifié ; GPU exact : identique (variante double ou virgule fixe) ou repli CPU.
    assert exact == cpu, backend
    assert len(standard) == len(fast) == len(cpu)
    assert backend.startswith("cpu") or "précision exacte" in backend, backend
    # Images en vol (lot P) : valeurs refusées hors de [1, 4] ; sortie identique quel que soit le nombre.
    if capabilities.get("render_workers"):
        lib.mvo_fel_set_render_workers.argtypes = [c.c_void_p, c.c_int]
        lib.mvo_fel_set_render_workers.restype = c.c_int
        for workers in (1, 3):
            handle = lib.mvo_fel_create(str(fixture.resolve()).encode(), 0, 2, 24000, 1001)
            try:
                assert lib.mvo_fel_set_render_workers(handle, 0) == -1
                assert lib.mvo_fel_set_render_workers(handle, 5) == -1
                assert lib.mvo_fel_set_render_workers(handle, workers) == 0
                assert lib.mvo_fel_set_precision(handle, b"exact") == 0
                assert lib.mvo_fel_set_device(handle, device.encode("ascii")) == 0
                data = bytearray()
                assert lib.mvo_fel_run(handle, WRITE(lambda _o, p, n: data.extend(c.string_at(p, n)) or 0),
                                       PROGRESS(lambda *_: None), None) == 0
                assert lib.mvo_fel_set_render_workers(handle, 2) == -1
                assert bytes(data) == cpu, workers
            finally:
                lib.mvo_fel_destroy(handle)


def yuv_output(lib, fixture: Path) -> None:
    """Sorties Y'CbCr 4:2:0 : taille des images, refus des valeurs inconnues ou tardives."""
    lib.mvo_fel_set_device.argtypes = [c.c_void_p, c.c_char_p]
    lib.mvo_fel_set_device.restype = c.c_int
    for (name, depth), device in ((formats, engine) for formats in (("yuv420p16le", 2), ("yuv420p10le", 2), ("yuv420p", 1))
                                  for engine in (b"cpu", b"auto")):
        handle = lib.mvo_fel_create(str(fixture.resolve()).encode(), 0, 2, 24000, 1001)
        assert lib.mvo_fel_set_device(handle, device) == 0
        assert lib.mvo_fel_set_output(handle, b"rgb48", b"topleft", b"blue") == -1
        assert lib.mvo_fel_set_output(handle, name.encode(), b"diagonal", b"blue") == -1
        assert lib.mvo_fel_set_output(handle, name.encode(), b"topleft", b"random") == -1
        assert lib.mvo_fel_set_output(handle, name.encode(), b"topleft", b"blue") == 0
        assert lib.mvo_fel_set_crop(handle, 1, 0, 0, 0) == -1
        assert lib.mvo_fel_set_crop(handle, 0, 8, 2, 8) == 0
        received = bytearray()
        result = lib.mvo_fel_run(handle, WRITE(lambda _o, p, n: received.extend(c.string_at(p, n)) or 0),
                                 PROGRESS(lambda *_: None), None)
        assert result == 0, (name, lib.mvo_fel_error(handle))
        assert lib.mvo_fel_set_output(handle, name.encode(), b"topleft", b"blue") == -1
        lib.mvo_fel_destroy(handle)
        frame = 254*128*3//2*depth
        assert 12*frame < len(received) < 12*frame + 4096, (name, len(received))


def fd_output(lib, fixture: Path) -> None:
    """Sortie directe sur pipe : même flux que le callback, puis aval fermé sans SIGPIPE."""
    import os
    import threading
    expected = bytearray()
    handle = lib.mvo_fel_create(str(fixture.resolve()).encode(), 0, 2, 24000, 1001)
    assert lib.mvo_fel_run(handle, WRITE(lambda _o, p, n: expected.extend(c.string_at(p, n)) or 0),
                           PROGRESS(lambda *_: None), None) == 0
    lib.mvo_fel_destroy(handle)
    for close_early in (False, True):
        read, write = os.pipe()
        received = bytearray()

        def consume():
            with os.fdopen(read, "rb") as stream:
                if close_early:
                    stream.read(4096)
                    return
                while chunk := stream.read(1 << 20):
                    received.extend(chunk)

        reader = threading.Thread(target=consume)
        reader.start()
        handle = lib.mvo_fel_create(str(fixture.resolve()).encode(), 0, 2, 24000, 1001)
        if os.name == "nt":
            import msvcrt
            target = msvcrt.get_osfhandle(write)
        else:
            target = write
        result = lib.mvo_fel_run_fd(handle, target, PROGRESS(lambda *_: None), None)
        os.close(write)
        reader.join()
        lib.mvo_fel_destroy(handle)
        if close_early:
            assert result == 2, result
        else:
            assert result == 0 and bytes(received) == bytes(expected), (result, len(received), len(expected))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("library", type=Path)
    parser.add_argument("--fixture", type=Path, default=Path(__file__).parent / "fixtures/synthetic-fel.hevc")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    smoke(args.library, args.fixture, args.output, args.device)
