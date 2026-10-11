"""Une translation connue conserve les trois canaux, les fractions et les bords."""

from pathlib import Path

import numpy as np
import pytest

from test_muxiveo_rife import parse_y4m, run_rife
from test_muxiveo_rife import pytestmark as pytestmark


@pytest.mark.parametrize('size', [(32, 32), (34, 22)])
@pytest.mark.parametrize('flow', [(.5, .5), (1., -1.), (128., -128.)])
@pytest.mark.parametrize('fp32', [False, True])
def test_rgb_warp_matches_known_translation(tmp_path: Path, size, flow, fp32):
    width, height = size
    xx, yy = np.meshgrid(np.arange(width), np.arange(height))
    original = (100 + xx * 6 + yy * 9).astype('<u2')
    chroma = np.full(width * height // 2, 512, dtype='<u2').tobytes()
    header = f'YUV4MPEG2 W{width} H{height} F24:1 Ip C420p10\n'.encode()
    data = header + b'FRAME\n' + original.tobytes() + chroma
    data += b'FRAME\n' + np.flip(original).tobytes() + chroma
    padded_w, padded_h = (width + 31) // 32 * 32, (height + 31) // 32 * 32
    model = tmp_path / 'translation'
    model.mkdir()
    (model / 'flownet.param').write_text(f'''7767517
5 5
Input a 0 1 in0
Input b 0 1 in1
Input t 0 1 in2
MemoryData flow 0 1 flow 0={padded_w} 1={padded_h} 2=2
rife.Warp translation 2 1 in0 flow out0
''')
    vectors = np.empty((2, padded_h, padded_w), dtype='<f4')
    vectors[0].fill(flow[0])
    vectors[1].fill(flow[1])
    (model / 'flownet.bin').write_bytes(vectors.tobytes())
    result = run_rife(data, '--model', str(model), '--engine', 'rife', '--matrix', 'bt709',
                      '--scene-threshold', '0', '--factor', '2', *(['--fp32'] if fp32 else []))
    assert result.returncode == 0, result.stderr.decode(errors='replace')
    output = parse_y4m(result.stdout)
    assert len(output.frames) == 4
    assert output.frames[0::2] == parse_y4m(data).frames
    actual = np.frombuffer(output.frames[1], dtype='<u2')
    # Référence analytique du plan affine, avec le padding RGB noir fourni
    # par la conversion native. Aux bords du tenseur paddé, les pixels sont répliqués.
    sx = np.clip(xx + flow[0], 0, padded_w - 1)
    sy = np.clip(yy + flow[1], 0, padded_h - 1)
    affine = 100 + np.minimum(sx, width - 1) * 6 + np.minimum(sy, height - 1) * 9
    coverage = np.clip(width - sx, 0, 1) * np.clip(height - sy, 0, 1)
    expected = 64 + (affine - 64) * coverage
    assert np.abs(actual[:width * height].reshape(height, width) - expected).max() <= 1
    assert np.abs(actual[width * height:].astype(np.int32) - 512).max() <= 1
