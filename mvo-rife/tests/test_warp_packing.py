"""Des caractéristiques compactées conservent translation, canaux et flux scalaire."""

from pathlib import Path

import numpy as np
import pytest

from test_muxiveo_rife import parse_y4m, run_rife
from test_muxiveo_rife import pytestmark as pytestmark


@pytest.mark.parametrize('channels', [2, 4])
@pytest.mark.parametrize('start', [0, 4])
@pytest.mark.parametrize('flow', [(.5, .5), (128., -128.)])
@pytest.mark.parametrize('fp32', [False, True])
def test_feature_warp_keeps_channels_and_translation(tmp_path: Path, channels, start, flow, fp32):
    width, height = 34, 22
    padded_w, padded_h = 64, 32
    xx, yy = np.meshgrid(np.arange(width), np.arange(height))
    original = (100 + xx * 6 + yy * 9).astype('<u2')
    chroma = np.full(width * height // 2, 512, dtype='<u2').tobytes()
    header = f'YUV4MPEG2 W{width} H{height} F24:1 Ip C420p10\n'.encode()
    data = header + b'FRAME\n' + original.tobytes() + chroma
    data += b'FRAME\n' + np.flip(original).tobytes() + chroma
    features = np.zeros((8, padded_h, padded_w), dtype='<f4')
    for group in (0, 4):
        features[group:group + 3, :height, :width] = (original.astype(np.float32) + group * 10 - 64) / 876
    # Sentinelles non retenues par le Crop : détecter un mélange de composantes.
    features[3].fill(-.5)
    features[7].fill(1.5)
    vectors = np.full((channels, padded_h, padded_w), 19, dtype='<f4')
    vectors[0].fill(flow[0])
    vectors[1].fill(flow[1])
    model = tmp_path / 'translation-features'
    model.mkdir()
    (model / 'flownet.param').write_text(f'''7767517
7 7
Input a 0 1 in0
Input b 0 1 in1
Input t 0 1 in2
MemoryData features 0 1 features 0={padded_w} 1={padded_h} 2=8
MemoryData flow 0 1 flow 0={padded_w} 1={padded_h} 2={channels}
rife.Warp translation 2 1 features flow translated
Crop rgb 1 1 translated out0 -23309=1,{start} -23310=1,{start + 3} -23311=1,0
''')
    (model / 'flownet.bin').write_bytes(features.tobytes() + vectors.tobytes())
    result = run_rife(data, '--model', str(model), '--engine', 'rife', '--matrix', 'bt709',
                      '--scene-threshold', '0', '--factor', '2', *(['--fp32'] if fp32 else []))
    assert result.returncode == 0, result.stderr.decode(errors='replace')
    output = parse_y4m(result.stdout)
    assert len(output.frames) == 4
    assert output.frames[0::2] == parse_y4m(data).frames
    actual = np.frombuffer(output.frames[1], dtype='<u2')
    sx = np.clip(xx + flow[0], 0, padded_w - 1)
    sy = np.clip(yy + flow[1], 0, padded_h - 1)
    affine = 100 + start * 10 + np.minimum(sx, width - 1) * 6 + np.minimum(sy, height - 1) * 9
    coverage = np.clip(width - sx, 0, 1) * np.clip(height - sy, 0, 1)
    expected = 64 + (affine - 64) * coverage
    assert np.abs(actual[:width * height].reshape(height, width) - expected).max() <= 1
    assert np.abs(actual[width * height:].astype(np.int32) - 512).max() <= 1
