"""Conserver les coefficients actifs et refuser toute preuve ou disposition ambiguë."""

from pathlib import Path
import shutil
import struct
import subprocess

import pytest

SOURCE = Path(__file__).resolve().parents[1] / 'src'
GRAPH = '''7767517
8 9
Input image 0 1 in0
Deconvolution flow 1 1 in0 raw 0=24 1=3 5=1 6=216
PixelShuffle shuffle 1 1 raw shuffled 0=2
Interp resize 1 1 shuffled scaled 0=2 1=2.0 2=2.0
Split split 1 2 scaled flow_input mask_input
Crop crop_flow 1 1 flow_input flow4 -23309=1,0 -23310=1,4 -23311=1,0
Crop crop_mask 1 1 mask_input mask1 -23309=1,4 -23310=1,5 -23311=1,0
Concat result 2 1 flow4 mask1 out0
'''


def payload(half=False):
    flag = 0x01306b47 if half else 0
    weights = b''.join(struct.pack('<e' if half else '<f', i / 256) for i in range(216))
    bias = b''.join(struct.pack('<f', -i / 32) for i in range(24))
    return struct.pack('<I', flag) + weights + bias


@pytest.fixture(scope='module')
def prune(tmp_path_factory):
    compiler = shutil.which('c++') or shutil.which('g++')
    if not compiler:
        pytest.skip('Compilateur C++ requis')
    folder = tmp_path_factory.mktemp('flow_pruning')
    driver = folder / 'probe.cpp'
    driver.write_text('''#include "flow_pruning.h"
#include <fstream>
#include <iostream>
#include <iterator>
int main(int argc, char** argv) {
    if (argc != 5) return 2;
    std::ifstream param_file(argv[1]), weights_file(argv[2], std::ios::binary);
    const std::string param((std::istreambuf_iterator<char>(param_file)), std::istreambuf_iterator<char>());
    const std::vector<unsigned char> weights((std::istreambuf_iterator<char>(weights_file)), std::istreambuf_iterator<char>());
    std::string output_param;
    std::vector<unsigned char> output_weights;
    const int count = prune_unused_flow(param, weights, output_param, output_weights);
    std::ofstream(argv[3]) << output_param;
    std::ofstream binary(argv[4], std::ios::binary);
    binary.write(reinterpret_cast<const char*>(output_weights.data()), output_weights.size());
    std::cout << count;
}
''')
    binary = folder / 'probe'
    subprocess.run([compiler, '-std=c++17', '-O2', '-Wall', '-Wextra', '-Werror',
                    '-I' + str(SOURCE), str(driver), str(SOURCE / 'flow_pruning.cpp'), '-o', str(binary)],
                   check=True, capture_output=True, timeout=90)

    def invoke(graph, weights):
        paths = [folder / name for name in ('input.param', 'input.bin', 'output.param', 'output.bin')]
        paths[0].write_text(graph)
        paths[1].write_bytes(weights)
        result = subprocess.run([str(binary), *map(str, paths)], check=True, capture_output=True, timeout=10)
        return int(result.stdout), paths[2].read_text(), paths[3].read_bytes()

    return invoke


@pytest.mark.parametrize('half', [False, True])
def test_retains_active_coefficients_and_biases_exactly(prune, half):
    original = payload(half)
    count, param, weights = prune(GRAPH, original)
    width = 2 if half else 4
    assert count == 1
    assert '0=20' in param and '6=180' in param
    assert weights == original[:4 + 180 * width] + original[4 + 216 * width:4 + 216 * width + 80]
    assert prune(param, weights) == (0, param, b'')
    assert param.splitlines()[4:] == GRAPH.splitlines()[4:]


@pytest.mark.parametrize('old,new', [
    ('1,5', '1,6'),
    ('-23311=1,0', '-23311=1,1'),
    ('-23310=1,5', '-23310=1,5 42=1'),
    ('shuffled 0=2', 'shuffled 0=2 1=1'),
    ('1=2.0 2=2.0', '1=2.0 2=2.0 3=8'),
    ('Crop crop_mask', 'ReLU crop_mask'),
    ('Input image', 'Input flow'),
    ('8 9', '7 9'),
    ('8 9', '8 10'),
    ('in0 raw', 'missing raw'),
    ('in0 raw', 'in0 in0'),
    ('0=24 1=3', '0=24 0=24 1=3'),
    ('6=216', '6=215'),
    ('5=1', '5=0'),
    ('6=216', '6=216 7=2'),
    ('6=216', '6=216 28=1'),
    ('6=216', '6=216 8=1'),
    ('6=216', '6=' + '9' * 200),
    ('Concat result', 'Uncontrolled result'),
])
def test_unsupported_graph_or_weights_fall_back_atomically(prune, old, new):
    graph = GRAPH.replace(old, new)
    assert prune(graph, payload()) == (0, graph, b'')


@pytest.mark.parametrize('weights', [b'', payload()[:3], payload()[:-1], payload() + b'\0',
                                    struct.pack('<I', 1) + payload()[4:]])
def test_truncation_unknown_tag_and_trailing_bytes_are_refused(prune, weights):
    assert prune(GRAPH, weights) == (0, GRAPH, b'')


def test_sixth_channel_consumed_by_another_branch_cannot_be_removed(prune):
    graph = GRAPH.replace('8 9', '9 10') + 'ReLU other 1 1 shuffled extra\n'
    assert prune(graph, payload()) == (0, graph, b'')


def test_external_final_output_cannot_be_reduced(prune):
    graph = GRAPH.replace('shuffled', 'out0').replace('mask1 out0', 'mask1 final')
    assert prune(graph, payload()) == (0, graph, b'')


def test_other_weight_blocks_and_constant_bytes_are_preserved(prune):
    graph = GRAPH.replace('8 9', '10 11').replace('Input image 0 1 in0\n',
        'Input image 0 1 image0\nConvolution head 1 1 image0 in0 0=1 1=1 5=1 6=3\n'
        'MemoryData constant 0 1 unused 0=2\n')
    head = struct.pack('<I', 0x01306b47) + struct.pack('<3e', .25, .5, .75) + b'\x12\x34' + struct.pack('<f', .125)
    constants = struct.pack('<2f', .1, .9)
    count, _, weights = prune(graph, head + constants + payload())
    assert count == 1
    assert weights[:len(head + constants)] == head + constants


def test_a_later_invalid_block_discards_the_entire_transformation(prune):
    graph = GRAPH.replace('8 9', '9 10') + 'Convolution later 1 1 out0 final 0=1 5=1 6=1\n'
    assert prune(graph, payload() + b'\0\0') == (0, graph, b'')


@pytest.mark.parametrize('suffix', ['11=2', '21=0'])
def test_unsupported_constant_storage_cannot_shift_weight_offsets(prune, suffix):
    graph = GRAPH.replace('8 9', '9 10') + 'MemoryData constant 0 1 other 0=2 ' + suffix + '\n'
    assert prune(graph, payload() + struct.pack('<2f', .1, .9)) == (0, graph, b'')


def test_dynamic_convolution_has_no_static_payload_to_rewrite(prune):
    graph = GRAPH.replace('8 9', '9 10') + 'Convolution later 1 1 out0 final 0=1 5=1 6=1 19=1\n'
    assert prune(graph, payload() + struct.pack('<I2f', 0, .1, .9)) == (0, graph, b'')
