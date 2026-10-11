"""Le partage exige des opérations et poids identiques ; un doute conserve les branches séparées."""
from pathlib import Path
import shutil
import subprocess

import pytest

SOURCE = Path(__file__).resolve().parents[1] / 'src'
GRAPH = '''7767517
13 13
Input a 0 1 in0
Input b 0 1 in1
Input t 0 1 in2
MemoryData beta 0 1 beta 0=2
Convolution a0 1 1 in0 a0 0=2 1=1 5=1 6=6 9=2
Convolution a1 1 1 a0 a1 0=2 1=1 5=1 6=4 9=2
Convolution a2 1 1 a1 a2 0=2 1=1 5=1 6=4 9=2
Deconvolution a3 1 1 a2 154 0=2 1=1 5=1 6=4
Convolution b0 1 1 in1 b0 0=2 1=1 5=1 6=6 9=2
Convolution b1 1 1 b0 b1 0=2 1=1 5=1 6=4 9=2
Convolution b2 1 1 b1 b2 0=2 1=1 5=1 6=4 9=2
Deconvolution b3 1 1 b2 166 0=2 1=1 5=1 6=4
Convolution final 1 1 154 out0 0=2 1=1 5=1 6=4
'''
DRIVER = r'''
#include "head_sharing.h"
#include <iostream>
#include <iterator>
#include <limits>
#include <string>
#include <vector>
int main(int argc, char** argv) {
    const std::string mode = argc > 1 ? argv[1] : "same";
    const std::string graph((std::istreambuf_iterator<char>(std::cin)), std::istreambuf_iterator<char>());
    HeadSharingProof proof(graph);
    const int counts[] = {2,6,2,4,2,4,2,4,2,6,2,4,2,4,2,4,2,4,2};
    for (int i=0; i<19; i++) {
        if (mode == "missing" && i == 18) break;
        if (mode == "early" && i == 17) { std::cout << proof.equivalent(); return 0; }
        std::vector<float> tensor(counts[i], .125f);
        if (mode == "weight" && i == 9) tensor[0] = .25f;
        if (mode == "bias" && i == 16) tensor[0] = .25f;
        if (mode == "flow" && i == 17) tensor[0] = .25f;
        if (mode == "nan" && i == 9) tensor[0] = std::numeric_limits<float>::quiet_NaN();
        const int type = i == 0 || i % 2 == 0 ? 1 : 0;
        proof.observe(counts[i], mode == "wrong-type" && i == 9 ? 1 : type,
                      tensor.data(), tensor.size()*sizeof(float));
    }
    if (mode == "extra") { float f=0; proof.observe(1,1,&f,sizeof(f)); }
    std::cout << proof.equivalent();
}
'''


@pytest.fixture(scope='module')
def probe(tmp_path_factory):
    compiler = shutil.which('c++') or shutil.which('g++')
    if not compiler:
        pytest.skip('Compilateur C++ requis pour le contrôle CPU du partage')
    folder = tmp_path_factory.mktemp('head-sharing')
    driver = folder / 'probe.cpp'
    binary = folder / 'probe'
    driver.write_text(DRIVER)
    subprocess.run([compiler, '-std=c++17', '-O2', '-I' + str(SOURCE), str(driver),
                    str(SOURCE / 'head_sharing.cpp'), '-o', str(binary)], check=True,
                   capture_output=True, timeout=90)
    return binary


def accepts(probe, graph=GRAPH, mode='same'):
    result = subprocess.run([str(probe), mode], input=graph, text=True, capture_output=True,
                            check=True, timeout=10)
    return result.stdout == '1'


def test_identical_heads_allow_sharing(probe):
    assert accepts(probe)


@pytest.mark.parametrize('mode', ['weight', 'bias', 'nan', 'missing', 'extra', 'wrong-type', 'early'])
def test_unverified_weights_never_allow_sharing(probe, mode):
    assert not accepts(probe, mode=mode)


def test_changed_flow_weights_do_not_disable_identical_heads(probe):
    assert accepts(probe, mode='flow')


def test_different_activation_disables_sharing(probe):
    assert not accepts(probe, GRAPH.replace('in1 b0 0=2 1=1 5=1 6=6 9=2',
                                           'in1 b0 0=2 1=1 5=1 6=6 9=1'))


def test_parameter_order_is_irrelevant(probe):
    assert accepts(probe, GRAPH.replace('in1 b0 0=2 1=1 5=1 6=6 9=2',
                                        'in1 b0 9=2 6=6 5=1 1=1 0=2'))


@pytest.mark.parametrize('old,new', [
    ('Convolution final', 'InnerProduct final'),
    ('1 1 in1 b0', '1 1 in0 b0'),
    ('1 1 in1 b0', '1 1 166 b0'),
    ('b2 166', 'b2 154'),
    ('b2 166', 'b2 other_head'),
    ('beta 0=2', 'beta 0=2 21=0'),
    ('13 13', '12 13'),
    ('in1 b0 0=2', 'in1 b0 19=1 0=2'),
])
def test_unsupported_or_invalid_graph_disables_sharing(probe, old, new):
    assert not accepts(probe, GRAPH.replace(old, new))
