"""La fusion conserve les consommateurs, paramètres et poids du graphe."""
from pathlib import Path
import shutil
import subprocess

import pytest

SOURCE = Path(__file__).resolve().parents[1] / 'src'
GRAPH = '''7767517
5 5
Input a 0 1 in0
Input b 0 1 in1
Input t 0 1 in2
BinaryOp addition 2 1 in0 in1 sum
ReLU activation 1 1 sum out0 0=0.2
'''


@pytest.fixture(scope='module')
def fuse(tmp_path_factory):
    compiler = shutil.which('c++') or shutil.which('g++')
    if not compiler:
        pytest.skip('Compilateur C++ requis pour vérifier la transformation du graphe')
    folder = tmp_path_factory.mktemp('fusion')
    driver = folder / 'probe.cpp'
    driver.write_text('''#include "graph_fusion.h"
#include <iostream>
#include <iterator>
int main() {
    const std::string input((std::istreambuf_iterator<char>(std::cin)), std::istreambuf_iterator<char>());
    std::string output;
    const int count = fuse_residual_activations(input, output);
    std::cout << count << "\\n" << output;
}
''')
    binary = folder / 'probe'
    subprocess.run([compiler, '-std=c++17', '-O2', '-I' + str(SOURCE), str(driver),
                    str(SOURCE / 'graph_fusion.cpp'), '-o', str(binary)],
                   check=True, capture_output=True, timeout=90)

    def invoke(graph):
        result = subprocess.run([str(binary)], input=graph, text=True, capture_output=True,
                                check=True, timeout=10)
        count, output = result.stdout.split('\n', 1)
        return int(count), output

    return invoke


def test_fuses_addition_without_changing_its_inputs_or_final_output(fuse):
    count, output = fuse(GRAPH)
    assert count == 1
    assert output.splitlines()[1] == '4 4'
    assert output.splitlines()[-1] == 'rife.AddRelu addition 2 1 in0 in1 out0 0=0.2'
    assert fuse(output) == (0, output)


@pytest.mark.parametrize('params', ['', '0=0', '0=0 1=0', '2=3.5 1=0 0=0'])
def test_explicit_and_implicit_addition_parameters(fuse, params):
    assert fuse(GRAPH.replace('in0 in1 sum\n', 'in0 in1 sum ' + params + '\n'))[0] == 1


@pytest.mark.parametrize('old,new', [
    ('in0 in1 sum\n', 'in0 in1 sum 0=2\n'),
    ('in0 in1 sum\n', 'in0 in1 sum 1=1\n'),
    ('in0 in1 sum\n', 'in0 in1 sum 42=1\n'),
    ('0=0.2', '0=nan'),
    ('0=0.2', '0=0.2 1=1'),
    ('0=0.2', '0=0.2suffix'),
    ('5 5', '4 5'),
    ('5 5', '5 6'),
    ('Input b', 'Input a'),
    ('in0 in1 sum', 'in0 missing sum'),
    ('in0 in1 sum', 'in0 in1 in0'),
    ('ReLU activation', 'Sigmoid activation'),
])
def test_unsupported_patterns_leave_graph_unchanged(fuse, old, new):
    graph = GRAPH.replace(old, new)
    assert fuse(graph) == (0, graph)


def test_shared_sum_cannot_be_removed(fuse):
    graph = GRAPH.replace('5 5', '6 6') + 'BinaryOp consumer 2 1 sum out0 tail\n'
    assert fuse(graph) == (0, graph)


def test_weighted_layers_and_bytes_order_are_preserved(fuse):
    graph = GRAPH.replace('5 5', '6 6') + 'Convolution weighted 1 1 out0 final 0=8 1=3 5=1 6=216\n'
    count, output = fuse(graph)
    assert count == 1
    assert output.splitlines()[-1] == graph.splitlines()[-1]
