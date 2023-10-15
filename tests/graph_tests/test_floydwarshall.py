import pytest
import numpy as np
from typing import List
from cualgo import graph as cg
from .test_data import shortestpath_n1000 as big_graph

def test_input_badtype():
    with pytest.raises(TypeError, match='incompatible function arguments'):
        cg.floydwarshall([['a', 'b'], ['c', 'd']])
    with pytest.raises(TypeError, match='incompatible function arguments'):
        cg.floydwarshall([[0, 1], [2, 'kaboom']])

def test_input_badsize():
    with pytest.raises(TypeError):
        cg.floydwarshall([0, 0]) #should be 2D array
    with pytest.raises(ValueError, match="square"):
        cg.floydwarshall([[0, 0]]) #should be 2D square

def test_output_type():
    assert      cg.floydwarshall([[1]]) == [[1]]
    assert type(cg.floydwarshall([[1]])[0][0]) == int, "Invoking C implmentation of <int> should returns [[int]]"
    assert      cg.floydwarshall(np.array([[1]], dtype=np.float64)) == [[1]]
    assert type(cg.floydwarshall(np.array([[1]], dtype=np.float64))[0][0]) == float, "Invoking C implmentation of <double> should returns [[float]]"

def to_float(int2d: List[List[int]]) -> List[List[float]]:
    return [[float(cell) for cell in row] for row in int2d]

@pytest.mark.parametrize("input, output", [
    #case1
    ([[0  , 7  , 999, 8],
      [999, 0  , 5  , 999],
      [999, 999, 0  , 2],
      [999, 999, 999, 0]]
    ,[[0, 7, 12, 8], [999, 0, 5, 7], [999, 999, 0, 2], [999, 999, 999, 0]])
    ,
    #case2
    ([[0, 5, 999, 10],
      [999, 0, 3, 999],
      [999, 999, 0, 1],
      [999, 999, 999, 0]]
    ,[[0, 5, 8, 9], [999, 0, 3, 4], [999, 999, 0, 1], [999, 999, 999, 0]])
    ,
    #big graph
    (big_graph.input, big_graph.output)
])
def test_graph(input, output):
    #2 implementations in C for int/double respectively
    assert cg.floydwarshall(input) == output
    assert cg.floydwarshall(to_float(input)) == to_float(output)