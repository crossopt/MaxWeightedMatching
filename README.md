# MaxWeightedMatching
A python3 implementation of a Poly(n) algorithm to find the maximum weighted matching in a non-bipartite graph.

## How to run:
#### randomized small tests:
python3 test_matching.py
#### finding a matching:
python3 matching.py

## Input format
N M  
from<sub>0</sub> to<sub>0</sub> weight<sub>0</sub>  
...  
from<sub>M - 1</sub> to<sub>M - 1</sub> weight<sub>M - 1</sub>  
<EOF>  
Where N is the number of vertices, M is the number of edges. from<sub>i</sub>/to<sub>i</sub> vertices are from 0 to N - 1. 
