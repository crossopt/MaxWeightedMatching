# MaxWeightedMatching
A python3 implementation of a Poly(n) algorithm to find the maximum weighted matching in a non-bipartite graph.

## How to run:
#### randomized small tests:
python3 test_matching.py
#### finding a matching:
python3 matching.py
#### demo test:
A test with 3 vertices and 3 edges in a cycle.

The resulting matching has weight 3: the match for vertex 0 is 1, the match for vertex 1 is 0, vertex 2 has no match.

Input:
3 3
0 1 3
1 2 2
2 0 1

Output:
3
1 0 None


## Input format
N M  
from<sub>0</sub> to<sub>0</sub> weight<sub>0</sub>  
...  
from<sub>M - 1</sub> to<sub>M - 1</sub> weight<sub>M - 1</sub>  
<EOF>  
Where N is the number of vertices, M is the number of edges. from<sub>i</sub>/to<sub>i</sub> vertices are from 0 to N - 1. 
