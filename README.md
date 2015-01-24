# hungarian-python
Python Implementation of Hungarian Algorithm

## Requirement

- numpy
- numba

## Usage

```python
from hungarian import Hungarian

N = 10  # problem size
cost_matrix = np.random.random((N, N))
h = Hungarian(N)
assignment, total_cost = h.execute(cost_matrix)
```
