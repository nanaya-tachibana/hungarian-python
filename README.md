# hungarian-python
Python Implementation of Hungarian Algorithm

## Requirement

- numpy
- numba

## Usage

```python
from hungarian import Hungarian

hg = Hungarian()
assignment, total_cost = hg.execute(cost_matrix)
```
