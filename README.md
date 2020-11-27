# mcr2-semseg

```python
def m(*d):
    n = 1
    for i in d: n *=i
    return torch.arange(n).view(*d) + 0.
```