# Multi-Domain Semantic Segmentation via the Principle of Rate Reduction
TODO: Tune optimizer type, lr, and scheduling (combine with below for auto-tuner?)

TODO: Auto run ending conditions: large val vs train acc gap, stagnating loss or acc

TODO: cosine similarity of dataset samples

TODO: how can we measure the sufficiency of z dimension? Might help in understanding 

## Experiments & Reports

### Segmentation Consensus
TODO: Try reducing class imbalance by weighting by 1/class_freq

TODO: confusion matrix visualization

TODO: visualize class differentiation over training, unet seems to keep improving after learning to segment
bg perfectly

Expect that the learned Z subspaces will quickly differentiate BG from the rest and then only have notably
large loss magnitudes for the other class samples. Lets try proving this experimentally / mathematically?

## Scratch Space
```python
def m(*d):
    n = 1
    for i in d: n *=i
    return torch.arange(n).view(*d) + 0.
```