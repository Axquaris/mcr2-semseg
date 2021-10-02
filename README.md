# Multi-Domain Semantic Segmentation via the Principle of Rate Reduction
[Presentation](https://docs.google.com/presentation/d/16jhEIYlq5gfa9y-dGH3DhlT9N-pKZ9w5CmFx15JuLDg/edit#slide=id.p)

[Conference-Style Report](https://github.com/Axquaris/mcr2-semseg/blob/main/mcr2-semseg.pdf)

Abstract:<br/>
We seek to further validate the objective and theoretical principles introduced in Maximal Coding Rate Reduction (MCR2) by scaling it to the task of semantic segmentation in real world environments, where the number of labels per inference and size of datasets is significantly greater than the classification datasets STL10 and CIFAR100. Additionally, we show that applying a learned mask to filter high-frequency background pixels can greatly improve both the empirical and qualitative performance of MCR2. Finally, while the MCR2 objective does not appear to improve in learning domain agnostic features to assist with domain generalization, it improves upon standard supervised learning losses when the input space contains samples from many varied domains.

<!--
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
-->
