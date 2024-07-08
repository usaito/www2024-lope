# Long-term Off-Policy Evaluation and Learning

This repository contains the experiment code used to perform the synthetic experiments in [Long-term Off-Policy Evaluation and Learning](https://dl.acm.org/doi/abs/10.1145/3589334.3645446) by [Yuta Saito](https://usait0.com/en/), Himan Abdollahpouri, Jesse Anderton, Ben Carterette, and Mounia Lalmas. The paper was accepted and presented at [The Web Conference 2024](https://www2024.thewebconf.org/).

## Abstract

Short- and long-term outcomes of an algorithm often differ, with damaging downstream effects. A known example is a click-bait algorithm, which may increase short-term clicks but damage long-term user engagement. A possible solution to estimate the long-term outcome is to run an online experiment or A/B test for the potential algorithms, but it takes months or even longer to observe the long-term outcomes of interest, making the algorithm selection process unacceptably slow. This work thus studies the problem of feasibly yet accurately estimating the long-term outcome of an algorithm using only historical and short-term experiment data. Existing approaches to this problem either need a restrictive assumption about the short-term outcomes called surrogacy or cannot effectively use short-term outcomes, which is inefficient. Therefore, we propose a new framework called Long-term Off-Policy Evaluation (LOPE), which is based on reward function decomposition. LOPE works under a more relaxed assumption than surrogacy and effectively leverages short-term rewards to substantially reduce the variance. Synthetic experiments show that LOPE outperforms existing approaches particularly when surrogacy is severely violated and the long-term reward is noisy. In addition, real-world experiments on large-scale A/B test data collected on a music streaming platform show that LOPE can estimate the long-term outcome of actual algorithms more accurately than existing feasible methods.


## Citation
```
@inproceedings{saito2024long,
  title={Long-term Off-Policy Evaluation and Learning},
  author={Saito, Yuta and Abdollahpouri, Himan and Anderton, Jesse and Carterette, Ben and Lalmas, Mounia},
  booktitle={Proceedings of the ACM on Web Conference 2024},
  pages={3432--3443},
  year={2024}
}
```
