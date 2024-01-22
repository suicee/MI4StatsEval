## Evaluation of Mutual Information between Summary Statistics and Physical Parameters

This repository provides a detailed implementation of the paper [Evaluating Summary Statistics with Mutual Information for Cosmological Inference](https://arxiv.org/abs/2307.04994), along with other experiments and codes regarding this topic.

### Basic Idea

In this research, we study the application of mutual information to assess the effectiveness of various summary statistics in cosmological inference. Our approach utilizes the Barber-Agakov lower bound and normalizing flow based variational distributions. For more details, please refer to our paper.

### Experiments

We conduct two experiments to demonstrate our method, using BA lower bound:

* [Use MI to verify power spectrum is the optimal statistics for CMB-like Gaussian random fields](MI_Estimation_CMB.ipynb)
* [Compare different summary statistics in an EoR inference task, where the field is highly non-Gaussian](MI_Estimation_21cm.ipynb)

We show that MI can correctly assess the effectiveness of different summary statistics in a given inference task.

### Other MI estimators

Additionally, we have tried other mutual information estimators from papers such as [On Variational Bounds of Mutual Information](https://arxiv.org/abs/1905.06922), [Understanding the Limitations of Variational Mutual Information Estimators](https://arxiv.org/abs/1910.06222), [Estimating Mutual Information](https://arxiv.org/abs/cond-mat/0305641).

The `other_MI_estimators` folder currently consists of two notebooks, each utilizing a different method to evaluate the MI in the second experiment mentioned above:

1. `MI_Estimation_Smile.ipynb`: Here, we conduct experiments with the [SMILE](https://arxiv.org/abs/1910.06222) estimator, which is a variance-reduced version of [MINE](https://arxiv.org/abs/1801.04062). This method employs the Donsker-Varadhan (DV) lower bound.
2. `MI_Estimation_KSG.ipynb`: In this notebook, we explore the usage of the [KSG](https://arxiv.org/abs/cond-mat/0305641) estimator, a K-nearest neighbor-based MI estimation method. However, since KSG is applicable only to low-dimensional data, we integrate it with a further compression operation for the summaries.
3. `MGC_CMB.ipynb` and `MGC.ipynb`: In these two notebooks we explore the usage of the [Multiscale Graph Correlation (MGC)](https://hyppo.neurodata.io/api/generated/hyppo.independence.mgc#hyppo.independence.MGC) to evaluate the correlation between different summaries and parameters.

In all of our experiments, we have consistently observed that training a well-performing model is crucial for obtaining reliable results. Enhancing the training procedure with more sophisticated techniques can potentially improve the results obtained using all of these methods. If you have any ideas, comments or questions, please feel free to reach out to Ce Sui at suic20@mails.tsinghua.edu.cn.
