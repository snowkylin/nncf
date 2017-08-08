# Neural Network-based Collaborative Filtering

TensorFlow implementation of paper _On Sampling Strategies for Neural Network-based Collaborative Filtering_ by Chen, Ting, et al. A neural network-based recommender systems with several sampling strategies (in progress).

## Prerequisites

* Python 3.5
* TensorFlow 1.2.0
* Keras
* Networkx
* NumPy

## Setup

* Download pre-trained GloVe word vectors from https://nlp.stanford.edu/projects/glove/ (6B tokens). Unzip it and copy `glove.6B.50d.txt` to `data/word_vectors`
* Download CiteUlike data and unzip it to `data/citeulike`.
* Run `script/build_word_vectors_from_glove.py` to build the word vector dictionary from pre-trained word vectors.
* Run `main.py` to start training.

## Sampling Strategies 

There are 4 sampling strategies provided in this program.

* Negative Sampling (`--sampling_strategy negative`)
* Stratified Sampling (`--sampling_strategy stratified_sampling`)
* Negative Sharing (`--sampling_strategy negative_sharing`)
* Stratified Sampling with Negative Sharing (`--sampling_strategy SS_with_NS`)

You can specify sampling startegies via flags, such as:

```
python main.py --sampling_strategy negative
```

## References

- Chen, Ting, et al. "[On Sampling Strategies for Neural Network-based Collaborative Filtering.](https://arxiv.org/abs/1706.07881)" _Proceedings of the 23th ACM SIGKDD international conference on Knowledge discovery and data mining. ACM_. 2017.
- Chen, Ting, et al. "[Joint Text Embedding for Personalized Content-based Recommendation.](https://arxiv.org/abs/1706.01084)" _arXiv preprint arXiv:1706.01084_ (2017).
- Kim, Yoon. "[Convolutional neural networks for sentence classification.](https://arxiv.org/abs/1408.5882)" _arXiv preprint arXiv:1408.5882_ (2014).
