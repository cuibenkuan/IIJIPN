# IIJIPN

The source code of "Intra-graph and Inter-graph Joint Information Propagation Network]{Intra-graph and Inter-graph Joint Information Propagation Network with Third-order Text Graph Tensor for Fake News Detection".

Requirements:

Python 3.6
Tensorflow/Tensorflow-gpu 1.12.0
Scipy 1.5.1

Usage:

Download pre-trained word embeddings glove.6B.300d.txt

To use your own dataset, put the text file under data/corpus/ and the label file under data/ as other datasets do.

Preprocess the text by running remove_words.py [--dataset DATASET]

Build graphs from the datasets by running python build_graph.py [--dataset DATASET]

Start training and inference by  running  python train.py [--dataset DATASET]
