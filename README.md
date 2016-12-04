Dynamic Coattention Networks For Question Answering
===========================================================

Tensorflow implementaion of [Dynamic Coattention Networks](https://arxiv.org/abs/1611.01604).

This repo contains work on progress.


Requirements
--------------

- Python 3.5.2
- TensorFlow r0.11
- spaCy 1.2.0


Data
-----

Retrieve json training files from [SQuAD](https://rajpurkar.github.io/SQuAD-explorer/).

```dataset.py``` contains code to create a dataset and vocab for training network.


Training the network
-----------------------

In order to train the network, execute
```
python main.py --mode train
```

Credit
-------

Modified codes are originally from [textsum](https://github.com/tensorflow/models/tree/master/textsum).
