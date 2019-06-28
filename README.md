# README #

[GitHub repository](https://github.com/gwenniger/multi-hare).[arXiv paper](https://arxiv.org/abs/1902.11208)

### What is this repository for? ###

This repository implements Multidimensional Long Short-Term Memory Recurrent Neural Networks for handwriting recogntion
using PyTorch.

Features include:

* Efficient Multi-directional Multi-Dimensional LSTM, using various optimizations described in the paper
  "No Padding Please: Efficient Neural Handwriting Recognition" 
* Pytorch data preparation for the ICDAR dataset
* Automatic generation of a dataset for testing during development, based on 
  variable length MNIST digit sequences.
* An adaptation of DataParallel which works with lists of variable-sized examples

While allured by the [March Hare](https://en.wikipedia.org/wiki/March_Hare)  from Alice in Wonderland,
this software should not be confused by it. 

### Setup and use ###
For running the pipeline code, you may need to execute:
" export PYTHONPATH='.' "in the console where you are running the script, to avoid "no module named X"
type of errors.
Furthermore, this repository is written for python version3, you may need to 
use a virtual environment to use it.

### External libraries ###
This repository uses several external libraries:
1. *ctcdecode*:  https://github.com/parlance/ctcdecode. 
This library implements a beam-search decoder with KenLM language model support.
2. PyTorch bindings for warp-ctc: https://github.com/SeanNaren/warp-ctc
This library provides PyTorch bindings for the fast CTC code developed by Baidu Research 
available from: https://github.com/baidu-research/warp-ctc.
3. KenLM: https://github.com/kpu/kenlm

<!---

* Quick summary
* Version
* [Learn Markdown](https://bitbucket.org/tutorials/markdowndemo)

### How do I get set up? ###

* Summary of set up
* Configuration
* Dependencies
* Database configuration
* How to run tests
* Deployment instructions

### Contribution guidelines ###

* Writing tests
* Code review
* Other guidelines

### Who do I talk to? ###

* Repo owner or admin
* Other community or team contact

-->
