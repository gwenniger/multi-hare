# README #

[GitHub repository](https://github.com/gwenniger/multi-hare).[arXiv paper](https://arxiv.org/abs/1902.11208)

### What is this repository for? ###

This repository implements Multidimensional Long Short-Term Memory Recurrent Neural Networks for handwriting recognition
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

### License notes ###

Most of the software was developed as part of my EDGE fellowship at Dublin City University, and the 
rights of the software are owned by Dublin City University. See LICENSE for the details of this license 
for **non-commercial** use of the software. Importantly, if you intend on using the Software for commercial purposes, 
you must obtain a commercial licence from the DCU Invent. 
For further information please contact info@invent.dcu.ie (see LICENCSE). 
The software makes use of certain third-party libraries, see Third_Party_Software_Notes.txt.
A small number of source files that were written after my EDGE project and/or developed in other contexts 
is released with the Apache License 2.0. The intention is that newly written and future added  source files 
will be released with the Apache License 2.0 as well.
 

### Setup and use ###
For running the pipeline code, you may need to execute:
" export PYTHONPATH='.' "in the console where you are running the script, to avoid "no module named X"
type of errors.
Furthermore, this repository is written for python version3, you may need to 
use a virtual environment to use it.

The main file for running experiments is: "modules/train_multi_dimensional_rnn_ctc.py".
The file "modules/opts.py" has been used for defining a range of options for training and evaluation.
This pattern was copied from OpenNMT-py (https://github.com/OpenNMT/OpenNMT-py)
which similarly uses the "opts" file extensively to define configuration options. The configuration options
are typically specified as command arguments. A typical use case is to make a shell script for an experiment
which contains the call of train_multi_dimensional_rnn_ctc.py and the required arguments.

There is a special dataset called "variable_length_mnist" which can be used for testing the software.
It generates random sequences of digits of variable length from the set of handwritten digit examples 
from the MNIST dataset. This is useful for testing, as it is much faster than training on data from ICDAR 
but still captures most though not all of the  properties of a real handwriting recognition task (i.e. ICDAR).

Language Models

The software uses KenLM language models, an interface for training language models is available at 
language_model/kenlm_interface.py. For replicating the results on the IAM dataset, the use of a strong language
model trained on the LOB and Brown corpus is important. However, the London_Oslo_Bergen (LOB) corpus, available from:
http://ota.ox.ac.uk/desc/0167 is in a somewhat archaic format. It therefore needs to be converted to be compatible with IAM. 
The file <em>monolingual_data_preprocessing/lob_original_preprocessor.py</em> takes care of this.
The BROWN corpus is taken from NLTK: http://www.nltk.org/nltk_data/. It needs much less preprocessing, the file 
to do this preprocessing  is <em>monolingual_data_preprocessing/brown_corpus_preprocessor.py</em>.

 

### External libraries ###
This repository uses several external libraries:
1. *ctcdecode*:  https://github.com/parlance/ctcdecode. 
This library implements a beam-search decoder with KenLM language model support.
2. PyTorch bindings for warp-ctc: https://github.com/SeanNaren/warp-ctc
This library provides PyTorch bindings for the fast CTC code developed by Baidu Research 
available from: https://github.com/baidu-research/warp-ctc.
3. KenLM: https://github.com/kpu/kenlm

### Citation ###

@article{DBLP:journals/corr/abs-1902-11208,
  author    = {Gideon Maillette de Buy Wenniger and Lambert Schomaker and Andy Way},
  title     = {No Padding Please: Efficient Neural Handwriting Recognition},
  journal   = {CoRR},
  volume    = {abs/1902.11208},
  year      = {2019},
  url       = {http://arxiv.org/abs/1902.11208}
}

Note that the paper has also been accepted and presented at ICDAR 2019
https://icdar2019.org/list-of-accepted-papers/.

In addition to citing the paper, please see point 4.1 of the license 
concerning how to credit Science Foundation Ireland for partly funding 
the development of this software.

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
