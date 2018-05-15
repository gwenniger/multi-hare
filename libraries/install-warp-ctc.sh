#!/bin/bash

#cd warp-ctc
#mkdir build; cd build
#cmake ..
#make


echo "Next install the binding...:"

cd warp-ctc/pytorch_binding
# See: https://github.com/pytorch/fairseq/issues/59
#pip3 install cffi
python setup.py install

