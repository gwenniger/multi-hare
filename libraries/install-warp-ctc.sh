#!/bin/bash

cd warp-ctc
mkdir build; cd build
cmake .. -DCMAKE_POLICY_VERSION_MINIMUM=3.5
make
cd ..
#cd warp-ctc/warpctc_pytorch/core
#mkdir build; cd build
#cmake ../../..
#make
#cd ../../..



echo "Next install the binding...:"
cd pytorch_binding
#cd warp-ctc/pytorch_binding
# See: https://github.com/pytorch/fairseq/issues/59
#pip3 install cffi
python setup.py install

