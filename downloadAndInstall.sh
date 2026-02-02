# A script to install the required submodules, run the installation steps and test the installed software
# Tested for ubnutu linux. Requires boost-dev to be installed by the package manager
# Also requires anaconda to be installed 

# Step 0. Create a new conda environment for the project
conda create --name multi-hare2
conda activate multi-hare


# Step 1. Get the submodules, see https://stackoverflow.com/questions/10168449/git-update-submodules-recursively
git submodule update --init --recursive

# Step 2: install pip and python/pytorch requirements
conda install pip
pip install -r requirements.txt

# Step 3: install warp-ctc
cd libraries/warp-ctc/
mkdir build
cd build
cmake ..
make -j8 -d
cd ../../
python3 setup.py install

# Step 4: install ctcdecode
# # Go into the ctcdecode folder
cd libraries/ctcdecode
# Go into the kenlm folder and checkout the ctcdecode-fix branch
cd third_party/kenlm
git checkout ctcdecode-fix
# Back up to the ctcdecode folder
cd ../..
# To find the installed version of torch, we compile without build isolation
pip install -v --no-build-isolation .
