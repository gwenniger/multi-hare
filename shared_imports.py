from __future__ import unicode_literals, print_function, division
import torch
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms

from io import open
import unicodedata
import string
import re
import random

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

use_cuda = torch.cuda.is_available()

exec(open("util/Timing.py").read())
exec(open("util/ImageVisualization.py").read())

# Shared definitions

