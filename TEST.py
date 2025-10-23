# -*- coding: utf-8 -*-
"""
Created on Tue Sep 23 15:37:42 2025

@author: Aleksander
"""

import sys
import os
import math
import numpy as np
import time
import matplotlib.pyplot as plt
import tracemalloc
import humanize

# Set target CUDA device. Must be an integer. Currently no error checking.
if (len(sys.argv) > 1):
  print("\nTarget CUDA device: ",str(sys.argv[1]))
  os.environ['CUDA_DEVICE'] = str(sys.argv[1])

# Import pyCUDA modules
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray

# Import project modules
import src.get_gpu_info as gpuinfo
from src.get_ccbin import get_ccbin
from src.input.parametersQubits import getparams
from src.input.wavefunction import wavefunction
from src.fft.convWavefunction import convWavefunction
from src.export.wavefunctionOut import wavefunctionOut
from src.export.exportTrace import exportTrace
from src.input.domain import domain
from src.index.kernelInds import kernelInds

from src.input.potential import potential
from src.export.makepath import makepath
from src.export.exportParameters import exportParameters

import reikna.cluda as cluda

# Built-in plotting
from src.plot.animate1D import animate1D
from src.plot.plot1Dtrace import plot1Dtrace
