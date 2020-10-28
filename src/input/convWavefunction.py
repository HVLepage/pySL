from src.input.wavefunction import wavefunction
import math
import numpy as np
import pycuda.driver as cuda
import reikna.fft as rfft
import reikna.cluda as cluda
import pycuda.gpuarray as gpuarray
import time


class convWavefunction(wavefunction):
    
    
    def convFFT(self, psi, V, mod, kernelInds, t, params):
        N = np.int32(params["N"])
        psi.wave_gpu.get(self.wave_gpu)
        self.fft(0)

        blockSize = 256
        numBlocks = math.ceil(N / blockSize)
        
        Nm1 = np.float32(N ** -1)
        Axpy = mod.get_function("Axpy")
        
        Axpy(psi,  V,  N,  t,  kernelInds,  Nm1, block=(blockSize,1,1), grid=(numBlocks,1))
        
        self.fft(1)
    
    def __init__(self, psi, params):
         self.dV = psi.dV
         self.wave_gpu=gpuarray.to_gpu(psi.wave)
         self.api = cluda.cuda_api()
         self.thr = self.api.Thread.create()
         fft_h = rfft.FFT(self.wave)
         self.fftc=fft_h.compile(self.thr)