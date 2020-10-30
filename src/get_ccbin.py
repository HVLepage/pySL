import platform
import os

def get_ccbin():

    # Compatibility with Windows
    if os.name == 'nt': 
        ccbin = 'cl.exe'

    # GPU-S6 has g++ version 9. CUDA 10.2 only supports g++ version 8. Use a specific compiler instead.
    if platform.node() == 'gpu-s6.tfm.phy.private.cam.ac.uk': 
        ccbin = 'cuda-g++'
    
    return ccbin