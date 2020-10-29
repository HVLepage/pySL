################################################################################################################################
# GPU hardware info
################################################################################################################################

import sys
import pycuda.driver as cuda
import pycuda.autoinit

def print_gpu_info():
    print("\n----------------------------------------------------------------")
    print("Python version: " + sys.version)
    print("pyCUDA version: " + pycuda.VERSION_TEXT)
    print("CUDA runtime version: %d.%d.%d"%cuda.get_version())
    print("----------------------------------------------------------------")
    print("CUDA capable device count:", cuda.Device.count())
    if (len(sys.argv) > 1):
        print("Target CUDA device: ",str(sys.argv[1]))
    else:
        print("Target CUDA device: 0")

    for devicenum in range(cuda.Device.count()):
        device=cuda.Device(devicenum)
        dev_name=device.name()
        dev_cc=device.compute_capability()
        dev_mem=device.total_memory()
        # attrs=device.get_attributes()

        print("\n===Attributes for device %d"%devicenum, ":",dev_name)
        print("   Compute capability: %d.%d"%dev_cc)
        print("   Total global memory: %.1fGB"%(dev_mem/1073741824))
        # for (key,value) in attrs.items():
        #     print("%s:%s"%(str(key),str(value)))