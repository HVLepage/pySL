import numpy as np
from src.input.trace1D import trace1D
from src.input.antisymmetrise import antisymmetrise
import math

def exportNpTrace(PsiOut,params,step,Filename="wave"):
    n_zero = math.log10((float)(params["TimeSteps"] / params["PrintStep"]) + 1) + 1
    count = int(step / params["PrintStep"])
    index = str(count).zfill((int(n_zero))) #fill string with zeros until n_zero characters long
    
    
    if (step >= 0):
        OutputFileName = "./out/" + params["SimName"] + "/waves/wave_iteration_" + index + ".txt" 
    else:
        OutputFileName = "./out/" + params["SimName"] + "/" + Filename + ".txt"
    if params["antisymmetrise"]:    
        PsiOut.wave=antisymmetrise(params,PsiOut.wave.real,PsiOut.wave.imag)
        norm=PsiOut.norm()
        
        PsiOut.wave /= np.sqrt(norm)
    sumxs=trace1D(params,PsiOut.wave.real,PsiOut.wave.imag)
    sumxs=np.transpose(sumxs)
    
    np.savetxt(OutputFileName,sumxs,fmt='%.6e')
    # if step ==0:
    #      OutputFileName = "./out/" + params["SimName"] + "/" +"wave_full" + ".txt"
    #      with open(OutputFileName, 'w+') as OutputFile:
    #         for x in np.nditer(PsiOut): #efficient iterating
    #             OutputFile.write("{:e} {:e} \n".format(x.real, x.imag)) #exponent
  