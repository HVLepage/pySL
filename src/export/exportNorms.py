
def exportNorms(params, wave, step):
    OutputFileName = "./out/" + params["SimName"] + "/norms.txt"
    
    mode = "a"
    
    if step == 0:
        mode = "w"
        
    with open(OutputFileName, mode) as OutputFile:
            OutputFile.write("{:.9e} \n".format( wave.norm() ))
    
    
    