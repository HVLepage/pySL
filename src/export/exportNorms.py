
def exportNorms(params, wave, step):
    OutputFileName = "./out/" + params["SimName"] + "/norms.txt"
    
    mode = "w+"
    
    if step == 0:
        mode = "w"
        
    with open(OutputFileName, mode) as OutputFile:
            OutputFile.write("{:e} \n".format( wave.norm() ))
    
    
    