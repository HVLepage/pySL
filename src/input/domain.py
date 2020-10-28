################################################################################################################################
# Define simulation domain
################################################################################################################################

# This function returns an array with the same dimensions as the wave function.
# Values on the boundary of the simulation domain are set to false, everything inside is set to true.
# Although this method uses more GPU memory, it reduces the number of operations required per iteration.
def DirichletBC(params):

    N = params["N"] # Total size of the wave function

    