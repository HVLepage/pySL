The parameters are set up for GaAs.

There are three parameters files in the parameters folder. When using them rename them as parameters.py and place in src/input. The main differences are the potentials present and whether the wave function is imported/formed manually.

parameters_gaas_1peigen is for getting the 1 particle eigenstates (non-interacting system). Running pySL.py will put the eigenstate  wave.txt in out/Particle_in_a_box_1D.

parameters_gaas_3peigen is for forming 3 particle eigenstate, in out/Particle_in_a_box_1D/npfrom1p are the single particle eigenstates, wave1 is the central particle, wave 2 the right particle and wave 3 the left (behind barrier)
When the simulation finishes, the eigenstate will be saved as wave_final.txt in out/Particle_in_a_box_1D (it will be the 200MB file)

parameters_gaas_qnc is for the nonadiabatic quantum newton's cradle evolution, the 3 particle eigenstate should be placed in src/input as wave.txt. The traced over wave functions are saved in out/Particle_in_a_box_1D/waves



If starting a simulation from scratch, you would generate the 3 single particle eigenstates using parameters_gaas_1peigen.py. The simulation would have to be run 3 times and each time wave.txt needs to be saved and placed in
out/Particle_in_a_box_1D/npfrom1p in preparation for the next step (as wave1, wave2, wave3)

Then to find the 3 particle eigenstate you would need the 3 wave files in out/Particle_in_a_box_1D/npfrom1p and parameters_gaas_3peigen, the simulation can then be run and wave_final.txt needs to be saved for the next step.

wave_final.txt then needs to be copied to src/input and renamed as wave.txt,  parameters_gaas_qnc must then be used. Traced over wavefunction is saved in out/Particle_in_a_box_1D/waves for the next step.

Then animatenp_sep.py (or animation_2Dtrace_separate.py if traced over just one particle) can be separately run to animate the above waves or expectation.py to plot the expectation as a function of time 
(requires out/Particle_in_a_box_1D/waves to be populated).


The first step has already been done however, so need to start from forming 3 particle eigenstate. Should be as easy as copying parameters_gaas_3peigen as parameters.py into src/input and running pySL.py (for a given sigma parameter)
Animate the files in out/Particle_in_a_box_1D/waves to ensure that the eigenstate has been found.
All parameters have a comment describing their function so should be easy to use.
In kernel.cu, should change #define Npdim 1  to however many particles there are i.e. either 1 or 3. Won't make much of a difference if you forget, but this ensures the wave function is calculated correctly at domain boundaries.
If don't want spin effects then comment out the code in kernel.cu that says //comment the below code out if don't want spin (there are 2 bits to comment out)


Extra plotting files etc. in misc folder. 3D plotting functionality seems to work but not fully tested, requires whole wavefunction to be exported during time evolution.
Animation is faster for some reason when done separately.