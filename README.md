CURRENTLY ONLY TESTED RSPACESOLVER FOR TIME EVOLUTION, KSPACE MAY NOT WORK

Changes 14/03/22
This version time evolves a quantum Newton's cradle with 3 particles. Less stable version than previous as haven't made sure the previous examples still work bug free.
Requires the initial wavefunction to be placed in src/input but the file is too large for github, can be found here https://drive.google.com/file/d/1ISP9183brwLQxUJ5P3p7TQISPlKMNjQ4/view?usp=sharing. Should be renamed wave.txt

1. Now allocate Coulomb potential on the GPU, orders of magnitude faster.
2. Improved the code for tracing over N particles in 1D, much faster but ideally this should be put on GPU.
3. Added the option to just export the trace of N particles.
4. Added a file npfrom1p.py to make N particle wavefunction as product of 1 particle wavefunctions, done on the GPU.
5. Added option for potential step which makes for a nice example.
6. Various smaller changes throughout.

Changes 25/02/22:
1. Removed the domain which meant that the wavefunction wasn't updated on boundaries. Replaced with code in kernel.cu in the rSpaceSL
function which ensures that only lattice sites within the domain are used to update the wavefunction on the boundaries. IMPORTANT: must 
change Npdim at top of kernel.cu file to the number of particles * number of dimensions used in your simulation. If you don't do this,
the results of the simulation won't be too different but it does prevent accessing unassigned memory and it corrects the nearest neighbours
for sites on the boundary.

2. Completely changed the CoulombPotential function in kernel.cu, currently creates a softened potential in 1/2 dimensions. The softening
parameter required is calculated in approxSoftening.py.

3. In potential.py corrected the array which holds the pairs of interacting particles. Also corrected the code which enables a different
single particle potential to be added to each particle. Currently allows a sinusoidal potential (1D) or a harmonic potential (1,2,3D) to be added, have to be
careful with this as the code isn't written very generally.

4. Added defWavefunction.py file which allows an arbitary wavefunction to be inputted. Currently only set up for Gaussian wave packets.

5. Added trace1D.py which is used to sum over the positions of other particles in 1D for the purpose of plotting.

6. Added several animation files and plots.

7. Prints some information about the stability of the simulation e.g. the max potential that can be used.

8. Various smaller changes throughout, should be commented.

Bugs:
1. For some reason the animiations run much faster separately, i.e. run the simulation then run one of the animation files with _sep at the end of the file name.
The animations can be run in the main simulation by uncommenting the code on lines 260.

Instructions:

Before running a simulation make the following changes to parameters.py

1. Choose your number of particles by changing Np
2. If using the eigensolver to set the wavefunction, choose which eigenstates you want with InputStates
3. If want to add a 1 particle potential for each particle, then set 1pPotential to True as well as the type of potential you want with 1pPotentialType
4. If want to use eigensolver set defWavefunction to False, if want to use arbitrary wavefunction set to true. Choose the type with wavefunctionType (Gaussian only currently)
5. If using a Gaussian wavepacket then set the parameters.
6. Change mass and charge arrays as appropriate.
7. Choose dt and number of timesteps etc. Should be fine as is.
8. Choose size of box with Xa, Xb etc.
9. Choose number of lattice sites with Nx etc. Set a particular dimension to 1 if want to do a simulation in a lower dimension. THEN CHANGE Npdim IN KERNEL.CU AS DESCRIBED ABOVE
10. If using a harmonic or sinusoidal single particle potential then set the parameters.
11. Choose the width of the Gaussians in the non-dynamic dimensions with sigSoftz (just change this if have 2 dynamic dimensions) and sigSofty. Have only tested 1D version i.e. setting both.

The program should be set up such that parameters not used in your particular simulation can be left as they are (i.e. don't have to be set to 0)

Examples:

Particle in a 1D box, in superposition of eigenstates:
1. Set Np to 1
2. Choose eigenstates
3. Set 1pPotential to False
4. Set defWavefunction to False
5. Set Xa to -100 and Xb to 100
6. Set Nx to 201 and Ny, Nz to 1
7. Set Npdim in kernel.cu to 1
Output: Animation of mod^2 plotted as a function of x, evolving with time.
Comments: Eigenvalues are accurate if calculate with length of box +2dx

Particle in a 1D box, propagating Gaussian wavepacket:
1. Set Np to 1
2. Set 1pPotential to False
3. Set defWavefunction to True and wavefunctionType to Gaussian
4. Set muxG to 0 (Gaussian centred at 0), sigmaxG to 200, kGx to 2pi/100
5. Set Xa to -1000 and Xb to 1000
6. Set Nx to 201 and Ny, Nz to 1
7. Set Npdim in kernel.cu to 1
Output: Animation of mod^2 plotted as a function of x, evolving with time.
Comments: Parameters have been chosen such that there is little dispersion.

Particle in a 1D harmonic potential, eigenstates.
1. Set Np to 1
2. Choose eigenstates
3. Set 1pPotential to True and set 1pPotentialType to Harmonic
4. Set defWavefunction to False
5. Set Xa to -100 and Xb to 100
6. Set Nx to 201 and Ny, Nz to 1
7. Set wX to 0.003*1.157676458e2 (2nd number is the conversion factor, wX is then divided by the same conversion factor so setting it like this means that omega=0.003 when calculating eigenvalues)
8. Set 1st value of xEqms array to 0.
9. Set Npdim in kernel.cu to 1
Output: Animation of mod^2 plotted as a function of x, evolving with time.
Comments: Eigenvalues = (n+0.5) * 0.003 where n=0,1,2...

Particle in a 2D box, propagating Gaussian wavepacket.
1. Set Np to 1
2. Set 1pPotential to False
3. Set defWavefunction to True and wavefunctionType to Gaussian
4. Set muxG to 0 and muyG to 0. sigmaxG to 200, sigmayG to 200, kGx to 2pi/100, kGy to 2pi/100
5. Set Xa to -1000, Xb to 1000, Ya to -1000, Yb to 1000
6. Set Nx and Ny to 81
7. Set Npdim in kernel.cu to 2
Output: nothing after main program finishes, run animation2D_sep.py and mod^2 wavefunction is plotted as a function of x and y. Animation is also saved in out/Particle_in_a_box_1D (yes the folder name should be changed at some point)

Two interacting particles in a 1D box.
1. Set Np to 2
2. Set 1pPotential to False
3. Set defWavefunction to True and wavefunctionType to Gaussian
4. Set muxG to -400, mux2G to 400, sigmaxG to 200, sigmax2G to 200, kGx to 2pi/100, kGx2 to -2pi/100
5. Set mass and charge arrays (must have 2 elements)
6. Set Xa to -1000, Xb to 1000
7. Set Nx to 81, Ny to 1
8. Set Npdim in kernel.cu to 2
9. Set sigSofty, sigSoftz to 240 if want tunnelling, 10 if want strong repulsion

Output: Before time propagation begins, the softened Coulomb potential will be plotted as well as 2 different plots of the initial wavefunction. Close these and propagation will begin.
Nothing will be plotted at the end. Either run animation2p_sep.py if you want an x1-x2 plot or animation2D_sum_sep.py if you want a 1D plot of each summed over wavefunction.

Three interacting particles in a 1D box.
1. Set Np to 3
2. Set 1pPotential to False
3. Set defWavefunction to True and wavefunctionType to Gaussian
4. Set muxG to -400, mux2G to 400, mux3G to 0, sigmaxG to 200, sigmax2G to 200, sigmax3G to 200, kGx to 2pi/100, kGx2 to -2pi/100, kGx3 to 0
5. Set mass and charge arrays (must have 3 elements)
6. Set Xa to -1000, Xb to 1000
7. Set Nx to 81, Ny to 1
8. Set Npdim in kernel.cu to 3
9. Set sigSofty, sigSoftz to 240 if want tunnelling, 10 if want strong repulsion

Output: Before time propagation begins, the softened Coulomb potential will be plotted as well as a plot of the initial wavefunction. Close these and propagation will begin.
Nothing will be plotted at the end. Run animation2D_sum_sep.py for a 1D plot of each summed over wavefunction.

