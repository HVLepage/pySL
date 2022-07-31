
//#include <math.h>
//#include <complex>
//#include <cmath>

/*******************************************************************************************************************************
Calculate Coulomb potential
*******************************************************************************************************************************/

#define Npdim 1
#define dim2p 1 //coulomb pairs (is same as dim)


__global__ void CoulombPotential(int N, float* V2P, int Nx, int Ny, int Nz, float Xa, float dx, float Ya, float dy, float Za, float dz, float RelPerm,
                                  int* Sizes, int dim, float* Xas,float* dXs, float softeningParam){       
    int ind = blockIdx.x * blockDim.x + threadIdx.x;
	
   
	if (ind<N){


		//convert 1D index into a 2*number of dimensions index
		int stride=N; //THIS IS N FOR 2 PARTICLES, SEE HOW PASSED TO FUNCTION
		
		int inds[2*dim2p];
		int temp_index =ind;
		
		for (int i=0;i<2*dim2p;i++){ //convert 1d index into a 2 particles *number of dimensions index
			stride=stride/Sizes[i%dim2p];
			inds[i]=temp_index/stride; //will floor as type int
			temp_index=temp_index -inds[i]*stride;
		}
		

		float Xs[dim2p*2]; //will hold the positions of each particle [x1,y1,z1,x2,y2,z2]
		for (int i=0; i<2*dim2p;i++){
			Xs[i]=Xas[i%dim2p]+inds[i]*dXs[i%dim2p]; //Xas holds the starting values for each dimension i.e. Xa, Ya etc.
		}
		
		
		float d=0; //dynamic distance
		for (int i=0; i<dim2p;i++){
			d+=pow((Xs[i+dim2p]-Xs[i]),2);
			
		}
		d=sqrt(d);
		

		
		float dsoft = sqrt(pow(d,2)+pow(softeningParam,2)); //won't work in 3D

		double VVal = (18.8950673 / RelPerm) * powf(dsoft, -1.0); //use Coulomb softening with cap at 100, 18.89... is conversion factor to natural units and nm
		VVal = (18.8950673 / RelPerm) * powf(dsoft, -1.0); //use Coulomb softening with cap at 100, 18.89... is conversion factor to natural units and nm
		if (VVal>100.0){
			VVal=100.0;
		}
		V2P[ind]=VVal;
		//V2P[ind]=0; //if want 0 interactions can just set here.
		
   }

}


__global__ void AllocateCoulombPotential(int N, float* VCoulomb, int p1, int p2, int dim, int N2P,int* Sizes, float* Charge, float* V2P, int Np){
	int ind = blockIdx.x * blockDim.x + threadIdx.x;
	
   
	if (ind<N){

		int stride=N;
				
		int inds[Npdim]; //dim =1 only here
		int temp_index =ind;
		
		for (int i=0;i<Np*dim;i++){ //may possibly slow down a lot for Np*dim>1
			stride=stride/Sizes[i%dim];
			inds[i]=temp_index/stride; //will floor as type int
			temp_index=temp_index -inds[i]*stride;
		}
        
        int inds2P[dim2p*2];

		for(int i=0;i<dim;i++){
			inds2P[i] = inds[p1 * dim + i] ;
            inds2P[i + dim] = inds[p2 * dim + i];
		}
		stride=N2P;
		int ind2P=0;
		for(int i=0;i <dim*2;i++){
			stride=stride/Sizes[i%dim];
			ind2P+=inds2P[i]*stride;

		}
		
    	VCoulomb[ind]=Charge[p1] * Charge[p2] * V2P[ind2P];
		
   }

}


__global__ void npfrom1p_gpu(int N, float* PsiReal, float* PsiIm, float* waveReal, float*waveIm, int Np, int p,int* Sizes,int dim){
	int ind = blockIdx.x * blockDim.x + threadIdx.x;
	
   
	if (ind<N){

		int stride=N;
				
		int inds[Npdim]; //dim =1 only here
		int temp_index =ind;

		for (int i=0;i<Np;i++){ //may possibly slow down a lot for Np*dim>1
			stride=stride/Sizes[i%dim];
			inds[i]=temp_index/stride; //will floor as type int
			temp_index=temp_index -inds[i]*stride;
		}
        
		if (p==0){
		
			PsiReal[ind]=waveReal[inds[p]];
			PsiIm[ind]=waveIm[inds[p]];
		}
		else{
			PsiReal[ind]=PsiReal[ind]*waveReal[inds[p]]-PsiIm[ind]*waveIm[inds[p]]; //complex number multiplication,  might not work
			PsiIm[ind]=PsiReal[ind]*waveIm[inds[p]]+PsiIm[ind]*waveReal[inds[p]];
		}

		
		
   }

}





/*******************************************************************************************************************************
Real-space Staggered Leapfrog solver
*******************************************************************************************************************************/
__global__ void rSpaceSL(int N, float* PsiRe, float* PsiIm, float* V,
                        float t, int iterateRe, int Np, int N1, int dim, 
                        float fbb, float* faa, int* Sizes, float* Mass, float* dXs,
						float Xa, float Ya, float wX, float* xEqms, float Acos, float kXcos, float tau, int oneptdepPotential, float* VCoulomb, float tauCoulomb,
						int potentialBarrier, float barrierPos, float barrierWidth, float barrierHeight, float tauBarrier,int adiabaticSine, int adiabaticCoulomb, int nonadiabaticBarrier,
						int* same_spin_pairs, int num_pairs) {

	int index = blockIdx.x * blockDim.x + threadIdx.x; // CUDA index
	
	if (index < N) { //check if index is within the wavefunction and NOT on a boundary
		
		float Vt = 0.0; //some potentially time dependent part of V

		
		
		
		if (oneptdepPotential==1){
			
			int stride=N;
				
			int inds[Npdim]; //dim =1 only here
			int temp_index =index;
			
			for (int i=0;i<Np*dim;i++){ //may possibly slow down a lot for Np*dim>1
				
				stride=stride/Sizes[i%dim];
				inds[i]=temp_index/stride; //will floor as type int
				temp_index=temp_index -inds[i]*stride;
			}
			
			for(int p=0; p<Np;p++){
				float x= Xa + inds[p] *dXs[0];
				Vt+=0.5*Mass[p]*(powf(wX,2) * powf((x-xEqms[p]),2) );
				if(adiabaticSine==1){
					Vt+=(Acos*(1+cosf(kXcos*x)))*expf(-t/tau);
				}
				if (potentialBarrier==1){
					if(nonadiabaticBarrier==1){
						Vt+=expf(-powf(x-barrierPos,2)/(2*powf(barrierWidth,2)))*barrierHeight*expf(-t/tauBarrier);
					}
					else{
						Vt+=expf(-powf(x-barrierPos,2)/(2*powf(barrierWidth,2)))*barrierHeight;

					}
				}
				
			}

			if(adiabaticCoulomb==1){
				Vt+=VCoulomb[index]*(1-expf(-t/tauCoulomb));
			}
			else{
				Vt+=VCoulomb[index];
			}
		}
		//printf("%f",Vt);
	
		int idxNearestNeighbor=1;
		if (iterateRe) { // Iterate real part of the wave function, for odd index in pySL.py, iterateRe=1
			PsiRe[index] += (V[index]+ Vt) * fbb * PsiIm[index]; // The potential part
			//PsiRe[index] += (V[index] *expf(-t/100) + Vt) * fbb * PsiIm[index]; //exponential decay of potential
			for (int d = 0; d < dim; d++) { //for each index need to do this
				for (int p = 0; p < Np; p++) {
					PsiRe[index] += (faa[d] / Mass[p]) * (2.0 * PsiIm[index]); // On-site kinetic term

					// Calculate the index offset of the nearest neighbor lattice site
					if (d == (dim - 1))
						idxNearestNeighbor = 1;
					else if (d == (dim - 2))
						idxNearestNeighbor = Sizes[dim - 1];
					else if (d == (dim - 3))
						idxNearestNeighbor = Sizes[dim - 1]* Sizes[dim - 2];

					int pp = Np - p -1; //goes from Np-1 to 0
					while (pp > 0) {
						idxNearestNeighbor *= N1;
						pp--;
					}
					//negative indices?? probably 0 but not always
					//PsiIm[-1]=4239874239847; //certainly messes everything up whereas changing PsiIm[-2] doesn't
					//this needs to be fixed for multiple dimensions as assumes that e.g. y=0 z=zb and y=dy z=0 are next to each other
					//adopt the method used in rSpaceSolverGPU.py or otherwise

					//effectively rewriting index1DtoND function
					int stride=N;
					
					int inds[Npdim];
					int temp_index =index;
					
					for (int i=0;i<Np*dim;i++){ //may possibly slow down a lot for Np*dim>1
						
						stride=stride/Sizes[i%dim];
						inds[i]=temp_index/stride; //will floor as type int
						temp_index=temp_index -inds[i]*stride;
					}
					// if(index==0){
					// 	printf("%i",inds[0]);
					// 	printf("\n");
					// }
				
					if(inds[d+p*dim]<Sizes[d]-1){
						PsiRe[index] -= (faa[d] / Mass[p]) * PsiIm[index + idxNearestNeighbor];
					}
					
					
					if(inds[d+p*dim]>0){
						PsiRe[index] -= (faa[d] / Mass[p]) * PsiIm[index - idxNearestNeighbor]; 
					}
					
			
					//comment the below code out if don't want spin
					for(int i=0;i<num_pairs;i++){
						if (inds[same_spin_pairs[2*i]]==inds[same_spin_pairs[2*i+1]]){
							PsiRe[index]=0;
						}

					}
	

				}
			}
		}

		
		else { // Iterate imaginary part of the wave function
		
		
			PsiIm[index] -= (V[index] + Vt) * fbb * PsiRe[index];
			//PsiIm[index] -= (V[index] *expf(-t/100)+ Vt) * fbb * PsiRe[index];
			for (int d = 0; d < dim; d++) {
				for (int p = 0; p < Np; p++) {
					PsiIm[index] -= (faa[d] / Mass[p]) * (2.0 * PsiRe[index]); // On-site kinetic term
          
					// Calculate the index offset of the nearest neighbor lattice site
					if (d == (dim - 1))
						idxNearestNeighbor = 1;
					else if (d == (dim - 2))
						idxNearestNeighbor = Sizes[dim - 1];
					else if (d == (dim - 3))
						idxNearestNeighbor = Sizes[dim - 1] * Sizes[dim - 2];

					int pp = Np - p -1;
					while (pp > 0) {
						idxNearestNeighbor *= N1;
						pp--;
					}
				
					int stride=N;
					int inds[Npdim];
					//int *inds = new int [Np*dim];
					int temp_index =index;

					for (int i=0;i<Np*dim;i++){
						stride=stride/Sizes[i%dim];
						inds[i]=temp_index/stride; //will floor as type int
						
						temp_index=temp_index -inds[i]*stride;
					}

					if(inds[d+p*dim]<Sizes[d]-1){
						PsiIm[index] += (faa[d] / Mass[p]) * PsiRe[index + idxNearestNeighbor];
					}
					
					if(inds[d+p*dim]>0){
						PsiIm[index] += (faa[d] / Mass[p]) * PsiRe[index - idxNearestNeighbor]; 
					}
					
					//comment the below code out if don't want spin
					for(int i=0;i<num_pairs;i++){
						if (inds[same_spin_pairs[2*i]]==inds[same_spin_pairs[2*i+1]]){
							PsiIm[index]=0;
						}

					}
	
				
				}
			}
		}
	}


}
