#include <pycuda-complex.hpp>
//#include <math.h>
//#include <complex>
//#include <cmath>

/*******************************************************************************************************************************
Calculate Coulomb potential
*******************************************************************************************************************************/

#define Npdim 3
#define dim2p 1 //coulomb pairs (is same as dim)


__global__ void CoulombPotential(int N, float* V2P, int Nx, int Ny, int Nz, float Xa, float dx, float Ya, float dy, float Za, float dz, float RelPerm,
                                 float sigmaFit0, float sigmaFit1, float sigmaFit2, float sigmaFit3, float sigmaFit4, float sigmaFit5, float sigmaFit6, 
                                 float sigmaFit7, float sigmaFit8, int* Sizes, int dim, float* Xas,float* dXs, float softeningParam){       
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
Multiply by potential, calculate time dependent potential part
*******************************************************************************************************************************/
__global__ void Axpy(pycuda::complex<float>* Psi, float* V, 
					int N, float t, 
					int* kernelInds, float Nm1, int Np, int N1, int dim, 
					float ASAW, float kSAW, float SAWoffset, float wSAW, float cSAW, 
					float RS, float DRt0, float URt0, float ATB2, float s2TB2, 
					float* Mass, float* dXs, float Xa, float Ya) {

	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < N) {

		float Vt = 0.0; //some potentially time depentent part of V

		// if (ASAW > 0.0) { // add SAW potential if necessary
			for (int p = 0; p < Np; p++) {
				// Set current coordinates
			
				int ind = ((index / (int)powf(N1, Np - p - 1)) % (N1)) * dim; //raising float to power
				float x = Xa + kernelInds[ind] * dXs[0];
				float y = dim > 1 ? Ya + kernelInds[ind + 1] * dXs[1] : 0.0;
				// Add SAW potential
				Vt += ASAW * (1.0 - cosf(kSAW * (x - SAWoffset) - wSAW * t));
				float TBRamp = 0.5 * (tanhf(RS * (y + (t - DRt0) * cSAW)) + tanhf(-RS * (y + (t - URt0) * cSAW)));
				Vt -= ATB2 * Mass[p] * expf(-s2TB2 * powf(x, 2.0) / 2.0) * TBRamp;
			}
		// }

		Psi[index] *= (V[index] + Vt) * Nm1;
		//Psi[index].y *= (V[index] + Vt) * Nm1;
	}
}

/*******************************************************************************************************************************
Trace out dimensions for export
*******************************************************************************************************************************/

__global__ void AddTrace(int n, pycuda::complex<float>* trace, pycuda::complex<float>* Psi, int p1, int d1, int p2, int d2, int N1, int Np, int dim, float diff, int* Sizes, int* kernelInds) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < n) {
		//int offset = i * params->dim * params->ParticlesNr; // this is an offset to know where to look in kernelInds for given i
		//Index1DtoNDkernel(i, kernelInds, Sizes, params, offset);
		int indp1 = ((i / (int)powf(N1, Np - p1 -2)) % (N1)) * dim;
		int indp2 = ((i / (int)powf(N1, Np - p2 -2)) % (N1)) * dim;
		int indTrace = kernelInds[indp2 + (d2 - 1) ] + Sizes[d2 - 1] * kernelInds[indp1 + (d1 - 1)];
		trace[indTrace] += (powf(Psi[i].real(), 2.0) + powf(Psi[i].imag(), 2.0)) * diff;
	}
}



/*******************************************************************************************************************************
Momentum-space Staggered Leapfrog solver
*******************************************************************************************************************************/
__global__ void kSpaceSL(int N, pycuda::complex<float>* Psi, pycuda::complex<float>* PsiConv, 
						float t, float dt, int Np, int N1K, int dim, float value, int* Sizes, 
						float* Mass, float* dKs, int iterateRe, int* kernelIndsK) {

	int index = blockIdx.x * blockDim.x + threadIdx.x;
	//int stride = blockDim.x * gridDim.x;
	//float reC, imC;


	if (index < N) {
		
		const   pycuda::complex<float> j(0.0,1.0);    
		int indN = 0;

		int l = (int)pow((float)dim, (float)Np);
		for (int i = 0; i < l; i++) {
			int stride = 1;
			for (int j = i + 1; j < l; j++) {
				stride *= Sizes[j % dim];
			}
			int p = (i ) / dim;
			int indp = ((index / (int)powf(N1K, Np - p - 1)) % (N1K)) * dim;
			int ind = kernelIndsK[indp + (i % dim)] < 0 ? kernelIndsK[indp + (i % dim)] + Sizes[i % dim] : kernelIndsK[indp + (i % dim)];
			indN += ind * stride;
		}


		//Staggered leapfrog in K space
		if (iterateRe == 0) {
			Psi[indN] += 2.0 * dt * (PsiConv[indN].imag());
			for (int d = 0; d < dim; d++) {
				for (int p = 0; p < Np; p++) {
					int indK = ((index / (int)powf(N1K, Np - p - 1)) % (N1K)) * dim;
					Psi[indN] += (value / Mass[p]) * powf(kernelIndsK[indK + d] * dKs[d], 2) * Psi[indN].imag();
				}
			}
		}
		else {
			Psi[indN] += pycuda::complex<float>(0.0, -2.0 * dt * (PsiConv[indN].real()));
			for (int d = 0; d < dim; d++) {
				for (int p = 0; p < Np; p++) {
					int indK = ((index / (int)powf(N1K, Np - p - 1)) % (N1K)) * dim;
					Psi[indN] -=  pycuda::complex<float>(0.0, (value / Mass[p]) * powf(kernelIndsK[indK + d] * dKs[d], 2) * Psi[indN].real());
				}
			}
		}
	}
}



/*******************************************************************************************************************************
Real-space Staggered Leapfrog solver
*******************************************************************************************************************************/
__global__ void rSpaceSL(int N, float* PsiRe, float* PsiIm, float* V,
                        float t, int iterateRe, int* kernelInds, int Np, int N1, int dim, 
                        float ASAW, float kSAW, float SAWoffset, float wSAW, float cSAW, 
                        float RS, float DRt0, float URt0, float ATB2, float s2TB2, 
                        float fbb, float* faa, int* Sizes, float* Mass, float* dXs,
						float Xa, float Ya, float wX, float* xEqms, float Acos, float kXcos, float tau, int oneptdepPotential, float* VCoulomb, float tauCoulomb) {

	int index = blockIdx.x * blockDim.x + threadIdx.x; // CUDA index
	
	if (index < N) { //check if index is within the wavefunction and NOT on a boundary
		
		float Vt = 0.0; //some potentially time dependent part of V

		for (int p = 0; p < Np; p++) {
			// Set current coordinates
			int ind = ((index / (int)powf(N1, Np - p - 1)) % (N1)) * dim; //raising float to power
			float x = Xa + kernelInds[ind] * dXs[0];
			float y = dim > 1 ? Ya + kernelInds[ind + 1] * dXs[1] : 0.0;
			// Add time-dependent SAW potential
			Vt += ASAW * (1.0 - cosf(kSAW * (x - SAWoffset) - wSAW * t));
      // time-dependent ramp potential
			float TBRamp = 0.5 * (tanhf(RS * (y + (t - DRt0) * cSAW)) + tanhf(-RS * (y + (t - URt0) * cSAW)));
			Vt -= ATB2 * Mass[p] * expf(-s2TB2 * powf(x, 2.0) / 2.0) * TBRamp;
		}
		
		
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
				//Vt+=Acos*(1+cosf(kXcos*x))*expf(-t/tau);
				Vt+=expf(-powf(x+45,2)/(2*powf(2,2)))*0.24*expf(-t/100);
				
			}

			//Vt+=VCoulomb[index]*(1-expf(-t/tauCoulomb));
			Vt+=VCoulomb[index];
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
					
					// if(fabsf(PsiRe[index])<1e-6){
					// 	PsiRe[index]=0;
					// 	PsiIm[index]=0;
					// }
					
					

					// PsiRe[index] -= (faa[d] / Mass[p]) * PsiIm[index + idxNearestNeighbor];
					// PsiRe[index] -= (faa[d] / Mass[p]) * PsiIm[index - idxNearestNeighbor]; 

					//PsiRe[0]=0;
					//PsiRe[N-1]=0;
					//delete [] inds;


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
					
	
					// if(fabsf(PsiIm[index])<1e-6){
					// 	PsiIm[index]=0;
					// 	PsiRe[index]=0;
					// }
					
					// PsiIm[index] += (faa[d] / Mass[p]) * PsiRe[index + idxNearestNeighbor];
					// PsiIm[index] += (faa[d] / Mass[p]) * PsiRe[index - idxNearestNeighbor]; 

					//PsiIm[0]=0;
					//PsiIm[N-1]=0;

					//delete [] inds;


					// }
				}
			}
		}
	}

	//delete [] inds;
}
