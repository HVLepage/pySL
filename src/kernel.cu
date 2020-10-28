// #include <pycuda-complex.hpp>
//#include <math.h>



__device__ int IndexNDto1Dkernel(int* inds, int* sizes, int Np, int dim, int offset) {
	int indOut = 0;
	int l = Np * dim;
	for (int i = 0; i < l; i++) {
		int stride = 1;
		for (int j = i + 1; j < l; j++) {
			stride *= sizes[j % dim];
		}
		indOut += inds[i + offset] * stride;
	}
	return indOut;
}

//// Return a 1D index from an ND row major array. Offset tells where to look in a big index memory block
__device__ void Index1DtoNDkernel(int ind, int* inds, int* sizes, int Np, int dim, int offset) {
	int l = Np * dim;
	int N = 1;
	for (int i = 0; i < dim; i++)
		N *= sizes[i];
	N = (int)pow((float)N, (float)Np);
	long stride = N;
	for (int i = 0; i < l; i++) {
		long temp = stride;
		stride = (long)((double)temp / (double)sizes[i % dim]);
		inds[i + offset] = ind / stride;
		ind = ind - inds[i + offset] * stride;
	}
}

__global__ void iterateGPUR(int N, float* PsiRe, float* PsiIm, float* V, bool* domain, float t, int iterateRe, int* kernelInds, int Np, int N1, int dim, float ASAW, float kSAW, float SAWoffset, float wSAW, float cSAW, float RS, float DRt0, float URt0, float ATB2, float s2TB2, float fbb, float* faa, int* Sizes, float* Mass, float* dXs, float Xa, float Ya) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	//int stride = blockDim.x * gridDim.x;
	//float TBRamp1, TBRamp2;

	if (index < N && domain[index]==1) {


		int indPMone;

		float Vt = 0.0; //some potentially time depentent part of V
		//int i = (index / (Nz * Ny)) % Nx;

		for (int p = 0; p < Np; p++) {
			// Set current coordinates
			//float x = Xa + i * dx;
			int ind = ((index / (int)powf(N1, Np - p - 1)) % (N1)) * dim;
			float x = Xa + kernelInds[ind] * dXs[0];
			float y = dim > 1 ? Ya + kernelInds[ind + 1] * dXs[1] : 0.0;
			// Add SAW potential
			Vt += ASAW * (1.0 - cosf(kSAW * (x - SAWoffset) - wSAW * t));

			float TBRamp = 0.5 * (tanhf(RS * (y + (t - DRt0) * cSAW)) + tanhf(-RS * (y + (t - URt0) * cSAW)));
			Vt -= ATB2 * Mass[p] * expf(-s2TB2 * powf(x, 2.0) / 2.0) * TBRamp;
		}
		
		
		int indPMoneOffset=1;
		if (iterateRe == 0) {
			PsiRe[index] += (V[index] + Vt) * fbb * PsiIm[index]; // The potential part
			for (int d = 0; d < dim; d++) {
				for (int p = 0; p < Np; p++) {
					PsiRe[index] += (faa[d] / Mass[p]) * (2.0 * PsiIm[index]); // On-site bit

					if (d == (dim - 1))
						indPMoneOffset = 1;
					else if (d == (dim - 2))
						indPMoneOffset = Sizes[dim - 1];
					else if (d == (dim - 3))
						indPMoneOffset = Sizes[dim - 1]* Sizes[dim - 2];

					
					int pp = Np - p -1;
					while (pp > 0) {
						indPMoneOffset *= N1;
						pp--;
					}

					indPMone = index - indPMoneOffset;

					PsiRe[index] -= (faa[d] / Mass[p]) * PsiIm[indPMone]; //off by one bit

					indPMone = index + indPMoneOffset;
					
					PsiRe[index] -= (faa[d] / Mass[p]) * PsiIm[indPMone];//off by one bit
					
				}
			}
		}
		else {// same as above but imaginary part
			PsiIm[index] -= (V[index] + Vt) * fbb * PsiRe[index];
			for (int d = 0; d < dim; d++) {
				for (int p = 0; p < Np; p++) {
					PsiIm[index] -= (faa[d] / Mass[p]) * (2.0 * PsiRe[index]);

					if (d == (dim - 1))
						indPMoneOffset = 1;
					else if (d == (dim - 2))
						indPMoneOffset = Sizes[dim - 1];
					else if (d == (dim - 3))
						indPMoneOffset = Sizes[dim - 1] * Sizes[dim - 2];

					int pp = Np - p -1;
					while (pp > 0) {
						indPMoneOffset *= N1;
						pp--;
					}
					indPMone = index - indPMoneOffset;

    				PsiIm[index] += (faa[d] / Mass[p]) * PsiRe[indPMone];
					
					indPMone = index + indPMoneOffset;
					
					PsiIm[index] += (faa[d] / Mass[p]) * PsiRe[indPMone];
					
				}
			}
		}
	}
}




__global__ void gqsl_x_re(float *re, float *im, float *V, float bb, float aa, int N)
{
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx > 0 && idx < N-1){
    //re[idx] = re[idx];
    re[idx] = re[idx] - (V[idx]*bb*im[idx] + aa*(2*im[idx] - (im[idx-1] + im[idx+1])));
  }
}

__global__ void gqsl_x_im(float *re, float *im, float *V, float bb, float aa, int N)
{
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx > 0 && idx < N-1){
    //im[idx] = im[idx];
    im[idx] = im[idx] +  V[idx]*bb*re[idx] + aa*(2*re[idx] - (re[idx-1] + re[idx+1]));
  }
}

















// __global__ void gqsl_re(pycuda::complex<float> *psi, float *V, float bb, float aa, int Nx)
// {
//   int idx = threadIdx.x + blockDim.x * blockIdx.x;
//   if (idx > 0 && idx < Nx-1){
//     //re[idx] = re[idx];
//     psi[idx].real() = psi[idx].real() - (V[idx]*bb*psi[idx].imag() + aa*(2*psi[idx].imag() - (psi[idx-1].imag() + psi[idx+1].imag())));
//   }
// }

// __global__ void gqsl_im(pycuda::complex<float> *psi, float *V, float bb, float aa, int Nx)
// {
//   int idx = threadIdx.x + blockDim.x * blockIdx.x;
//   if (idx > 0 && idx < Nx-1){
//     //im[idx] = im[idx];
//     psi[idx].imag() = psi[idx].imag() +  V[idx]*bb*psi[idx].real() + aa*(2*psi[idx].real() - (psi[idx-1].real() + psi[idx+1].real()));
//   }
// }