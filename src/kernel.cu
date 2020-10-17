// #include <pycuda-complex.hpp>


__global__ void gqsl_re(float *re, float *im, float *V, float bb, float aa, int N)
{
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx > 0 && idx < N-1){
    //re[idx] = re[idx];
    re[idx] = re[idx] - (V[idx]*bb*im[idx] + aa*(2*im[idx] - (im[idx-1] + im[idx+1])));
  }
}

__global__ void gqsl_im(float *re, float *im, float *V, float bb, float aa, int N)
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