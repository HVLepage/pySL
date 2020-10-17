__global__ void gqsl_re(float *re, float *im, float *V, float bb, float aa, int Nx)
{
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx > 0 && idx < Nx-1){
    //re[idx] = re[idx];
    re[idx] = re[idx] - (V[idx]*bb*im[idx] + aa*(2*im[idx] - (im[idx-1] + im[idx+1])));
  }
}

__global__ void gqsl_im(float *re, float *im, float *V, float bb, float aa, int Nx)
{
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx > 0 && idx < Nx-1){
    //im[idx] = im[idx];
    im[idx] = im[idx] +  V[idx]*bb*re[idx] + aa*(2*re[idx] - (re[idx-1] + re[idx+1]));
  }
}
