#pragma once

// Workaround for CUTLASS synclog __device__ functions being called from __host__ __device__
// This header must be included before any CUTLASS headers

#if defined(__CUDACC__)
  // Tell NVCC to treat these specific errors as warnings
  #pragma nv_diag_suppress 20011
  #pragma nv_diag_suppress 20014
#endif
