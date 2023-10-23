import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np

mod = SourceModule("""
__global__ void minkowski_distance(float *a, float *b, float *result, int n, float p)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    float sum = 0;
    if (idx < n) {
        float diff = fabs(a[idx] - b[idx]);
        sum += powf(diff, p);
    }
    atomicAdd(result, sum);
}
""")

# Initialize input vectors
a = np.random.rand(10000).astype(np.float32)
b = np.random.rand(10000).astype(np.float32)
result = np.array([0], dtype=np.float32)

# Transfer data to GPU
a_gpu = cuda.mem_alloc(a.nbytes)
b_gpu = cuda.mem_alloc(b.nbytes)
result_gpu = cuda.mem_alloc(result.nbytes)

cuda.memcpy_htod(a_gpu, a)
cuda.memcpy_htod(b_gpu, b)
cuda.memcpy_htod(result_gpu, result)

# Set up and launch CUDA kernel
func = mod.get_function("minkowski_distance")
block_size = (256, 1, 1)
grid_size = ((10000 + 256 - 1) // 256, 1, 1)
p = np.float32(3)  # For example, p=3 for cubic distance
func(a_gpu, b_gpu, result_gpu, np.int32(10000), p, block=block_size, grid=grid_size)

# Copy result back to CPU
cuda.memcpy_dtoh(result, result_gpu)
result = np.power(result[0], 1/p)

print(f'Minkowski Distance: {result}')