#include <cuda_runtime.h>
#include <assert.h>
#include <iostream>
#include <sstream>
#include <fstream>

#define BASE_THREAD_NUM 32
#define BLOCK_DIM 1024
#define MAX_DIMS 10
#define TILE 32
typedef double scalar_t;

#define ADD_FUNC       1
#define MUL_FUNC       2
#define ID_FUNC        3
#define NEG_FUNC       4
#define LT_FUNC        5
#define EQ_FUNC        6
#define SIGMOID_FUNC   7
#define RELU_FUNC      8
#define RELU_BACK_FUNC 9
#define LOG_FUNC       10
#define LOG_BACK_FUNC  11
#define EXP_FUNC       12
#define INV_FUNC       13
#define INV_BACK_FUNC  14
#define IS_CLOSE_FUNC  15
#define MAX_FUNC       16

__device__ scalar_t fn(int fn_id, scalar_t x, scalar_t y=0) {
    switch(fn_id) {
      case ADD_FUNC: {
        return x + y;
      }
      case MUL_FUNC: {
        return x * y;
      }
      case ID_FUNC: {
      	return x;
      }
      case NEG_FUNC: {
        return -x;
      }
      case LT_FUNC: {
        if (x < y) {
          return 1.0;
        }
        else {
          return 0.0;
        }
      }
      case EQ_FUNC: {
        if (x == y) {
          return 1.0;
        }
        else {
          return 0.0;
        }
      }
      case SIGMOID_FUNC: {
        if (x >= 0) {
          return 1.0 / (1.0 + exp(-x));
        }
        else {
          return exp(x) / (1.0 + exp(x));
        }
      }
      case RELU_FUNC: {
        return max(x, 0.0);
      }
      case RELU_BACK_FUNC: {
        if (x > 0) {
          return y;
        }
        else {
          return 0.0;
        }
      }
      case LOG_FUNC: {
        return log(x + 1e-6);
      }
      case LOG_BACK_FUNC: {
        return y / (x + 1e-6);
      }
      case EXP_FUNC: {
        return exp(x);
      }
      case INV_FUNC: {
        return scalar_t(1.0 / x);
      }
      case INV_BACK_FUNC: {
        return -(1.0 / (x * x)) * y;
      }
      case IS_CLOSE_FUNC: {
        return (x - y < 1e-2) && (y - x < 1e-2);
      }
      case MAX_FUNC: {
        if (x > y) {
          return x;
        }
        else {
          return y;
        }
      }
      default: {
        return x + y;
      }
    }
    
}


__device__ int index_to_position(const int* index, const int* strides, int num_dims) {
  /**
   * Converts a multidimensional tensor index into a single-dimensional position in storage
   * based on strides.
   * Args:
   *    index: index tuple of ints
   *    strides: tensor strides
   *    num_dims: number of dimensions in the tensor, e.g. shape/strides of [2, 3, 4] has 3 dimensions
   * 
   * Returns:
   *    int - position in storage
  */
    int position = 0;
    for (int i = 0; i < num_dims; ++i) {
        position += index[i] * strides[i];
    }
    return position;
}

__device__ void to_index(int ordinal, const int* shape, int* out_index, int num_dims) {
  /**
   * Convert an ordinal to an index in the shape. Should ensure that enumerating position 0 ... size of 
   * a tensor produces every index exactly once. It may not be the inverse of index_to_position.
   * Args:
   *    ordinal: ordinal position to convert
   *    shape: tensor shape
   *    out_index: return index corresponding to position
   *    num_dims: number of dimensions in the tensor
   * 
   * Returns:
   *    None (Fills in out_index) 
  */
    int cur_ord = ordinal;
    for (int i = num_dims - 1; i >= 0; --i) {
        int sh = shape[i];
        out_index[i] = cur_ord % sh;
        cur_ord /= sh;
    }
}

__device__ void broadcast_index(const int* big_index, const int* big_shape, int num_dims_big, int* out_index, const int* shape, int num_dims) {
  /**
   * Convert a big_index into big_shape to a smaller out_index into shape following broadcasting rules. 
   * In this case it may be larger or with more dimensions than the shape given. 
   * Additional dimensions may need to be mapped to 0 or removed.
   * 
   * Args:
   *    big_index: multidimensional index of bigger tensor
   *    big_shape: tensor shape of bigger tensor
   *    nums_big_dims: number of dimensions in bigger tensor
   *    out_index: multidimensional index of smaller tensor
   *    shape: tensor shape of smaller tensor  
   *    num_dims: number of dimensions in smaller tensor
   * 
   * Returns:
   *    None (Fills in out_index) 
  */
    for (int i = 0; i < num_dims; ++i) {
        if (shape[i] > 1) {
            out_index[i] = big_index[i + (num_dims_big - num_dims)];
        } else {
            out_index[i] = 0;
        }
    }
}


__global__ void MatrixMultiplyKernel(
    scalar_t* out,
    const int* out_shape,
    const int* out_strides,
    scalar_t* a_storage,
    const int* a_shape,
    const int* a_strides,
    scalar_t* b_storage,
    const int* b_shape,
    const int* b_strides
) {
  /**
   * Multiply two (compact) matrices into an output (also comapct) matrix. Matrix a and b are both in a batch 
   * format, with shape [batch_size, m, n], [batch_size, n, p].
   * Requirements:
   * - All data must be first moved to shared memory.
   * - Only read each cell in a and b once.
   * - Only write to global memory once per kernel.
   * There is guarantee that a_shape[0] == b_shape[0], a_shape[2] == b_shape[1], 
   * and out_shape[0] == a_shape[0], out_shape[1] == b_shape[1]
   * 
   * Args:
   *   out: compact 1D array of size batch_size x m x p to write the output to
   *   out_shape: shape of the output array
   *   out_strides: strides of the output array
   *   a_storage: compact 1D array of size batch_size x m x n
   *   a_shape: shape of the a array
   *   a_strides: strides of the a array
   *   b_storage: comapct 2D array of size batch_size x n x p
   *   b_shape: shape of the b array
   *   b_strides: strides of the b array
   * 
   * Returns:
   *   None (Fills in out array)
   */

    __shared__ scalar_t a_shared[TILE][TILE];
    __shared__ scalar_t b_shared[TILE][TILE];

    // In each block, we will compute a batch of the output matrix
    // All the threads in the block will work together to compute this batch
    int batch = blockIdx.z;
    int a_batch_stride = a_shape[0] > 1 ? a_strides[0] : 0;
    int b_batch_stride = b_shape[0] > 1 ? b_strides[0] : 0;


    /// BEGIN ASSIGN1_2
    /// TODO
    // Hints:
    // 1. Compute the row and column of the output matrix this block will compute
    // 2. Compute the position in the output array that this thread will write to
    // 3. Iterate over tiles of the two input matrices, read the data into shared memory
    // 4. Synchronize to make sure the data is available to all threads
    // 5. Compute the output tile for this thread block
    // 6. Synchronize to make sure all threads are done computing the output tile for (row, col)
    // 7. Write the output to global memory

    assert(false && "Not Implemented");
    /// END ASSIGN1_2
}


__global__ void mapKernel(
    scalar_t* out, 
    int* out_shape, 
    int* out_strides, 
    int out_size, 
    scalar_t* in_storage, 
    int* in_shape, 
    int* in_strides,
    int shape_size,
    int fn_id
) {
  /**
   * Map function. Apply a unary function to each element of the input array and store the result in the output array.
   * Optimization: Parallelize over the elements of the output array.
   * 
   * You may find the following functions useful:
   * - index_to_position: converts an index to a position in a compact array
   * - to_index: converts a position to an index in a multidimensional array
   * - broadcast_index: converts an index in a smaller array to an index in a larger array
   * 
   * Args:
   *  out: compact 1D array of size out_size to write the output to
   *  out_shape: shape of the output array
   *  out_strides: strides of the output array
   *  out_size: size of the output array
   *  in_storage: compact 1D array of size in_size
   *  in_shape: shape of the input array
   *  in_strides: strides of the input array
   *  shape_size: number of dimensions in the input and output arrays, assume dimensions are the same
   *  fn_id: id of the function to apply to each element of the input array
   * 
   * Returns:
   *  None (Fills in out array)
   */

    int out_index[MAX_DIMS];
    int in_index[MAX_DIMS];
    
    /// BEGIN ASSIGN1_2
    /// TODO
    // Hints:
    // 1. Compute the position in the output array that this thread will write to
    // 2. Convert the position to the out_index according to out_shape
    // 3. Broadcast the out_index to the in_index according to in_shape (optional in some cases)
    // 4. Calculate the position of element in in_array according to in_index and in_strides
    // 5. Calculate the position of element in out_array according to out_index and out_strides
    // 6. Apply the unary function to the input element and write the output to the out memory
    
    assert(false && "Not Implemented");
    /// END ASSIGN1_2
}


__global__ void reduceKernel(
    scalar_t* out, 
    int* out_shape, 
    int* out_strides, 
    int out_size, 
    scalar_t* a_storage, 
    int* a_shape, 
    int* a_strides, 
    int reduce_dim,
    double reduce_value,
    int shape_size,
    int fn_id
) {
  /**
   * Reduce function. Apply a reduce function to elements of the input array a and store the result in the output array.
   * Optimization: 
   * Parallelize over the reduction operation. Each kernel performs one reduction.
   * e.g. a = [[1, 2, 3], [4, 5, 6]], kernel0 computes reduce([1, 2, 3]), kernel1 computes reduce([4, 5, 6]).
   * 
   * You may find the following functions useful:
   * - index_to_position: converts an index to a position in a compact array
   * - to_index: converts a position to an index in a multidimensional array
   * 
   * Args:
   *  out: compact 1D array of size out_size to write the output to
   *  out_shape: shape of the output array
   *  out_strides: strides of the output array
   *  out_size: size of the output array
   *  a_storage: compact 1D array of size in_size
   *  a_shape: shape of the input array
   *  a_strides: strides of the input array
   *  reduce_dim: dimension to reduce on
   *  reduce_value: initial value for the reduction
   *  shape_size: number of dimensions in the input & output array, assert dimensions are the same
   *  fn_id: id of the reduce function, currently only support add, multiply, and max
   *  
   * 
   * Returns:
   *  None (Fills in out array)
   */

    // __shared__ double cache[BLOCK_DIM]; // Uncomment this line if you want to use shared memory to store partial results
    int out_index[MAX_DIMS];

    /// BEGIN ASSIGN1_2
    /// TODO
    // 1. Define the position of the output element that this thread or this block will write to
    // 2. Convert the out_pos to the out_index according to out_shape
    // 3. Initialize the reduce_value to the output element
    // 4. Iterate over the reduce_dim dimension of the input array to compute the reduced value
    // 5. Write the reduced value to out memory
    
    assert(false && "Not Implemented");
    /// END ASSIGN1_2
}

__global__ void zipKernel(
    scalar_t* out, 
    int* out_shape, 
    int* out_strides, 
    int out_size,
    int out_shape_size,
    scalar_t* a_storage, 
    int* a_shape, 
    int* a_strides,
    int a_shape_size,
    scalar_t* b_storage, 
    int* b_shape, 
    int* b_strides,
    int b_shape_size,
    int fn_id
) {
  /**
   * Zip function. Apply a binary function to elements of the input array a & b and store the result in the output array.
   * Optimization: Parallelize over the elements of the output array.
   * 
   * You may find the following functions useful:
   * - index_to_position: converts an index to a position in a compact array
   * - to_index: converts a position to an index in a multidimensional array
   * - broadcast_index: converts an index in a smaller array to an index in a larger array
   * 
   * Args:
   *  out: compact 1D array of size out_size to write the output to
   *  out_shape: shape of the output array
   *  out_strides: strides of the output array
   *  out_size: size of the output array
   *  out_shape_size: number of dimensions in the output array
   *  a_storage: compact 1D array of size in_size
   *  a_shape: shape of the input array
   *  a_strides: strides of the input array
   *  a_shape_size: number of dimensions in the input array
   *  b_storage: compact 1D array of size in_size
   *  b_shape: shape of the input array
   *  b_strides: strides of the input array
   *  b_shape_size: number of dimensions in the input array
   *  fn_id: id of the function to apply to each element of the a & b array
   *  
   * 
   * Returns:
   *  None (Fills in out array)
   */

    int out_index[MAX_DIMS];
    int a_index[MAX_DIMS];
    int b_index[MAX_DIMS];

    /// BEGIN ASSIGN1_2
    /// TODO
    // Hints:
    // 1. Compute the position in the output array that this thread will write to
    // 2. Convert the position to the out_index according to out_shape
    // 3. Calculate the position of element in out_array according to out_index and out_strides
    // 4. Broadcast the out_index to the a_index according to a_shape
    // 5. Calculate the position of element in a_array according to a_index and a_strides
    // 6. Broadcast the out_index to the b_index according to b_shape
    // 7.Calculate the position of element in b_array according to b_index and b_strides
    // 8. Apply the binary function to the input elements in a_array & b_array and write the output to the out memory
    
    assert(false && "Not Implemented");
    /// END ASSIGN1_2
}


extern "C" {

void MatrixMultiply(
    scalar_t* out,
    int* out_shape,
    int* out_strides,
    scalar_t* a_storage,
    int* a_shape,
    int* a_strides,
    scalar_t* b_storage,
    int* b_shape,
    int* b_strides,
    int batch, int m, int p
) {
    dim3 blockDims(BASE_THREAD_NUM, BASE_THREAD_NUM, 1); // Adjust these values based on your specific requirements
    dim3 gridDims((m + BASE_THREAD_NUM - 1) / BASE_THREAD_NUM, (p + BASE_THREAD_NUM - 1) / BASE_THREAD_NUM, batch);
    MatrixMultiplyKernel<<<gridDims, blockDims>>>(
        out, out_shape, out_strides, a_storage, a_shape, a_strides, b_storage, b_shape, b_strides
    );
    cudaDeviceSynchronize();
}

void tensorMap(
    scalar_t* out, 
    int* out_shape, 
    int* out_strides, 
    int out_size, 
    scalar_t* in_storage, 
    int* in_shape, 
    int* in_strides,
    int shape_size,
    int fn_id
) {
    int threadsPerBlock = BASE_THREAD_NUM;
    int blocksPerGrid = (out_size + threadsPerBlock - 1) / threadsPerBlock;
    mapKernel<<<blocksPerGrid, threadsPerBlock>>>(out, out_shape, out_strides, out_size, in_storage, in_shape, in_strides, shape_size, fn_id);
    cudaDeviceSynchronize();
}


void tensorZip(
    scalar_t* out, 
    int* out_shape, 
    int* out_strides, 
    int out_size,
    int out_shape_size,
    scalar_t* a_storage, 
    int* a_shape, 
    int* a_strides,
    int a_shape_size,
    scalar_t* b_storage, 
    int* b_shape, 
    int* b_strides,
    int b_shape_size,
    int fn_id
) {
    int threadsPerBlock = BASE_THREAD_NUM;
    int blocksPerGrid = (out_size + threadsPerBlock - 1) / threadsPerBlock;
    zipKernel<<<blocksPerGrid, threadsPerBlock>>>(
      out, out_shape, out_strides, out_size, out_shape_size,
      a_storage, a_shape, a_strides, a_shape_size,
      b_storage, b_shape, b_strides, b_shape_size,
      fn_id);
    cudaDeviceSynchronize();
}



void tensorReduce(
    scalar_t* out, 
    int* out_shape, 
    int* out_strides, 
    int out_size, 
    scalar_t* a_storage, 
    int* a_shape, 
    int* a_strides, 
    int reduce_dim, 
    double reduce_value,
    int shape_size,
    int fn_id
) {
    int threadsPerBlock = BASE_THREAD_NUM;
    int blocksPerGrid = (out_size + threadsPerBlock - 1) / threadsPerBlock;
    reduceKernel<<<blocksPerGrid, threadsPerBlock>>>(
        out, out_shape, out_strides, out_size, 
        a_storage, a_shape, a_strides, 
        reduce_dim, reduce_value, shape_size, fn_id
    );
    cudaDeviceSynchronize();
}

}
