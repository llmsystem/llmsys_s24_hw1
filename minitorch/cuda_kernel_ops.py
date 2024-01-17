from typing import Callable, Optional

from . import operators
from .tensor import Tensor
from .tensor_data import (
    shape_broadcast,
)
from .tensor_ops import MapProto, TensorOps
import os
import ctypes
import numpy as np
import pycuda.gpuarray as gpuarray
import pycuda.driver as drv
from pycuda.compiler import SourceModule
import pycuda.autoinit

# Load the shared library
try:
    lib = ctypes.CDLL("minitorch/cuda_kernels/combine.so")
except:
    print("cuda kernels not implemented: combine.so not found")

# function map
fn_map = {
  operators.add: 1,
  operators.mul: 2,
  operators.id: 3,
  operators.neg: 4,
  operators.lt: 5,
  operators.eq: 6,
  operators.sigmoid: 7,
  operators.relu: 8,
  operators.relu_back: 9,
  operators.log: 10,
  operators.log_back: 11,
  operators.exp: 12,
  operators.inv: 13,
  operators.inv_back: 14,
  operators.is_close: 15,
  operators.max: 16
}

THREADS_PER_BLOCK = 32

class CudaKernelOps(TensorOps):
    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        "See `tensor_ops.py`"
        fn_id = fn_map[fn]

        def ret(a: Tensor, out: Optional[Tensor] = None) -> Tensor:
            if out is None:
                out = a.zeros(a.shape)

            # Define the argument type for the tensorMap function
            lib.tensorMap.argtypes = [
                ctypes.POINTER(ctypes.c_double),  # out
                ctypes.POINTER(ctypes.c_int),    # out_shape
                ctypes.POINTER(ctypes.c_int),    # out_strides
                ctypes.c_int,                    # out_size
                ctypes.POINTER(ctypes.c_double),  # in_storage
                ctypes.POINTER(ctypes.c_int),    # in_shape
                ctypes.POINTER(ctypes.c_int),    # in_strides
                ctypes.c_int,                    # shape_len
                ctypes.c_int,                    # fn_id
            ]

            # Define the return type for the tensorMap function
            lib.tensorMap.restype = None

            # Convert the numpy arrays to gpuarrays that can be loaded to the gpu
            out_array_gpu = gpuarray.to_gpu(out._tensor._storage)
            out_shape_gpu = gpuarray.to_gpu(out._tensor._shape.astype(np.int32))
            out_strides_gpu = gpuarray.to_gpu(out._tensor._strides.astype(np.int32))
            in_array_gpu = gpuarray.to_gpu(a._tensor._storage)
            in_shape_gpu = gpuarray.to_gpu(a._tensor._shape.astype(np.int32))
            in_strides_gpu = gpuarray.to_gpu(a._tensor._strides.astype(np.int32))


            # Call the function
            lib.tensorMap(
                ctypes.cast(out_array_gpu.ptr, ctypes.POINTER(ctypes.c_double)),
                ctypes.cast(out_shape_gpu.ptr, ctypes.POINTER(ctypes.c_int)),
                ctypes.cast(out_strides_gpu.ptr, ctypes.POINTER(ctypes.c_int)),
                ctypes.c_int(out.size),
                ctypes.cast(in_array_gpu.ptr, ctypes.POINTER(ctypes.c_double)),
                ctypes.cast(in_shape_gpu.ptr, ctypes.POINTER(ctypes.c_int)),
                ctypes.cast(in_strides_gpu.ptr, ctypes.POINTER(ctypes.c_int)),
                ctypes.c_int(len(a.shape)),
                ctypes.c_int(fn_id)
            )
            
            # Copy the gpuarray back to the cpu
            out._tensor._storage = out_array_gpu.get()

            # Free the gpuarrays
            out_array_gpu.gpudata.free()
            out_shape_gpu.gpudata.free()
            out_strides_gpu.gpudata.free()
            in_array_gpu.gpudata.free()
            in_shape_gpu.gpudata.free()
            in_strides_gpu.gpudata.free()
            return out

        return ret

    @staticmethod
    def zip(fn: Callable[[float, float], float]) -> Callable[[Tensor, Tensor], Tensor]:
        fn_id = fn_map[fn]

        def ret(a: Tensor, b: Tensor) -> Tensor:
            c_shape = shape_broadcast(a.shape, b.shape)
            out = a.zeros(c_shape)

            # Define the argument type for the tensorZip function
            lib.tensorZip.argtypes = [
                ctypes.POINTER(ctypes.c_double),  # out
                ctypes.POINTER(ctypes.c_int),    # out_shape
                ctypes.POINTER(ctypes.c_int),    # out_strides
                ctypes.c_int,                    # out_size
                ctypes.c_int,                    # out_shape_size
                ctypes.POINTER(ctypes.c_double),  # a_storage
                ctypes.POINTER(ctypes.c_int),    # a_shape
                ctypes.POINTER(ctypes.c_int),    # a_strides
                ctypes.c_int,                    # a_shape_size
                ctypes.POINTER(ctypes.c_double),  # b_storage
                ctypes.POINTER(ctypes.c_int),    # b_shape
                ctypes.POINTER(ctypes.c_int),    # b_strides
                ctypes.c_int,                    # b_shape_size
                ctypes.c_int,                    # fn_id
            ]

            # Define the return type for the tensorZip function
            lib.tensorZip.restype = None

            # BEGIN ASSIGN1_2
            # TODO
            # 1. Convert the numpy arrays to gpuarrays that can be loaded to the gpu
            # 2. Call the tensorZip function implemented in CUDA
            # 3. Copy the gpuarray back to the cpu
            # 4. Free the gpuarrays

            raise NotImplementedError("Zip Function Not Implemented Yet")
            # END ASSIGN1_2
            
            return out

        return ret

    @staticmethod
    def reduce(
        fn: Callable[[float, float], float], reduce_value: float = 0.0
    ) -> Callable[[Tensor, int], Tensor]:
        fn_id = fn_map[fn]

        def ret(a: Tensor, dim: int) -> Tensor:
            out_shape = list(a.shape)
            out_shape[dim] = (a.shape[dim] - 1) // 1024 + 1
            out = a.zeros(tuple(out_shape))

            # Define the return type for the tensorReduce function
            lib.tensorReduce.argtypes = [
                ctypes.POINTER(ctypes.c_double), # out
                ctypes.POINTER(ctypes.c_int),    # out_shape
                ctypes.POINTER(ctypes.c_int),    # out_strides
                ctypes.c_int,                    # out_size
                ctypes.POINTER(ctypes.c_double), # in_storage
                ctypes.POINTER(ctypes.c_int),    # in_shape
                ctypes.POINTER(ctypes.c_int),    # in_strides
                ctypes.c_int,                    # reduce_dim
                ctypes.c_double,                 # reduce_value
                ctypes.c_int,                    # shape_len, assert len(out_shape) == len(in_shape)
                ctypes.c_int,                    # fn_id
            ]

            # Define the return type for the tensorReduce function
            lib.tensorReduce.restype = None

            # BEGIN ASSIGN1_2
            # TODO
            # 1. Convert the numpy arrays to gpuarrays that can be loaded to the gpu
            # 2. Call the tensorReduce function implemented in CUDA
            # 3. Copy the gpuarray back to the cpu
            # 4. Free the gpuarrays
            
            raise NotImplementedError("Reduce Function Not Implemented Yet")
            # END ASSIGN1_2
            
            return out

        return ret

    @staticmethod
    def matrix_multiply(a: Tensor, b: Tensor) -> Tensor:
        both_2d = 0
        if len(a.shape) == 2:
            a = a.contiguous().view(1, a.shape[0], a.shape[1])
            both_2d += 1
        if len(b.shape) == 2:
            b = b.contiguous().view(1, b.shape[0], b.shape[1])
            both_2d += 1
        both_2d = both_2d == 2

        ls = list(shape_broadcast(a.shape[:-2], b.shape[:-2]))
        ls.append(a.shape[-2])
        ls.append(b.shape[-1])
        assert a.shape[-1] == b.shape[-2]
        out = a.zeros(tuple(ls))

        # Define the argument type for the tensorZip function
        lib.MatrixMultiply.argtypes = [
            ctypes.POINTER(ctypes.c_double),  # out
            ctypes.POINTER(ctypes.c_int),    # out_shape
            ctypes.POINTER(ctypes.c_int),    # out_strides
            ctypes.POINTER(ctypes.c_double),  # a_storage
            ctypes.POINTER(ctypes.c_int),    # a_shape
            ctypes.POINTER(ctypes.c_int),    # a_strides
            ctypes.POINTER(ctypes.c_double),  # b_storage
            ctypes.POINTER(ctypes.c_int),    # b_shape
            ctypes.POINTER(ctypes.c_int),    # b_strides
            ctypes.c_int,                    # batch_size
            ctypes.c_int,                    # out_shape[1], m
            ctypes.c_int,                    # out_shape[2], p
        ]

        # Define the return type for the tensorZip function
        lib.MatrixMultiply.restype = None

        # BEGIN ASSIGN1_2
        # TODO
        # 1. Convert the numpy arrays to gpuarrays that can be loaded to the gpu
        # 2. Call the Matmul function implemented in CUDA
        # 3. Copy the gpuarray back to the cpu
        # 4. Free the gpuarrays

        raise NotImplementedError("Matrix Multiply Function Not Implemented Yet")
        # END ASSIGN1_2
        
        # Undo 3d if we added it.
        if both_2d:
            out = out.view(out.shape[1], out.shape[2])
        return out
