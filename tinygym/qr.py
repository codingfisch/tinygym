import os, ctypes

from tinygrad import Tensor
CODE = '''
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

void qr(float* A, float* Q, float* R, int m, int n) {
    for (int j = 0; j < n; j++) {
        float norm = 0.0;
        for (int i = 0; i < m; i++)
            norm += A[i * n + j] * A[i * n + j];
        R[j * n + j] = sqrt(norm);
        if (R[j * n + j] < 1e-8) {
            R[j * n + j] = 1e-8;
        }
        for (int i = 0; i < m; i++)
            Q[i * n + j] = A[i * n + j] / R[j * n + j];
        for (int k = j + 1; k < n; k++) {
            R[j * n + k] = 0.0;
            for (int i = 0; i < m; i++)
                R[j * n + k] += Q[i * n + j] * A[i * n + k];
            for (int i = 0; i < m; i++)
                A[i * n + k] -= Q[i * n + j] * R[j * n + k];
        }
    }
}
'''


class QR(ctypes.CDLL):
    def __init__(self):
        dll_path = __file__.replace('.py', '_dll.so')
        if not os.path.isfile(dll_path):
            os.system(f'echo "{CODE}" | gcc -shared -o {dll_path} -fPIC -x c -')
        super().__init__(dll_path)
        self.qr.argtypes = 3 * [ctypes.POINTER(ctypes.c_float)] + 2 * [ctypes.c_int]

    def __call__(self, x):
        q = (ctypes.c_float * x.numel())(0)
        r = (ctypes.c_float * x.shape[1]**2)(0)
        a = (ctypes.c_float * x.numel())(*x.contiguous().view(-1).tolist())
        self.qr(a, q, r, x.shape[0], x.shape[1])
        return Tensor(list(q)).view(*x.shape), Tensor(list(r)).view(x.shape[1], x.shape[1])
