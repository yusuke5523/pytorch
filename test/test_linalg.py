import torch
import unittest
import itertools
import warnings
import math
from math import inf, nan, isnan
from itertools import product
from functools import reduce
import random
from random import randrange

from torch.testing._internal.common_utils import \
    (TestCase, run_tests, IS_MACOS, IS_WINDOWS, TEST_WITH_ASAN, make_tensor,
     slowTest, TEST_WITH_ROCM, TEST_SCIPY, iter_indices)
from torch.testing._internal.common_device_type import \
    (instantiate_device_type_tests, dtypes, dtypesIfCUDA,
     onlyCUDA, skipCUDAIfNoMagma, skipCPUIfNoLapack, precisionOverride,
     skipCUDAIfNoMagmaAndNoCusolver, onlyOnCPUAndCUDA, dtypesIfCPU, onlyCPU,
     skipCUDAIf)
from torch.testing._internal.jit_metaprogramming_utils import gen_script_fn_and_args
from torch.testing._internal.common_cuda import tf32_on_and_off, with_tf32_off, tf32_is_not_fp32
from torch.autograd import gradcheck

import numpy as np

if TEST_SCIPY:
    import scipy

AMPERE_OR_ROCM = TEST_WITH_ROCM or tf32_is_not_fp32()

class TestLinalg(TestCase):
    exact_dtype = True

    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    def test_pca_lowrank(self, device):
        from torch.testing._internal.common_utils import random_lowrank_matrix, random_sparse_matrix

        dtype = torch.double

        def run_subtest(guess_rank, actual_rank, matrix_size, batches, device, pca, **options):
            density = options.pop('density', 1)
            if isinstance(matrix_size, int):
                rows = columns = matrix_size
            else:
                rows, columns = matrix_size
            if density == 1:
                a_input = random_lowrank_matrix(actual_rank, rows, columns, *batches, device=device, dtype=dtype)
                a = a_input
            else:
                a_input = random_sparse_matrix(rows, columns, density, device=device, dtype=dtype)
                a = a_input.to_dense()

            u, s, v = pca(a_input, q=guess_rank, **options)

            self.assertEqual(s.shape[-1], guess_rank)
            self.assertEqual(u.shape[-2], rows)
            self.assertEqual(u.shape[-1], guess_rank)
            self.assertEqual(v.shape[-1], guess_rank)
            self.assertEqual(v.shape[-2], columns)

            A1 = u.matmul(s.diag_embed()).matmul(v.transpose(-2, -1))
            ones_m1 = torch.ones(batches + (rows, 1), dtype=a.dtype, device=device)
            c = a.sum(axis=-2) / rows
            c = c.reshape(batches + (1, columns))
            A2 = a - ones_m1.matmul(c)
            self.assertEqual(A1, A2)

            if density == 1:
                # actual rank is known only for dense input
                detect_rank = (s.abs() > 1e-5).sum(axis=-1)
                self.assertEqual(actual_rank * torch.ones(batches, device=device, dtype=torch.int64), detect_rank)
                U, S, V = torch.svd(A2)
                self.assertEqual(s[..., :actual_rank], S[..., :actual_rank])

        all_batches = [(), (1,), (3,), (2, 3)]
        for actual_rank, size, all_batches in [
                (2, (17, 4), all_batches),
                (2, (100, 4), all_batches),
                (6, (100, 40), all_batches),
                (12, (1000, 1000), [()]),
        ]:
            for batches in all_batches:
                for guess_rank in [
                        actual_rank,
                        actual_rank + 2,
                        actual_rank + 6,
                ]:
                    if guess_rank <= min(*size):
                        run_subtest(guess_rank, actual_rank, size, batches, device, torch.pca_lowrank)
                        run_subtest(guess_rank, actual_rank, size[::-1], batches, device, torch.pca_lowrank)

        # sparse input
        for guess_rank, size in [
                (4, (17, 4)), (4, (4, 17)), (16, (17, 17)),
                (21, (100, 40)), (20, (40, 100)), (600, (1000, 1000))]:
            for density in [0.005, 0.1]:
                run_subtest(guess_rank, None, size, (), device, torch.pca_lowrank, density=density)

        # jitting support
        jitted = torch.jit.script(torch.pca_lowrank)
        guess_rank, actual_rank, size, batches = 2, 2, (17, 4), ()
        run_subtest(guess_rank, actual_rank, size, batches, device, jitted)

    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(torch.double)
    def test_matrix_power(self, device, dtype):
        def run_test(M, sign=1):
            if sign == -1:
                M = M.inverse()
            MP2 = torch.matrix_power(M, 2)
            self.assertEqual(MP2, torch.matmul(M, M))

            MP3 = torch.matrix_power(M, 3)
            self.assertEqual(MP3, torch.matmul(MP2, M))

            MP4 = torch.matrix_power(M, 4)
            self.assertEqual(MP4, torch.matmul(MP2, MP2))

            MP6 = torch.matrix_power(M, 6)
            self.assertEqual(MP6, torch.matmul(MP3, MP3))

            MP0 = torch.matrix_power(M, 0)
            self.assertEqual(MP0, torch.eye(M.size(-2), dtype=dtype).expand_as(M))

        # Single matrix
        M = torch.randn(5, 5, dtype=dtype, device=device)
        run_test(M)

        # Batch matrices
        M = torch.randn(3, 3, 3, dtype=dtype, device=device)
        run_test(M)

        # Many batch matrices
        M = torch.randn(2, 3, 3, 3, dtype=dtype, device=device)
        run_test(M)

        # This is for negative powers
        from torch.testing._internal.common_utils import random_fullrank_matrix_distinct_singular_value
        M = random_fullrank_matrix_distinct_singular_value(5, dtype=dtype, device=device)
        run_test(M, sign=-1)

        M = random_fullrank_matrix_distinct_singular_value(3, 3, dtype=dtype, device=device)
        run_test(M, sign=-1)

        M = random_fullrank_matrix_distinct_singular_value(3, 2, 3, dtype=dtype, device=device)
        run_test(M, sign=-1)

    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(torch.float, torch.complex64)
    def test_matrix_exp_utils(self, device, dtype):
        # test linear combination
        def run_test(coeff_shape, data_shape):
            coeffs = torch.rand(*coeff_shape, device=device, dtype=torch.float)
            x = torch.rand(coeff_shape[1], *data_shape, device=device, dtype=dtype)

            res1 = torch._compute_linear_combination(x, coeffs)
            res2 = (x.unsqueeze(0) * coeffs.view(*coeff_shape, *([1] * len(data_shape)))).sum(1)
            self.assertEqual(res1, res2, atol=1e-5, rtol=0.0)

            # check `out=` version
            res3 = torch.zeros(coeff_shape[0], *data_shape, device=device, dtype=dtype)
            torch._compute_linear_combination(x, coeffs, out=res3)
            self.assertEqual(res1, res3, atol=1e-5, rtol=0.0)

            res4 = torch.ones(coeff_shape[0], *data_shape, device=device, dtype=dtype)
            torch._compute_linear_combination(x, coeffs, out=res4)
            self.assertEqual(res1, res4 - 1.0, atol=1e-5, rtol=0.0)

            res5 = torch.ones(coeff_shape[0], *data_shape, device=device, dtype=dtype)
            res5_clone = res5.clone()
            torch._compute_linear_combination(x, coeffs, out=res5)
            self.assertEqual(res1, res5 - res5_clone, atol=1e-5, rtol=0.0)

        run_test([1, 3], [2, 2])
        run_test([3, 1], [2, 2])
        run_test([1, 10], [10, 10])
        run_test([10, 1], [10, 10])
        run_test([5, 3], [2, 2])
        run_test([5, 3], [100, 100])
        run_test([3, 4], [3, 3, 3])
        run_test([3, 4], [3, 3, 3, 3])

    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(torch.float, torch.double, torch.complex64, torch.complex128)
    def test_matrix_exp_boundary_cases(self, device, dtype):

        with self.assertRaisesRegex(RuntimeError, "expected a tensor of floating or complex types"):
            torch.randn(3, 3).type(torch.int).matrix_exp()

        with self.assertRaisesRegex(RuntimeError, "with dim at least 2"):
            torch.randn(3).matrix_exp()

        with self.assertRaisesRegex(RuntimeError, "expected a tensor of squared matrices"):
            torch.randn(3, 2, 1).matrix_exp()

        # check 1x1 matrices
        x = torch.randn(3, 3, 1, 1)
        mexp = x.matrix_exp()
        self.assertEqual(mexp, x.exp())

    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(torch.float, torch.double)
    # Although tf32 is always disabled on matrix_exp, this test uses matmul,
    # which has tf32 on by default
    @with_tf32_off
    def test_matrix_exp_analytic(self, device, dtype):
        # check zero matrix
        x = torch.zeros(20, 20, dtype=dtype, device=device)
        self.assertTrue((x.matrix_exp() == torch.eye(20, 20, dtype=dtype, device=device)).all().item())

        def normalize_to_1_operator_norm(sample, desired_norm):
            sample_norm, _ = sample.abs().sum(-2).max(-1)
            sample_to_1_norm = sample / sample_norm.unsqueeze(-1).unsqueeze(-1)
            return sample_to_1_norm * desired_norm

        def gen_good_cond_number_matrices(*n):
            """
            Generates a diagonally-domimant matrix
            with the eigenvalues centered at 1
            and the radii at most (n[-1] - 1) / (n[-2] ** 2)
            """
            identity = torch.eye(n[-2], n[-1], dtype=dtype, device=device).expand(*n)
            x = torch.rand(*n, dtype=dtype, device=device) / (n[-1] ** 2)
            x = (x - x * identity) + identity
            return x

        def run_test(*n):
            if dtype == torch.float:
                thetas = [
                    1.192092800768788e-07,  # deg 1
                    5.978858893805233e-04,  # deg 2
                    5.116619363445086e-02,  # deg 4
                    5.800524627688768e-01,  # deg 8
                    1.461661507209034e+00,  # deg 12
                    3.010066362817634e+00   # deg 18
                ]
            else:  # if torch.double
                thetas = [
                    2.220446049250313e-16,  # deg 1
                    2.580956802971767e-08,  # deg 2
                    3.397168839976962e-04,  # deg 4
                    4.991228871115323e-02,  # deg 8
                    2.996158913811580e-01,  # deg 12
                    1.090863719290036e+00   # deg 18
                ]

            # generate input
            q = gen_good_cond_number_matrices(*n)
            qinv = torch.inverse(q)
            d = torch.randn(n[:-1], dtype=dtype, device=device)
            x = torch.matmul(q, torch.matmul(torch.diag_embed(d), qinv))
            x_norm, _ = x.abs().sum(-2).max(-1)

            # test simple analytic whatever norm generated
            mexp = x.matrix_exp()
            mexp_analytic = torch.matmul(
                q,
                torch.matmul(
                    torch.diag_embed(d.exp()),
                    qinv
                )
            )
            self.assertEqual(mexp, mexp_analytic, atol=1e-3, rtol=0.0)

            # generate norms to test different degree expansions
            sample_norms = []
            for i in range(len(thetas) - 1):
                sample_norms.append(0.5 * (thetas[i] + thetas[i + 1]))
            sample_norms = [thetas[0] / 2] + sample_norms + [thetas[-1] * 2]

            # matrices to equal norm
            for sample_norm in sample_norms:
                x_normalized = normalize_to_1_operator_norm(x, sample_norm)

                mexp = x_normalized.matrix_exp()
                mexp_analytic = torch.matmul(
                    q,
                    torch.matmul(
                        torch.diag_embed((d / x_norm.unsqueeze(-1) * sample_norm).exp()),
                        qinv
                    )
                )
                self.assertEqual(mexp, mexp_analytic, atol=1e-3, rtol=0.0)

        # single matrix
        run_test(2, 2)
        run_test(3, 3)
        run_test(4, 4)
        run_test(5, 5)
        run_test(100, 100)
        run_test(200, 200)

        # small batch of matrices
        run_test(3, 2, 2)
        run_test(3, 3, 3)
        run_test(3, 4, 4)
        run_test(3, 5, 5)
        run_test(3, 100, 100)
        run_test(3, 200, 200)

        # large batch of matrices
        run_test(3, 3, 2, 2)
        run_test(3, 3, 3, 3)
        run_test(3, 3, 4, 4)
        run_test(3, 3, 5, 5)
        run_test(3, 3, 100, 100)
        run_test(3, 3, 200, 200)

    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(torch.float, torch.double)
    def test_matrix_exp_batch(self, device, dtype):

        def run_test(*n):
            tensors_batch = torch.zeros(n, dtype=dtype, device=device)
            tensors_batch = tensors_batch.view(-1, n[-2], n[-1])

            num_matrices = tensors_batch.size(0)
            tensors_list = []
            for i in range(num_matrices):
                tensors_list.append(torch.randn(n[-2], n[-1], dtype=dtype, device=device))

            for i in range(num_matrices):
                tensors_batch[i, ...] = tensors_list[i]

            tensors_exp_map = (x.matrix_exp() for x in tensors_list)
            tensors_exp_batch = tensors_batch.matrix_exp()

            for i, tensor_exp in enumerate(tensors_exp_map):
                self.assertEqual(tensors_exp_batch[i, ...], tensor_exp)

        # small batch of matrices
        run_test(3, 2, 2)
        run_test(3, 3, 3)
        run_test(3, 4, 4)
        run_test(3, 5, 5)

        # large batch of matrices
        run_test(3, 3, 2, 2)
        run_test(3, 3, 3, 3)
        run_test(3, 3, 4, 4)
        run_test(3, 3, 5, 5)

    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(torch.float, torch.double)
    # Although tf32 is always disabled on matrix_exp, this test uses matmul,
    # which has tf32 on by default
    @with_tf32_off
    def test_matrix_exp_compare_with_taylor(self, device, dtype):

        def normalize_to_1_operator_norm(sample, desired_norm):
            sample_norm, _ = sample.abs().sum(-2).max(-1)
            sample_to_1_norm = sample / sample_norm.unsqueeze(-1).unsqueeze(-1)
            return sample_to_1_norm * desired_norm

        def gen_good_cond_number_matrices(*n):
            """
            Generates a diagonally-domimant matrix
            with the eigenvalues centered at 1
            and the radii at most (n[-1] - 1) / (n[-2] ** 2)
            """
            identity = torch.eye(n[-2], n[-1], dtype=dtype, device=device).expand(*n)
            x = torch.rand(*n, dtype=dtype, device=device) / (n[-1] ** 2)
            x = (x - x * identity) + identity
            return x

        def get_taylor_approximation(a, deg):
            identity = torch.eye(a.size(-2), a.size(-1), dtype=dtype, device=device).expand_as(a)
            res = identity
            taylor_term = identity

            for i in range(1, deg + 1):
                taylor_term = torch.matmul(a, taylor_term) / i
                res = res + taylor_term

            return res

        def scale_square(a, deg):
            if a.norm() < 1.0:
                return get_taylor_approximation(a, 12)
            else:
                s = int(torch.log2(a.norm()).ceil().item())
                b = a / (2 ** s)
                b = get_taylor_approximation(b, 18)
                for _ in range(s):
                    b = torch.matmul(b, b)
                return b

        def run_test(*n):
            degs = [1, 2, 4, 8, 12, 18]
            if dtype == torch.float:
                thetas = [
                    1.192092800768788e-07,  # deg 1
                    5.978858893805233e-04,  # deg 2
                    5.116619363445086e-02,  # deg 4
                    5.800524627688768e-01,  # deg 8
                    1.461661507209034e+00,  # deg 12
                    3.010066362817634e+00   # deg 18
                ]
            else:  # if torch.double
                thetas = [
                    2.220446049250313e-16,  # deg 1
                    2.580956802971767e-08,  # deg 2
                    3.397168839976962e-04,  # deg 4
                    4.991228871115323e-02,  # deg 8
                    2.996158913811580e-01,  # deg 12
                    1.090863719290036e+00   # deg 18
                ]

            # generate norms to test different degree expansions
            sample_norms = []
            for i in range(len(thetas) - 1):
                sample_norms.append(0.5 * (thetas[i] + thetas[i + 1]))
            sample_norms = [thetas[0] / 2] + sample_norms + [thetas[-1] * 2]
            degs = [degs[0]] + degs

            for sample_norm, deg in zip(sample_norms, degs):
                x = gen_good_cond_number_matrices(*n)
                x = normalize_to_1_operator_norm(x, sample_norm)

                mexp = x.matrix_exp()
                mexp_taylor = scale_square(x, deg)

                self.assertEqual(mexp, mexp_taylor, atol=1e-2, rtol=0.0)

        # single matrix
        run_test(2, 2)
        run_test(3, 3)
        run_test(4, 4)
        run_test(5, 5)

        # small batch of matrices
        run_test(3, 2, 2)
        run_test(3, 3, 3)
        run_test(3, 4, 4)
        run_test(3, 5, 5)

        # large batch of matrices
        run_test(3, 3, 2, 2)
        run_test(3, 3, 3, 3)
        run_test(3, 3, 4, 4)
        run_test(3, 3, 5, 5)

    @skipCUDAIfNoMagmaAndNoCusolver
    @skipCPUIfNoLapack
    def test_inverse(self, device):
        from torch.testing._internal.common_utils import random_fullrank_matrix_distinct_singular_value

        def test_inverse_helper(matrix, batches, n):
            identity = torch.eye(n, dtype=torch.float64, device=device)

            # correctness test, check matrix*matrix_inverse == identity
            matrix_inverse = torch.inverse(matrix)

            self.assertEqual(identity.expand_as(matrix), torch.matmul(matrix, matrix_inverse), atol=1e-8, rtol=0)
            self.assertEqual(identity.expand_as(matrix), torch.matmul(matrix_inverse, matrix), atol=1e-8, rtol=0)

            # torch.inverse with out and batches
            matrix_inverse_out = torch.empty(*batches, n, n, dtype=torch.float64, device=device)
            torch.inverse(matrix, out=matrix_inverse_out)
            self.assertEqual(matrix_inverse_out, matrix_inverse, atol=0, rtol=0)

            # batched matrices: 3+ dimensional tensors, check matrix_inverse same as single-inverse for each matrix
            if matrix.ndim > 2:
                expected_inv_list = []
                for mat in matrix.contiguous().view(-1, n, n):
                    expected_inv_list.append(torch.inverse(mat))
                expected_inv = torch.stack(expected_inv_list).view(*batches, n, n)
                self.assertEqual(matrix_inverse, expected_inv)

        for batches, n in product(
            [[], [1], [4], [2, 3], [32]],
            [5, 256]
        ):
            # large batch size and large matrix size will be tested in test_inverse_many_batches (slow test)
            if batches and batches[0] == 32 and n == 256:
                continue
            _matrices = random_fullrank_matrix_distinct_singular_value(n, *batches).to(device)
            test_inverse_helper(_matrices, batches, n)
            test_inverse_helper(_matrices.transpose(-2, -1), batches, n)
            test_inverse_helper(
                random_fullrank_matrix_distinct_singular_value(n * 2, *batches).to(device)
                .view(-1, n * 2, n * 2)[:, ::2, ::2].view(*batches, n, n),
                batches, n
            )

        # incorrect input test
        with self.assertRaisesRegex(RuntimeError, "must be batches of square matrices"):
            torch.inverse(torch.randn(2, 3, 4, 3))

        # test for zero-sized tensor
        def test_inverse_helper_zero_size(size):
            data = torch.zeros(*size, device=device)
            out = torch.inverse(data)
            self.assertTrue(out.size() == data.size())

        test_inverse_helper_zero_size([0, 0])
        test_inverse_helper_zero_size([3, 0, 0])
        test_inverse_helper_zero_size([0, 3, 3])

        from numpy.linalg import inv
        matrices = random_fullrank_matrix_distinct_singular_value(3, 2).to(device).permute(0, 2, 1)
        assert not matrices.is_contiguous()
        matrices_inverse = torch.inverse(matrices)
        expected_inv = torch.as_tensor(inv(matrices.cpu().numpy()))
        self.assertEqual(matrices_inverse, expected_inv.to(device))

    @skipCUDAIfNoMagmaAndNoCusolver
    @skipCPUIfNoLapack
    @onlyOnCPUAndCUDA   # TODO: XLA doesn't raise exception
    def test_inverse_singular(self, device):
        def helper(batch_dim, n):
            x = torch.eye(3, 3, dtype=torch.float, device=device).reshape((1, 3, 3)).repeat(batch_dim, 1, 1)
            x[n, -1, -1] = 0

            with self.assertRaisesRegex(RuntimeError, rf'For batch {n}: U\(3,3\) is zero'):
                torch.inverse(x)

        for params in [(1, 0), (2, 0), (2, 1), (4, 0), (4, 2), (10, 2)]:
            helper(*params)

    @slowTest
    @skipCUDAIfNoMagmaAndNoCusolver
    @skipCPUIfNoLapack
    def test_inverse_many_batches(self, device):
        from torch.testing._internal.common_utils import random_fullrank_matrix_distinct_singular_value

        def test_inverse_many_batches_helper(b, n):
            matrices = random_fullrank_matrix_distinct_singular_value(b, n, n).to(device)
            matrices_inverse = torch.inverse(matrices)
            self.assertEqual(torch.matmul(matrices_inverse, matrices),
                             torch.eye(b, dtype=torch.float64, device=device).expand_as(matrices))

        test_inverse_many_batches_helper(5, 256)
        test_inverse_many_batches_helper(3, 512)
        test_inverse_many_batches_helper(64, 64)

    @precisionOverride({torch.float32: 1e-3, torch.complex64: 1e-3})
    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypesIfCPU(torch.float32, torch.float64, torch.complex64, torch.complex128)
    @dtypesIfCUDA(torch.float32, torch.float64)
    def test_pinverse(self, device, dtype):
        from torch.testing._internal.common_utils import random_fullrank_matrix_distinct_singular_value as fullrank

        def run_test(M):
            # Testing against definition for pseudo-inverses
            MPI = torch.pinverse(M)
            if M.numel() > 0:
                self.assertEqual(M, M.matmul(MPI).matmul(M))
                self.assertEqual(MPI, MPI.matmul(M).matmul(MPI))
                self.assertEqual(M.matmul(MPI), (M.matmul(MPI)).transpose(-2, -1).conj())
                self.assertEqual(MPI.matmul(M), (MPI.matmul(M)).transpose(-2, -1).conj())
            else:
                self.assertEqual(M.shape, MPI.shape[:-2] + (MPI.shape[-1], MPI.shape[-2]))
        for sizes in [(5, 5), (3, 5, 5), (3, 7, 5, 5),  # square matrices
                      (3, 2), (5, 3, 2), (7, 5, 3, 2),  # fat matrices
                      (2, 3), (5, 2, 3), (7, 5, 2, 3),  # thin matrices
                      (0, 0), (0, 2), (2, 0), (3, 0, 0), (0, 3, 0), (0, 0, 3)]:  # zero numel matrices
            M = torch.randn(*sizes, dtype=dtype, device=device)
            run_test(M)

        # Test inverse and pseudo-inverse for invertible matrix
        for sizes in [(5, 5), (3, 5, 5), (3, 7, 5, 5)]:
            matsize = sizes[-1]
            batchdims = sizes[:-2]
            M = fullrank(matsize, *batchdims, dtype=dtype, device=device)
            self.assertEqual(torch.eye(matsize, dtype=dtype, device=device).expand(sizes), M.pinverse().matmul(M),
                             atol=1e-7, rtol=0, msg='pseudo-inverse for invertible matrix')

    # TODO: once there is more support for complex dtypes on GPU, they shall be added to above test
    # particularly when RuntimeError: _th_bmm_out not supported on CUDAType for ComplexFloat is fixed
    @unittest.expectedFailure
    @onlyCUDA
    @skipCUDAIfNoMagma
    @dtypes(torch.complex64, torch.complex128)
    def test_pinverse_complex_xfailed(self, device, dtype):
        size = (3, 5, 5)
        M = torch.randn(*sizes, dtype=dtype, device=device)
        MPI = torch.pinverse(M)
        self.assertEqual(M, M.matmul(MPI).matmul(M))

    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    def test_matrix_rank(self, device):
        a = torch.eye(10, device=device)
        self.assertEqual(torch.matrix_rank(a).item(), 10)
        self.assertEqual(torch.matrix_rank(a, True).item(), 10)

        a[5, 5] = 0
        self.assertEqual(torch.matrix_rank(a).item(), 9)
        self.assertEqual(torch.matrix_rank(a, True).item(), 9)

        a = torch.randn(24, 42, device=device)
        self.assertEqual(torch.matrix_rank(a), torch.matrix_rank(a.t()))
        aaT = torch.mm(a, a.t())
        self.assertEqual(torch.matrix_rank(aaT), torch.matrix_rank(aaT, True))
        aTa = torch.mm(a.t(), a)
        self.assertEqual(torch.matrix_rank(aTa), torch.matrix_rank(aTa, True))

        from numpy.linalg import matrix_rank
        a = torch.randn(35, 75, device=device)
        self.assertEqual(torch.matrix_rank(a).item(), matrix_rank(a.cpu().numpy()))
        self.assertEqual(torch.matrix_rank(a, 0.01).item(), matrix_rank(a.cpu().numpy(), 0.01))

        aaT = torch.mm(a, a.t())
        self.assertEqual(torch.matrix_rank(aaT).item(), matrix_rank(aaT.cpu().numpy()))
        self.assertEqual(torch.matrix_rank(aaT, 0.01).item(), matrix_rank(aaT.cpu().numpy(), 0.01))

        if np.lib.NumpyVersion(np.__version__) >= '1.14.0':
            self.assertEqual(torch.matrix_rank(aaT, True).item(), matrix_rank(aaT.cpu().numpy(), True))
            self.assertEqual(torch.matrix_rank(aaT, 0.01, True).item(),
                             matrix_rank(aaT.cpu().numpy(), 0.01, True))

    def _test_svd_helper(self, shape, some, col_maj, device, dtype):
        cpu_tensor = torch.randn(shape, device='cpu').to(dtype)
        device_tensor = cpu_tensor.to(device=device)
        if col_maj:
            cpu_tensor = cpu_tensor.t()
            device_tensor = device_tensor.t()
        cpu_result = torch.svd(cpu_tensor, some=some)
        device_result = torch.svd(device_tensor, some=some)
        m = min(cpu_tensor.shape[-2:])
        # torch.svd returns torch.return_types.svd which is a tuple of (U, V, S).
        # - When some==False, U[..., m:] can be arbitrary.
        # - When some==True, U shape: [..., m], V shape: [m, m]
        # - Signs are not deterministic. If the sign of a column of U is changed
        #   then the corresponding column of the V has to be changed.
        # Thus here we only compare result[..., :m].abs() from CPU and device.
        for x, y in zip(cpu_result, device_result):
            self.assertEqual(x[..., :m].abs(), y[..., :m].abs(), atol=1e-5, rtol=0)

    _float_types_no_half = [torch.float, torch.double]
    _complex_types = [torch.cfloat, torch.cdouble]

    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(*(_float_types_no_half + _complex_types))
    def test_svd_square(self, device, dtype):
        self._test_svd_helper((10, 10), True, False, device, dtype)

    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(*_float_types_no_half)
    def test_svd_square_col_maj(self, device, dtype):
        self._test_svd_helper((10, 10), True, True, device, dtype)

    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(*_float_types_no_half)
    def test_svd_tall_some(self, device, dtype):
        self._test_svd_helper((20, 5), True, False, device, dtype)

    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(*_float_types_no_half)
    def test_svd_tall_all(self, device, dtype):
        self._test_svd_helper((20, 5), False, False, device, dtype)

    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(*_float_types_no_half)
    def test_svd_tall_some_col_maj(self, device, dtype):
        self._test_svd_helper((5, 20), True, True, device, dtype)

    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(*_float_types_no_half)
    def test_svd_tall_all_col_maj(self, device, dtype):
        self._test_svd_helper((5, 20), False, True, device, dtype)

    @onlyCPU
    @dtypes(*(torch.testing.get_all_complex_dtypes() + [torch.float, torch.double]))
    def test_addbmm(self, device, dtype):
        # num_batches = 10
        # M, N, O = 12, 8, 5
        num_batches = 2
        M, N, O = 2, 3, 4
        b1 = torch.randn(num_batches, M, N, dtype=dtype, device=device)
        b2 = torch.randn(num_batches, N, O, dtype=dtype, device=device)
        res = torch.bmm(b1, b2)
        res2 = torch.tensor((), dtype=dtype, device=device).resize_as_(res[0]).zero_()
        res3 = torch.tensor((), dtype=dtype, device=device).resize_as_(res[0]).zero_()

        res2.addbmm_(b1, b2)
        self.assertEqual(res2, res.sum(0, False))
        res3.copy_(res2)

        with self.maybeWarnsRegex(
                UserWarning, "This overload of addbmm_ is deprecated"):
            res2.addbmm_(1, b1, b2)
        self.assertEqual(res2, res.sum(0, False) * 2),
        res3.addbmm_(b1, b2, beta=1)
        self.assertEqual(res2, res3)

        with self.maybeWarnsRegex(
                UserWarning, "This overload of addbmm_ is deprecated"):
            res2.addbmm_(1., .5, b1, b2)
        self.assertEqual(res2, res.sum(0, False) * 2.5)
        res3.addbmm_(b1, b2, beta=1., alpha=.5)
        self.assertEqual(res2, res3)

        with self.maybeWarnsRegex(
                UserWarning, "This overload of addbmm is deprecated"):
            self.assertEqual(res2, torch.addbmm(1, res2, 0, b1, b2))

        res4 = torch.addbmm(res2, b1, b2, beta=1, alpha=.5)
        self.assertEqual(res4, res.sum(0, False) * 3),

        res5 = torch.addbmm(res2, b1, b2, beta=0, alpha=1)
        self.assertEqual(res5, res.sum(0, False))

        res6 = torch.addbmm(res2, b1, b2, beta=.1, alpha=.5)
        self.assertEqual(res6, res2 * .1 + .5 * res.sum(0)),

    @onlyCPU
    @dtypes(*(torch.testing.get_all_complex_dtypes() + [torch.float, torch.double]))
    def test_baddbmm(self, device, dtype):
        num_batches = 10
        M, N, O = 12, 8, 5
        b1 = torch.randn(num_batches, M, N, dtype=dtype, device=device)
        b2 = torch.randn(num_batches, N, O, dtype=dtype, device=device)
        res = torch.bmm(b1, b2)
        res2 = torch.tensor((), dtype=dtype, device=device).resize_as_(res).zero_()
        res3 = torch.tensor((), dtype=dtype, device=device).resize_as_(res).zero_()

        res2.baddbmm_(b1, b2)
        self.assertEqual(res2, res)
        res3.copy_(res2)

        with self.maybeWarnsRegex(
                UserWarning, "This overload of baddbmm_ is deprecated"):
            res2.baddbmm_(1, b1, b2)
        self.assertEqual(res2, res * 2)
        res3.baddbmm_(b1, b2, beta=1)
        self.assertEqual(res3, res2)

        with self.maybeWarnsRegex(
                UserWarning, "This overload of baddbmm_ is deprecated"):
            res2.baddbmm_(1, .5, b1, b2)
        self.assertEqual(res2, res * 2.5)
        res3.baddbmm_(b1, b2, beta=1, alpha=.5)
        self.assertEqual(res3, res2)


        with self.maybeWarnsRegex(
                UserWarning, "This overload of baddbmm is deprecated"):
            self.assertEqual(torch.baddbmm(1, res2, 0, b1, b2), res2)

        res4 = torch.baddbmm(res2, b1, b2, beta=1, alpha=.5)
        self.assertEqual(res4, res * 3, atol=2e-5, rtol=0)

        res5 = torch.baddbmm(res2, b1, b2, beta=0, alpha=1)
        self.assertEqual(res5, res)

        res6 = torch.baddbmm(res2, b1, b2, beta=.1, alpha=.5)
        self.assertEqual(res6, res2 * .1 + res * .5)

    @slowTest
    @onlyOnCPUAndCUDA
    @dtypes(torch.float32, torch.float64, torch.bfloat16, torch.int32, torch.int64, torch.cfloat, torch.cdouble)
    @dtypesIfCUDA(torch.float32, torch.float64, torch.cfloat, torch.cdouble)
    @tf32_on_and_off(0.01)
    def test_mm(self, device, dtype):
        def _test_mm(n, m, p, dtype, genf):
            # helper function
            def matrixmultiply(mat1, mat2):
                n = mat1.size(0)
                m = mat1.size(1)
                p = mat2.size(1)
                res = torch.zeros(n, p, dtype=dtype, device=device)
                for i, j in iter_indices(res):
                    res[i, j] = sum(mat1[i, k] * mat2[k, j] for k in range(m))
                return res

            # contiguous case
            mat1 = genf(n, m)
            mat2 = genf(m, p)
            res = torch.mm(mat1, mat2)

            res2 = matrixmultiply(mat1, mat2)
            self.assertEqual(res, res2)

            # non contiguous case 1
            mat1 = genf(n, m)
            mat2 = genf(p, m).t()
            res = torch.mm(mat1, mat2)

            res2 = matrixmultiply(mat1, mat2)
            self.assertEqual(res, res2)

            # non contiguous case 2
            mat1 = genf(m, n).t()
            mat2 = genf(m, p)
            res = torch.mm(mat1, mat2)

            res2 = matrixmultiply(mat1, mat2)
            self.assertEqual(res, res2)

            # non contiguous case 3
            mat1 = genf(m, n).t()
            mat2 = genf(p, m).t()
            res = torch.mm(mat1, mat2)

            res2 = matrixmultiply(mat1, mat2)
            self.assertEqual(res, res2)

            # test with zero stride
            mat1 = genf(n, m)
            mat2 = genf(m, 1).expand(m, p)
            res = torch.mm(mat1, mat2)

            res2 = matrixmultiply(mat1, mat2)
            self.assertEqual(res, res2)

            # explicitly exercise the _out variant in torch.mm().
            # contiguous case
            mat1 = genf(n, m)
            mat2 = genf(m, p)
            res = genf(n, p)
            torch.mm(mat1, mat2, out=res)

            res2 = matrixmultiply(mat1, mat2)
            self.assertEqual(res, res2)

            # explicitly exercise the _out variant in torch.mm().
            # non contiguous case 3
            mat1 = genf(m, n).t()
            mat2 = genf(p, m).t()
            res = genf(n, p)
            torch.mm(mat1, mat2, out=res)

            res2 = matrixmultiply(mat1, mat2)
            self.assertEqual(res, res2)

        def genf_int(x, y):
            return torch.randint(0, 100, (x, y), dtype=dtype, device=device)

        def genf_bfloat(x, y):
            return torch.randn(x, y, dtype=torch.float32, device=device).to(dtype)

        def genf_float(x, y):
            return torch.randn(x, y, dtype=dtype, device=device)

        for (n, m, p) in [(20, 10, 5), (15, 5, 10), (5, 18, 10)]:
            if (dtype == torch.int32) or (dtype == torch.int64):
                genf = genf_int
            elif (dtype == torch.bfloat16):
                genf = genf_bfloat
            else:
                genf = genf_float

            _test_mm(n, m, p, dtype, genf)

    @onlyOnCPUAndCUDA
    @dtypes(torch.float32, torch.float64)
    def test_strided_mm_bmm(self, device, dtype):
        # Tests strided view case with stride smaller than corresponding dimension size
        x = torch.tensor([[1., 2., 3.], [4., 5., 6.]], dtype=dtype, device=device)
        new_shape = [2, 2, 2]
        new_stride = [3, 1, 1]
        sx = torch.as_strided(x, size=new_shape, stride=new_stride)

        torch_fn = lambda x: torch.bmm(x, x)  # noqa: E731
        np_fn = lambda x: np.matmul(x, x)  # noqa: E731
        self.compare_with_numpy(torch_fn, np_fn, sx)

        torch_fn = lambda x: torch.mm(x, x)  # noqa: E731
        self.compare_with_numpy(torch_fn, np_fn, sx[0])

    @onlyCPU
    @dtypes(*(torch.testing.get_all_complex_dtypes() + [torch.float, torch.double]))
    def test_bmm(self, device, dtype):
        num_batches = 10
        M, N, O = 23, 8, 12
        b1 = torch.randn(num_batches, M, N, dtype=dtype, device=device)
        b2 = torch.randn(num_batches, N, O, dtype=dtype, device=device)
        res = torch.bmm(b1, b2)
        for i in range(num_batches):
            r = torch.mm(b1[i], b2[i])
            self.assertEqual(r, res[i])
        if torch.cuda.is_available():
            # check that mixed arguments are rejected
            self.assertRaises(RuntimeError, lambda: torch.bmm(b1, b2.cuda()))
            self.assertRaises(RuntimeError, lambda: torch.bmm(b1.cuda(), b2))

    @onlyCPU
    @skipCPUIfNoLapack
    @dtypes(torch.double)
    def test_eig(self, device, dtype):
        a = torch.Tensor(((1.96, 0.00, 0.00, 0.00, 0.00),
                          (-6.49, 3.80, 0.00, 0.00, 0.00),
                          (-0.47, -6.39, 4.17, 0.00, 0.00),
                          (-7.20, 1.50, -1.51, 5.70, 0.00),
                          (-0.65, -6.34, 2.67, 1.80, -7.10))).t().contiguous().to(dtype=dtype, device=device)
        e = torch.eig(a)[0]
        ee, vv = torch.eig(a, True)
        te = torch.tensor((), dtype=dtype, device=device)
        tv = torch.tensor((), dtype=dtype, device=device)
        eee, vvv = torch.eig(a, True, out=(te, tv))
        self.assertEqual(e, ee, atol=1e-12, rtol=0)
        self.assertEqual(ee, eee, atol=1e-12, rtol=0)
        self.assertEqual(ee, te, atol=1e-12, rtol=0)
        self.assertEqual(vv, vvv, atol=1e-12, rtol=0)
        self.assertEqual(vv, tv, atol=1e-12, rtol=0)

        # test reuse
        X = torch.randn(4, 4, dtype=dtype, device=device)
        X = torch.mm(X.t(), X)
        e = torch.zeros(4, 2, dtype=dtype, device=device)
        v = torch.zeros(4, 4, dtype=dtype, device=device)
        torch.eig(X, True, out=(e, v))
        Xhat = torch.mm(torch.mm(v, torch.diag(e.select(1, 0))), v.t())
        self.assertEqual(X, Xhat, atol=1e-8, rtol=0, msg='VeV\' wrong')
        self.assertFalse(v.is_contiguous(), 'V is contiguous')

        torch.eig(X, True, out=(e, v))
        Xhat = torch.mm(v, torch.mm(e.select(1, 0).diag(), v.t()))
        self.assertEqual(X, Xhat, atol=1e-8, rtol=0, msg='VeV\' wrong')
        self.assertFalse(v.is_contiguous(), 'V is contiguous')

        # test non-contiguous
        X = torch.randn(4, 4, dtype=dtype, device=device)
        X = torch.mm(X.t(), X)
        e = torch.zeros(4, 2, 2, dtype=dtype, device=device)[:, 1]
        v = torch.zeros(4, 2, 4, dtype=dtype, device=device)[:, 1]
        self.assertFalse(v.is_contiguous(), 'V is contiguous')
        self.assertFalse(e.is_contiguous(), 'E is contiguous')
        torch.eig(X, True, out=(e, v))
        Xhat = torch.mm(torch.mm(v, torch.diag(e.select(1, 0))), v.t())
        self.assertEqual(X, Xhat, atol=1e-8, rtol=0, msg='VeV\' wrong')

        # test invalid input
        self.assertRaisesRegex(
            RuntimeError,
            'A should be 2 dimensional',
            lambda: torch.eig(torch.ones((2))))
        self.assertRaisesRegex(
            RuntimeError,
            'A should be square',
            lambda: torch.eig(torch.ones((2, 3))))
        self.assertRaisesRegex(
            RuntimeError,
            'A should not contain infs or NaNs',
            lambda: torch.eig(np.inf * torch.ones((2, 2))))
        self.assertRaisesRegex(
            RuntimeError,
            'A should not contain infs or NaNs',
            lambda: torch.eig(np.nan * torch.ones((2, 2))))

    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(torch.double)
    def test_lobpcg_basic(self, device, dtype):
        self._test_lobpcg_method(device, dtype, 'basic')

    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(torch.double)
    def test_lobpcg_ortho(self, device, dtype):
        self._test_lobpcg_method(device, dtype, 'ortho')

    def _test_lobpcg_method(self, device, dtype, method):
        from torch.testing._internal.common_utils import random_symmetric_pd_matrix, random_sparse_pd_matrix
        from torch._linalg_utils import matmul, qform
        from torch._lobpcg import lobpcg

        def test_tracker(worker):
            k = worker.iparams['k']
            nc = worker.ivars['converged_count']
            if k <= nc:
                tol = worker.fparams['tol']
                rerr = worker.tvars['rerr']
                X = worker.X
                E = worker.E
                B = worker.B
                A = worker.A
                dtype = X.dtype
                device = X.device

                # Check convergence
                self.assertLessEqual(rerr[:k].max(), tol)

                # Check B-orthogonality
                I = torch.eye(k, k, dtype=dtype, device=device)
                self.assertEqual(qform(B, X[:, :k]), I)

                # Check block equation
                self.assertEqual(qform(A, X[:, :k]) / E[:k], I, atol=0.2, rtol=0)

        orig_lobpcg = lobpcg

        def lobpcg(*args, **kwargs):
            kwargs['tracker'] = test_tracker
            kwargs['niter'] = 1000
            kwargs['method'] = method
            kwargs['tol'] = 1e-8
            return orig_lobpcg(*args, **kwargs)
        prec = 5e-4

        # check dense input
        mm = torch.matmul
        for batches in [(), (2,), (2, 3)]:
            for m, n, k in [
                    (9, 3, 1),
                    (9, 3, 2),
                    (9, 2, 2),
                    (100, 15, 5),
            ]:
                # skip tests that are known to fail with the basic
                # LOBPCG method due to calling cholesky on singular
                # input
                if method == 'basic' and (m, n, k) in [(9, 2, 2), (100, 15, 5)]:
                    continue
                A = random_symmetric_pd_matrix(m, *batches, device=device, dtype=dtype)
                B = random_symmetric_pd_matrix(m, *batches, device=device, dtype=dtype)

                # classical eigenvalue problem, smallest eigenvalues
                E, V = lobpcg(A, k=k, n=n, largest=False)
                self.assertEqual(E.shape, batches + (k,))
                self.assertEqual(V.shape, batches + (m, k))
                self.assertEqual(matmul(A, V), mm(V, E.diag_embed()), atol=prec, rtol=0)
                e = torch.symeig(A)[0]
                e_smallest = e[..., :k]
                self.assertEqual(E, e_smallest)

                # classical eigenvalue problem, largest eigenvalues
                E, V = lobpcg(A, k=k, n=n, largest=True)
                e_largest, _ = torch.sort(e[..., -k:], descending=True)
                self.assertEqual(E, e_largest, atol=prec, rtol=0)
                self.assertEqual(matmul(A, V), mm(V, E.diag_embed()), atol=prec, rtol=0)

                # generalized eigenvalue problem, smallest eigenvalues
                E, V = lobpcg(A, B=B, k=k, n=n, largest=False)
                self.assertEqual(matmul(A, V), mm(matmul(B, V), E.diag_embed()), atol=prec, rtol=0)

                # generalized eigenvalue problem, largest eigenvalues
                E, V = lobpcg(A, B=B, k=k, n=n, largest=True)
                self.assertEqual(matmul(A, V) / E.max(), mm(matmul(B, V), (E / E.max()).diag_embed()),
                                 atol=prec, rtol=0)

        # check sparse input
        for m, n, k, density in [
                (5, 1, 1, 0.8),
                (9, 3, 2, 0.5),
                (100, 1, 1, 0.1),
                (1000, 7, 3, 0.01),
        ]:
            # skip tests that are known to fail with the basic LOBCG
            # method due to insufficient accuracy
            if method == 'basic' and (m, n, k, density) in [(1000, 7, 3, 0.01)]:
                continue
            A = random_sparse_pd_matrix(m, density=density, device=device, dtype=dtype)
            B = random_sparse_pd_matrix(m, density=density, device=device, dtype=dtype)
            A_eigenvalues = torch.arange(1, m + 1, dtype=dtype) / m
            e_smallest = A_eigenvalues[..., :k]
            e_largest, _ = torch.sort(A_eigenvalues[..., -k:], descending=True)

            # classical eigenvalue problem, smallest eigenvalues
            E, V = lobpcg(A, k=k, n=n, largest=False)
            self.assertEqual(E, e_smallest)
            self.assertEqual(matmul(A, V), mm(V, E.diag_embed()), atol=prec, rtol=0)

            # classical eigenvalue problem, largest eigenvalues
            E, V = lobpcg(A, k=k, n=n, largest=True)
            self.assertEqual(matmul(A, V), mm(V, E.diag_embed()), atol=prec, rtol=0)
            self.assertEqual(E, e_largest)

            # generalized eigenvalue problem, smallest eigenvalues
            E, V = lobpcg(A, B=B, k=k, n=n, largest=False)
            self.assertEqual(matmul(A, V), matmul(B, mm(V, E.diag_embed())), atol=prec, rtol=0)

            # generalized eigenvalue problem, largest eigenvalues
            E, V = lobpcg(A, B=B, k=k, n=n, largest=True)
            self.assertEqual(matmul(A, V) / E.max(), mm(matmul(B, V), (E / E.max()).diag_embed()),
                             atol=prec, rtol=0)

    @skipCPUIfNoLapack
    @onlyCPU
    @dtypes(torch.double)
    def test_lobpcg_torchscript(self, device, dtype):
        from torch.testing._internal.common_utils import random_sparse_pd_matrix
        from torch._linalg_utils import matmul as mm

        lobpcg = torch.jit.script(torch.lobpcg)

        m = 500
        k = 5
        A1 = random_sparse_pd_matrix(m, density=2.0 / m, device=device, dtype=dtype)
        X1 = torch.randn((m, k), dtype=dtype, device=device)
        E1, V1 = lobpcg(A1, X=X1)
        eq_err = torch.norm((mm(A1, V1) - V1 * E1), 2) / E1.max()
        self.assertLess(eq_err, 1e-6)

    @unittest.skipIf(not TEST_SCIPY or (TEST_SCIPY and scipy.__version__ < '1.4.1'), "Scipy not found or older than 1.4.1")
    @skipCPUIfNoLapack
    @onlyCPU
    @dtypes(torch.double)
    def test_lobpcg_scipy(self, device, dtype):
        """Compare torch and scipy.sparse.linalg implementations of lobpcg
        """
        import time
        import scipy
        from torch.testing._internal.common_utils import random_sparse_pd_matrix
        from torch._linalg_utils import matmul as mm
        from scipy.sparse.linalg import lobpcg as scipy_lobpcg
        import scipy.sparse

        def toscipy(A):
            if A.layout == torch.sparse_coo:
                values = A.coalesce().values().cpu().numpy().copy()
                indices = A.coalesce().indices().cpu().numpy().copy()
                return scipy.sparse.coo_matrix((values, (indices[0], indices[1])), A.shape)
            return A.cpu().numpy().copy()

        niter = 1000
        repeat = 10
        m = 500   # size of the square matrix
        k = 7     # the number of requested eigenpairs
        A1 = random_sparse_pd_matrix(m, density=2.0 / m, device=device, dtype=dtype)
        B1 = random_sparse_pd_matrix(m, density=2.0 / m, device=device, dtype=dtype)
        X1 = torch.randn((m, k), dtype=dtype, device=device)

        A2 = toscipy(A1)
        B2 = toscipy(B1)
        X2 = toscipy(X1)

        lambdas1 = []

        def tracker(worker):
            lambdas1.append(worker.E[:])

        tol = 1e-8
        # tol for scipy lobpcg will be choosed so that the number of
        # iterations will be equal or very close to pytorch lobpcg
        # (that is around 170-180)

        # Standard eigenvalue problem
        E1, V1 = torch.lobpcg(A1, X=X1, niter=niter, largest=True, tracker=tracker, tol=tol)
        E2, V2, lambdas2 = scipy_lobpcg(A2, X2, maxiter=niter, largest=True, retLambdaHistory=True, tol=1.1 * tol)
        iters1 = len(lambdas1)
        iters2 = len(lambdas2)
        self.assertLess(abs(iters1 - iters2), 0.05 * max(iters1, iters2))

        E2a, V2a = scipy_lobpcg(A2, X2, maxiter=niter, largest=False)

        eq_err = torch.norm((mm(A1, V1) - V1 * E1), 2) / E1.max()
        eq_err_scipy = (abs(A2.dot(V2) - V2 * E2)**2).sum() ** 0.5 / E2.max()
        self.assertLess(eq_err, 1e-6)        # std
        self.assertLess(eq_err_scipy, 1e-6)  # std

        self.assertEqual(E1, torch.from_numpy(E2.copy()))

        # Generalized eigenvalue problem
        lambdas1 = []

        def tracker(worker):
            lambdas1.append(worker.E[:])

        E1, V1 = torch.lobpcg(A1, B=B1, X=X1, niter=niter, largest=True, tracker=tracker, tol=tol)
        E2, V2, lambdas2 = scipy_lobpcg(A2, X2, B=B2, maxiter=niter, largest=True, retLambdaHistory=True, tol=39 * tol)
        E2a, V2a = scipy_lobpcg(A2, X2, B=B2, maxiter=niter, largest=False)
        iters1 = len(lambdas1)
        iters2 = len(lambdas2)
        self.assertLess(abs(iters1 - iters2), 0.05 * max(iters1, iters2))

        eq_err = torch.norm((mm(A1, V1) - mm(B1, V1) * E1), 2) / E1.max()
        eq_err_scipy = (abs(A2.dot(V2) - B2.dot(V2) * E2)**2).sum() ** 0.5 / E2.max()
        self.assertLess(eq_err, 1e-6)        # general
        self.assertLess(eq_err_scipy, 1e-6)  # general

        self.assertEqual(E1, torch.from_numpy(E2.copy()))

        # Timings
        elapsed_ortho = 0
        elapsed_ortho_general = 0
        elapsed_scipy = 0
        elapsed_general_scipy = 0
        for i in range(repeat):
            start = time.time()
            torch.lobpcg(A1, X=X1, niter=niter, method='ortho', tol=tol)
            end = time.time()
            elapsed_ortho += end - start

            start = time.time()
            torch.lobpcg(A1, X=X1, B=B1, niter=niter, method='ortho', tol=tol)
            end = time.time()
            elapsed_ortho_general += end - start

            start = time.time()
            scipy_lobpcg(A2, X2, maxiter=niter, tol=1.1 * tol)
            end = time.time()
            elapsed_scipy += end - start

            start = time.time()
            scipy_lobpcg(A2, X2, B=B2, maxiter=niter, tol=39 * tol)
            end = time.time()
            elapsed_general_scipy += end - start

        elapsed_ortho_ms = 1000.0 * elapsed_ortho / repeat
        elapsed_ortho_general_ms = 1000.0 * elapsed_ortho_general / repeat
        elapsed_scipy_ms = 1000.0 * elapsed_scipy / repeat
        elapsed_general_scipy_ms = 1000.0 * elapsed_general_scipy / repeat

        print('''
CPU timings: torch.lobpcg vs scipy.sparse.linalg.lobpcg
-------------------------------------------------------
              | standard    | generalized | method
torch.lobpcg  | {:10.2f}  | {:10.2f}  | ortho
scipy_lobpcg  | {:10.2f}  | {:10.2f}  | N/A
-(input size: {:4}, eigenpairs:{:2}, units: ms per call)-
        '''.format(elapsed_ortho_ms, elapsed_ortho_general_ms,
                   elapsed_scipy_ms, elapsed_general_scipy_ms,
                   m, k))

        # Handling of very small tolerence
        tol = 1e-100

        lambdas1 = []

        def tracker(worker):
            lambdas1.append(worker.E[:])

        E1, V1 = torch.lobpcg(A1, X=X1, niter=niter, largest=True, tracker=tracker, tol=tol)
        iters1 = len(lambdas1)
        eq_err = torch.norm((mm(A1, V1) - V1 * E1), 2) / E1.max()

        try:
            E2, V2, lambdas2 = scipy_lobpcg(A2, X2, maxiter=niter, largest=True, retLambdaHistory=True, tol=tol)
            iters2 = len(lambdas2)
            eq_err_scipy = (abs(A2.dot(V2) - V2 * E2)**2).sum() ** 0.5 / E2.max()
        except Exception as msg:
            print('Calling scipy_lobpcg failed [standard]:', msg)
            iters2 = -1
            eq_err_scipy = -1

        lambdas1 = []

        def tracker(worker):
            lambdas1.append(worker.E[:])

        E1, V1 = torch.lobpcg(A1, X=X1, B=B1, niter=niter, largest=True, tracker=tracker, tol=tol)
        iters1_general = len(lambdas1)
        eq_err_general = torch.norm((mm(A1, V1) - mm(B1, V1) * E1), 2) / E1.max()

        try:
            E2, V2, lambdas2 = scipy_lobpcg(A2, X2, B=B2, maxiter=niter, largest=True, retLambdaHistory=True, tol=tol)
            iters2_general = len(lambdas2)
            eq_err_general_scipy = (abs(A2.dot(V2) - B2.dot(V2) * E2)**2).sum() ** 0.5 / E2.max()
        except Exception as msg:
            print('Calling scipy_lobpcg failed [generalized]:', msg)
            iters2_general = -1
            eq_err_general_scipy = -1

        print('''\
Handling of small tol={:6.0e}: torch.lobpcg vs scipy.sparse.linalg.lobpcg
----------------------------------------------------------------------------
              | standard    | generalized |  niter | method
torch.lobpcg  | {:10.2e}  | {:10.2e}  | {:6} | ortho
scipy_lobpcg  | {:10.2e}  | {:10.2e}  | {:6} | N/A
---(input size: {:4}, eigenpairs:{:2}, units: relative error, maxiter={:4})---
'''.format(tol, eq_err, eq_err_general, iters1, eq_err_scipy, eq_err_general_scipy, iters2, m, k, niter))

    def _test_addmm_addmv(self, f, t, m, v, *, alpha=None, beta=None, transpose_out=False):
        dtype = t.dtype
        numpy_dtype = dtype
        if dtype in {torch.bfloat16}:
            numpy_dtype = torch.float
        if dtype.is_complex:
            alpha = 0.9 + 0.3j if alpha is None else alpha
            beta = 0.5 + 0.6j if beta is None else beta
        else:
            alpha = 1.2 if alpha is None else alpha
            beta = 0.8 if beta is None else beta
        res1 = f(t, m, v, alpha=alpha, beta=beta)
        res2 = torch.full_like(res1, math.nan)
        if transpose_out:
            res2 = res2.t().clone(memory_format=torch.contiguous_format).t()
        f(t, m, v, alpha=alpha, beta=beta, out=res2)
        res3 = alpha * (m.to(numpy_dtype).cpu().numpy() @ v.to(numpy_dtype).cpu().numpy())
        if beta != 0:
            res3 += (beta * t).to(numpy_dtype).cpu().numpy()
        res3 = torch.from_numpy(res3).to(dtype)
        self.assertEqual(res1, res2)
        self.assertEqual(res1, res3)

    @precisionOverride({torch.bfloat16: 1e-0, torch.half: 5e-4, torch.float: 1e-4, torch.double: 1e-8,
                        torch.cfloat: 1e-4, torch.cdouble: 1e-8})
    @dtypesIfCUDA(*torch.testing.get_all_complex_dtypes(),
                  *([torch.float32, torch.float64, torch.bfloat16]
                    if TEST_WITH_ROCM else torch.testing.get_all_fp_dtypes(include_bfloat16=AMPERE_OR_ROCM)))
    @dtypes(torch.bfloat16, torch.float, torch.double, torch.cfloat, torch.cdouble)
    def test_addmv(self, device, dtype):
        # have to use torch.randn(...).to(bfloat16) instead of
        # torch.randn(..., dtype=bfloat16). randn does not support
        # bfloat16 yet.
        ts = [
            torch.randn(10, device=device).to(dtype),
            torch.randn(1, device=device).to(dtype).expand(10),
        ]
        vs = [
            torch.randn(100, device=device).to(dtype),
            torch.ones(1, device=device).to(dtype).expand(100),  # to reduce errors for low precision
        ]
        ms = [
            # 0d
            torch.ones((), device=device).to(dtype).expand(10, 100),  # to reduce errors for low precision
            # 1d
            torch.randn((1, 100), device=device).to(dtype).expand(10, 100),
            # this initialization reduces errors for low precision for broadcasted matrices
            # by making sure that intermediate and result values are exactly representable
            # in low precision type
            torch.randint(3, (10, 1), dtype=torch.float, device=device).to(dtype).expand(10, 100),
            # 2d
            torch.randn((10, 100), device=device).to(dtype),
            torch.randn((100, 10), device=device).to(dtype).t(),
        ]
        for m, v, t in product(ms, vs, ts):
            self._test_addmm_addmv(torch.addmv, t, m, v)
        # Test beta=0, t=nan
        t = torch.full((10,), math.nan, device=device).to(dtype)
        for m, v in product(ms, vs):
            self._test_addmm_addmv(torch.addmv, t, m, v, beta=0)

    @dtypesIfCUDA(*([torch.half, torch.float, torch.double]
                    + ([torch.bfloat16] if TEST_WITH_ROCM else [])))
    @dtypes(torch.float, torch.double)
    def test_addmv_rowmajor_colmajor_incx_incy_lda(self, device, dtype):
        # tests (o, s)*(s).  o is output size, s is summed size.
        o = 5
        s = 3
        a_data = torch.arange(1, o * s + 1, device=device, dtype=dtype).view(o, s)
        x_data = torch.arange(1, s + 1, 1, device=device, dtype=dtype)
        y_data = torch.ones(o, device=device, dtype=dtype)
        control = torch.tensor([15., 33., 51., 69., 87.], device=device, dtype=dtype)

        def _test(row_major, incx, incy, lda_tail):
            if row_major:
                a_storage = torch.full((o, s + lda_tail), float('nan'), device=device, dtype=dtype)
            else:
                a_storage = torch.full((s, o + lda_tail), float('nan'), device=device, dtype=dtype).permute(1, 0)
            a = a_storage[:o, :s].copy_(a_data)

            x_storage = torch.full((s, incx), float('nan'), device=device, dtype=dtype)
            x = x_storage[:, 0].copy_(x_data)

            y_storage = torch.full((o, incy), float('nan'), device=device, dtype=dtype)
            y = y_storage[:, 0].copy_(y_data)

            self._test_addmm_addmv(torch.addmv, y, a, x)

        for row_major, incx, incy, lda_tail in product((False, True), (1, 2), (1, 2), (0, 1)):
            _test(row_major, incx, incy, lda_tail)

    @precisionOverride({torch.double: 1e-8, torch.float: 1e-4, torch.bfloat16: 0.6,
                        torch.half: 1e-1, torch.cfloat: 1e-4, torch.cdouble: 1e-8})
    @dtypesIfCUDA(*torch.testing.get_all_complex_dtypes(), *torch.testing.get_all_fp_dtypes(include_bfloat16=AMPERE_OR_ROCM))
    @dtypes(*torch.testing.get_all_complex_dtypes(), *torch.testing.get_all_fp_dtypes())
    @tf32_on_and_off(0.05)
    def test_addmm(self, device, dtype):
        M = torch.randn(10, 25, device=device).to(dtype)
        m1 = torch.randn(10, 50, device=device).to(dtype)
        m2 = torch.randn(50, 25, device=device).to(dtype)
        self._test_addmm_addmv(torch.addmm, M, m1, m2)

        # Test 0-strided
        M = torch.randn(10, 1, device=device).to(dtype).expand(10, 25)
        m1 = torch.randn(10, 1, device=device).to(dtype).expand(10, 50)
        m2 = torch.randn(50, 25, device=device).to(dtype)
        self._test_addmm_addmv(torch.addmm, M, m1, m2)

        # Test beta=0, M=nan
        M = torch.full((10, 25), math.nan, device=device).to(dtype)
        m1 = torch.randn(10, 50, device=device).to(dtype)
        m2 = torch.randn(50, 25, device=device).to(dtype)
        self._test_addmm_addmv(torch.addmm, M, m1, m2, beta=0)

        # Test transpose
        for t1, t2, t3, t4 in product([True, False], repeat=4):
            def maybe_transpose(cond, m):
                if not cond:
                    return m
                return m.t().clone(memory_format=torch.contiguous_format).t()

            M = maybe_transpose(t1, torch.randn(10, 25, device=device).to(dtype))
            m1 = maybe_transpose(t2, torch.randn(10, 50, device=device).to(dtype))
            m2 = maybe_transpose(t3, torch.randn(50, 25, device=device).to(dtype))
            self._test_addmm_addmv(torch.addmm, M, m1, m2, transpose_out=t4)

    @dtypes(torch.float, torch.double)
    @dtypesIfCUDA(*([torch.float, torch.double] +
                    ([] if TEST_WITH_ROCM else torch.testing.get_all_complex_dtypes())))
    @tf32_on_and_off(0.005)
    def test_addmm_sizes(self, device, dtype):
        for m in [0, 1, 25]:
            for n in [0, 1, 10]:
                for k in [0, 1, 8]:
                    M = torch.randn(n, m, device=device).to(dtype)
                    m1 = torch.randn(n, k, device=device).to(dtype)
                    m2 = torch.randn(k, m, device=device).to(dtype)
                    self._test_addmm_addmv(torch.addmm, M, m1, m2)

    @onlyCUDA
    def test_matmul_45724(self, device):
        # https://github.com/pytorch/pytorch/issues/45724
        a = torch.rand(65537, 22, 64, device=device, dtype=torch.half)
        b = torch.rand(65537, 64, 22, device=device, dtype=torch.half)
        c = torch.full((65537, 22, 22), math.nan, dtype=torch.half, device=device)
        cpu_result = torch.matmul(a.cpu().float(), b.cpu().float()).cuda().half()
        torch.matmul(a, b, out=c)
        self.assertEqual(c, cpu_result)

    def _test_dot_vdot_vs_numpy(self, device, dtype, torch_fn, np_fn):
        def compare_with_numpy_bin_op(torch_fn, np_fn, x, y):
            y_np = y.cpu().numpy()

            # `compare_with_numpy` takes care of moving `x` to correct device for calling np_fn.
            self.compare_with_numpy(lambda inp: torch_fn(inp, y), lambda inp: np_fn(inp, y_np), x)

        # Use this tensor for out variant tests.
        out = torch.randn((), dtype=dtype, device=device)

        def compare_out_variant(torch_fn, x, y):
            torch_fn(v1, v2, out=out)
            self.assertEqual(torch_fn(v1, v2), out)

        for _ in range(10):
            numel = random.randint(10, 1000)
            v1 = torch.randn(numel, dtype=dtype, device=device)
            v2 = torch.randn(numel, dtype=dtype, device=device)
            compare_with_numpy_bin_op(torch_fn, np_fn, v1, v2)
            compare_out_variant(torch_fn, v1, v2)

            # Test 0-strided
            v3 = torch.randn(1, dtype=dtype, device=device).expand(numel)
            compare_with_numpy_bin_op(torch_fn, np_fn, v1, v3)
            compare_out_variant(torch_fn, v1, v3)

            compare_with_numpy_bin_op(torch_fn, np_fn, v3, v1)
            compare_out_variant(torch_fn, v3, v1)

            # Test stride greater than 1
            v4 = torch.randn(numel, numel, dtype=dtype, device=device)[:, numel - 1]
            compare_with_numpy_bin_op(torch_fn, np_fn, v1, v4)
            compare_out_variant(torch_fn, v1, v4)

            compare_with_numpy_bin_op(torch_fn, np_fn, v4, v1)
            compare_out_variant(torch_fn, v4, v1)

    @precisionOverride({torch.cfloat: 1e-4, torch.float32: 5e-5})
    @dtypes(torch.float, torch.double, torch.cfloat, torch.cdouble)
    def test_dot_vs_numpy(self, device, dtype):
        self._test_dot_vdot_vs_numpy(device, dtype, torch.dot, np.dot)

    @precisionOverride({torch.cfloat: 1e-4, torch.float32: 5e-5})
    @dtypes(torch.float, torch.double, torch.cfloat, torch.cdouble)
    def test_vdot_vs_numpy(self, device, dtype):
        self._test_dot_vdot_vs_numpy(device, dtype, torch.vdot, np.vdot)

    def _test_dot_vdot_invalid_args(self, device, torch_fn, complex_dtypes=False):
        if complex_dtypes:
            x = torch.randn(1, dtype=torch.cfloat, device=device)
            y = torch.randn(3, dtype=torch.cdouble, device=device)
        else:
            x = torch.randn(1, dtype=torch.float, device=device)
            y = torch.randn(3, dtype=torch.double, device=device)

        with self.assertRaisesRegex(RuntimeError,
                                    'dot : expected both vectors to have same dtype'):
            torch_fn(x, y)

        with self.assertRaisesRegex(RuntimeError,
                                    '1D tensors expected'):
            torch_fn(x.reshape(1, 1), y)

        with self.assertRaisesRegex(RuntimeError,
                                    'inconsistent tensor size'):
            torch_fn(x.expand(9), y.to(x.dtype))

        if self.device_type != 'cpu':
            x_cpu = x.expand(3).cpu()

            with self.assertRaisesRegex(RuntimeError,
                                        'expected all tensors to be on the same device'):
                torch_fn(x_cpu, y.to(x.dtype))

    @onlyOnCPUAndCUDA
    def test_vdot_invalid_args(self, device):
        self._test_dot_vdot_invalid_args(device, torch.vdot)
        self._test_dot_vdot_invalid_args(device, torch.vdot, complex_dtypes=True)

    @onlyOnCPUAndCUDA
    def test_dot_invalid_args(self, device):
        self._test_dot_vdot_invalid_args(device, torch.dot)
        self._test_dot_vdot_invalid_args(device, torch.dot, complex_dtypes=True)

    @onlyCPU
    @skipCPUIfNoLapack
    def test_orgqr_errors(self, device):
        test_cases = [
            # input1 size, input2 size, error regex
            ((10,), (2,), r"'input' should be 2 dimensional"),
            ((10, 6), (20,), r"input.size\(1\) must be greater than or equal to input2.size\(0\)"),
            ((6, 10), (5,), r"input.size\(0\) must be greater than or equal to input.size\(1\)"),
            ((0, 0), (0,), r"'input' should not be empty"),
            ((2, 2), (2, 0,), r"'tau' should not be empty")
        ]
        for a_size, tau_size, error_regex in test_cases:
            a = torch.rand(*a_size, device=device)
            tau = torch.rand(*tau_size, device=device)
            with self.assertRaisesRegex(RuntimeError, error_regex):
                torch.orgqr(a, tau)

    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(torch.double, torch.cfloat, torch.cdouble)
    def test_lu(self, device, dtype):
        from torch.testing._internal.common_utils import random_matrix

        def run_test(device, pivot):
            def run_subtest(matrix_size, batches, device, pivot, singular=False, a=None):
                if isinstance(matrix_size, int):
                    rows = columns = matrix_size
                else:
                    rows, columns = matrix_size
                if a is None:
                    a = random_matrix(rows, columns, *batches, **dict(singular=singular, dtype=dtype)).to(device)
                a_LU_info, pivots_info, info_ = a.lu(pivot=pivot, get_infos=True)
                self.assertEqual(a_LU_info.size(), torch.Size(batches + (rows, columns)))
                self.assertEqual(pivots_info.size(), torch.Size(batches + (min(rows, columns),)))
                self.assertEqual(info_.size(), torch.Size(batches))
                # If a randomly generated input matrix is singular,
                # then info_ contains indices i such that U[i, i] ==
                # 0. This however conveys that the factorization was
                # successful albeit with a singular input. Therefore,
                # we require info.min() >= 0
                self.assertGreaterEqual(info_.min(), 0)
                a_LU, pivots = a.lu(pivot=pivot)
                self.assertEqual(a_LU, a_LU_info)
                self.assertEqual(pivots_info, pivots)


                if (self.device_type == 'cpu') or (not dtype.is_complex):
                    P, L, U = torch.lu_unpack(a_LU, pivots)

                    self.assertEqual(P.matmul(L.matmul(U)), a)
                else:
                    # TODO(@nikitaved): remove this once bmm_out is avaiable on CUDA for complex types

                    # squash batch dimensions for easier iteration
                    a = a.view(-1, a.size(-2), a.size(-1))
                    a_LU = a_LU.view(-1, a_LU.size(-2), a_LU.size(-1))
                    pivots = pivots.view(-1, pivots.size(-1))

                    P, L, U = torch.lu_unpack(a_LU, pivots)

                    for i in range(a.size(0)):
                        self.assertEqual(
                            P.select(0, i) @ L.select(0, i) @ U.select(0, i),
                            a.select(0, i)
                        )

                if self.device_type == 'cuda':
                    # lu without pivoting is implemented only for cuda device
                    a_LU_info_nopiv, nopiv, info_nopiv = a.lu(pivot=False, get_infos=True)
                    P_nopiv, L_nopiv, U_nopiv = torch.lu_unpack(a_LU_info_nopiv, nopiv)

                    if (self.device_type == 'cpu') or (not dtype.is_complex):
                        self.assertEqual(P_nopiv.matmul(L_nopiv.matmul(U_nopiv)), a)
                    else:
                        # TODO(@nikitaved): remove this once bmm_out is avaiable on CUDA for complex types
                        for i in range(a.size(0)):
                            self.assertEqual(
                                P_nopiv.select(0, i) @ L_nopiv.select(0, i) @ U_nopiv.select(0, i),
                                a.select(0, i)
                            )

                    k = min(rows, columns)
                    self.assertEqual(nopiv, torch.arange(1, 1 + k, device=device, dtype=torch.int32).expand(a.shape[:-2] + (k, )))
                    if not singular:
                        # It is not guaranteed that LU factorization
                        # without pivoting is able to determine if a
                        # matrix is singular while LU factorization
                        # with pivoting is. Therefore, we require the
                        # equality of info-s only for non-singular
                        # matrices.
                        # NOTE: infor_ is reshaped because info_nopiv might have
                        # squashed batch dimensions for complex types on CUDA,
                        # see the TODOs above.
                        self.assertEqual(info_.reshape(info_nopiv.shape), info_nopiv)

            for ms, batch in product([3, 5, 7, (4, 2), (3, 4)], [(), (2,), (3,), (3, 5)]):
                run_subtest(ms, batch, device, pivot)
                run_subtest(ms, batch, device, pivot, singular=True)

                # Reproducer of a magma bug, see https://bitbucket.org/icl/magma/issues/13/getrf_batched-kernel-produces-nans-on
                a = torch.ones(batch + (ms if isinstance(ms, tuple) else (ms, ms)), dtype=torch.double, device=device)
                run_subtest(ms, batch, device, pivot, singular=True, a=a)

            # Info should be positive for rank deficient matrices
            a = torch.ones(5, 3, 3, device=device)
            self.assertGreater(a.lu(pivot=pivot, get_infos=True)[2][0], 0)

        run_test(device, True)

        if self.device_type == 'cpu':
            # Error checking, no pivoting variant on CPU
            with self.assertRaisesRegex(RuntimeError, 'lu without pivoting is not implemented on the CPU'):
                torch.lu(torch.empty(1, 2, 2), pivot=False)
        else:
            run_test(device, False)

    @skipCPUIfNoLapack
    @skipCUDAIfNoMagma
    @dtypes(torch.double)
    def test_lu_unpack(self, device, dtype):
        def run_test(pivot):
            for shape in ((3, 3), (5, 3, 3), (7, 3, 5, 5), (7, 5, 3, 3, 3)):
                a = torch.randn(*shape, dtype=dtype, device=device)
                a_lu, p = torch.lu(a, pivot=pivot)
                p_ref, l_ref, u_ref = torch.lu_unpack(a_lu, p)
                self.assertEqual(p_ref.matmul(l_ref.matmul(u_ref)), a)

        run_test(True)

        if self.device_type == 'cuda':
            run_test(False)

    @dtypes(*(torch.testing.get_all_complex_dtypes() + torch.testing.get_all_fp_dtypes()))
    def test_blas_nan_out(self, device, dtype):
        # These functions should work correctly with NaN filled outputs,
        # but need special handling, see [NOTE: cpu_zero]
        b = 3
        n = 5
        m = 7
        p = 11

        # torch.mv
        nm = torch.randn((m, n), device=device).t()
        _m = torch.randn((), device=device).expand(m)
        _m_out = torch.full((m,), float('nan'), device=device)
        self.assertEqual(torch.mv(nm, _m), torch.mv(nm, _m, out=_m_out))
        self.assertEqual(0, torch.isnan(torch.mv(nm, _m)).sum())

        # torch.mm
        mp = torch.randn((p, m), device=device).t()
        np_out = torch.full((n, p), float('nan'), device=device)
        self.assertEqual(torch.mm(nm, mp), torch.mm(nm, mp, out=np_out))

        if dtype.is_complex and device.startswith('cuda'):
            return

        # torch.bmm
        bnm = torch.randn((b, m, n), device=device).transpose(1, 2)
        bmp = torch.randn((b, p, m), device=device).transpose(1, 2)
        bnp_out = torch.full((b, n, p), float('nan'), device=device)
        self.assertEqual(torch.bmm(bnm, bmp), torch.bmm(bnm, bmp, out=bnp_out))

    @onlyCPU  # not supported by CUBLAS
    def test_blas_mv_large_input(self, device):
        # This would previously fail if the allocated output had NaNs, see:
        # https://github.com/pytorch/pytorch/issues/31663 and [NOTE: cpu_zero]
        n = 3000
        m = 200

        nm = torch.randn((m, n), device=device).t()
        _m = torch.randn((), device=device).expand(m)
        _m_out = torch.full((m,), 0., device=device)

        self.assertEqual(torch.mv(nm, _m), torch.mv(nm, _m, out=_m_out))

    @tf32_on_and_off(0.005)
    def test_tensordot(self, device):
        a = torch.arange(60., device=device).reshape(3, 4, 5)
        b = torch.arange(24., device=device).reshape(4, 3, 2)
        c = torch.tensordot(a, b, dims=([1, 0], [0, 1])).cpu()
        cn = torch.from_numpy(np.tensordot(a.cpu().numpy(), b.cpu().numpy(),
                                           axes=([1, 0], [0, 1])))
        self.assertEqual(c, cn)

        cout = torch.zeros((5, 2))
        torch.tensordot(a, b, dims=([1, 0], [0, 1]), out=cout).cpu()
        self.assertEqual(c, cout)

        a = torch.randn(2, 3, 4, 5, device=device)
        b = torch.randn(4, 5, 6, 7, device=device)
        c = torch.tensordot(a, b, dims=2).cpu()
        cn = torch.from_numpy(np.tensordot(a.cpu().numpy(), b.cpu().numpy(),
                                           axes=2))

        with self.assertRaisesRegex(RuntimeError, "expects dims >= 0"):
            torch.tensordot(a, b, dims=-1)

        self.assertEqual(c, cn)
        c = torch.tensordot(a, b).cpu()
        cn = torch.from_numpy(np.tensordot(a.cpu().numpy(), b.cpu().numpy()))
        self.assertEqual(c, cn)

    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    def test_lapack_empty(self, device):
        # FIXME: these are just a selection of LAPACK functions -- we need a general strategy here.
        # The LAPACK functions themselves generally do NOT work with zero sized dimensions, although
        # numpy/sci often has a direct wrapper (e.g. lu_factor) and a wrapper that "does the right thing"
        # (e.g. lu).  We often name our functions identically to the lapack function, so it will take work
        # to name / migrate-to better wrappers.
        def fn(torchfn, *args):
            return torchfn(*tuple(torch.randn(shape, device=device) if isinstance(shape, tuple) else shape
                                  for shape in args))

        # inverse, pinverse
        self.assertEqual((0, 0), fn(torch.inverse, (0, 0)).shape)
        self.assertEqual((5, 0), fn(torch.pinverse, (0, 5)).shape)
        self.assertEqual((0, 5), fn(torch.pinverse, (5, 0)).shape)
        self.assertEqual((0, 0), fn(torch.pinverse, (0, 0)).shape)

        # det, logdet, slogdet
        self.assertEqual(torch.tensor(1., device=device), fn(torch.det, (0, 0)))
        self.assertEqual(torch.tensor(0., device=device), fn(torch.logdet, (0, 0)))
        self.assertEqual((torch.tensor(1., device=device), torch.tensor(0., device=device)),
                         fn(torch.slogdet, (0, 0)))

        # eig, symeig
        evalues, evectors = fn(torch.eig, (0, 0), True)
        self.assertEqual([(0, 2), (0, 0)], [evalues.shape, evectors.shape])
        evalues, evectors = fn(torch.symeig, (0, 0), True)
        self.assertEqual([(0,), (0, 0)], [evalues.shape, evectors.shape])

        # qr
        q, r = fn(torch.qr, (3, 0), True)
        self.assertEqual([(3, 0), (0, 0)], [q.shape, r.shape])
        q, r = fn(torch.qr, (0, 3), True)
        self.assertEqual([(0, 0), (0, 3)], [q.shape, r.shape])
        q, r = fn(torch.qr, (3, 0), False)
        self.assertEqual([(3, 3), (3, 0)], [q.shape, r.shape])

        # lstsq
        self.assertRaises(RuntimeError, lambda: torch.lstsq(torch.randn(0, 0), torch.randn(0, 0)))
        self.assertRaises(RuntimeError, lambda: torch.lstsq(torch.randn(0,), torch.randn(0, 0)))

    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    def test_geqrf(self, device):
        a = torch.randn(5, 5, device=device)
        b, c = torch.geqrf(a)
        b_placeholder, c_placeholder = torch.empty_like(b), torch.empty_like(c)
        torch.geqrf(a, out=(b_placeholder, c_placeholder))
        self.assertEqual(b, b_placeholder)
        self.assertEqual(c, c_placeholder)

    def triangular_solve_test_helper(self, A_dims, b_dims, upper, unitriangular,
                                     device, dtype):
        triangle_function = torch.triu if upper else torch.tril
        b = torch.randn(*b_dims, dtype=dtype, device=device)
        A = torch.randn(*A_dims, dtype=dtype, device=device)
        A_triangular = triangle_function(A)
        if unitriangular:
            A_triangular.diagonal(dim1=-2, dim2=-1).fill_(1.)
        return b, A_triangular

    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(torch.double)
    def test_triangular_solve(self, device, dtype):
        for (k, n), (upper, unitriangular, transpose) in product(zip([2, 3, 5], [3, 5, 7]),
                                                                 product([True, False], repeat=3)):
            b, A = self.triangular_solve_test_helper((n, n), (n, k), upper,
                                                     unitriangular, device, dtype)
            x = torch.triangular_solve(b, A, upper=upper, unitriangular=unitriangular, transpose=transpose)[0]
            if transpose:
                self.assertLessEqual(b.dist(A.t().mm(x)), 4e-12)
            else:
                self.assertLessEqual(b.dist(A.mm(x)), 4e-12)

    @skipCPUIfNoLapack
    @skipCUDAIfNoMagma
    @dtypes(torch.double)
    def test_triangular_solve_batched(self, device, dtype):
        def triangular_solve_batch_helper(A_dims, b_dims, upper, unitriangular, transpose):
            b, A = self.triangular_solve_test_helper(A_dims, b_dims, upper,
                                                     unitriangular, device, dtype)
            x_exp_list = []
            for i in range(b_dims[0]):
                x_exp_list.append(torch.triangular_solve(b[i], A[i], upper=upper,
                                                         unitriangular=unitriangular,
                                                         transpose=transpose)[0])
            x_exp = torch.stack(x_exp_list)  # Stacked output
            x_act = torch.triangular_solve(b, A, upper=upper,
                                           unitriangular=unitriangular,
                                           transpose=transpose)[0]  # Actual output
            self.assertEqual(x_act, x_exp)  # Equality check
            if transpose:
                self.assertLessEqual(b.dist(torch.matmul(A.transpose(-2, -1), x_act)), 3e-12)  # Correctness check
            else:
                self.assertLessEqual(b.dist(torch.matmul(A, x_act)), 3e-12)  # Correctness check

        for (upper, unitriangular, transpose), batchsize in product(product([True, False], repeat=3), [1, 3, 4]):
            triangular_solve_batch_helper((batchsize, 5, 5), (batchsize, 5, 10),
                                          upper, unitriangular, transpose)


    @slowTest
    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(torch.double)
    def test_triangular_solve_batched_many_batches(self, device, dtype):
        for upper, transpose, unitriangular in product([True, False], repeat=3):
            b, A = self.triangular_solve_test_helper((256, 256, 5, 5), (5, 1),
                                                     upper, unitriangular, device, dtype)
            x, _ = torch.triangular_solve(b, A,
                                          upper=upper, transpose=transpose, unitriangular=unitriangular)
            if transpose:
                A = A.transpose(-2, -1)
            self.assertEqual(torch.matmul(A, x), b.expand(A.shape[:-2] + (5, 1)))

            b, A = self.triangular_solve_test_helper((3, 3), (512, 512, 3, 1),
                                                     upper, unitriangular, device, dtype)
            x, _ = torch.triangular_solve(b, A, upper=upper, transpose=transpose,
                                          unitriangular=unitriangular)
            if transpose:
                A = A.transpose(-2, -1)
            self.assertEqual(torch.matmul(A, x), b)

    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @unittest.skipIf(not TEST_SCIPY, "SciPy not found")
    @dtypes(torch.double)
    def test_triangular_solve_batched_broadcasting(self, device, dtype):
        from scipy.linalg import solve_triangular as tri_solve

        def scipy_tri_solve_batched(A, B, upper, trans, diag):
            batch_dims_A, batch_dims_B = A.shape[:-2], B.shape[:-2]
            single_dim_A, single_dim_B = A.shape[-2:], B.shape[-2:]
            expand_dims = tuple(torch._C._infer_size(torch.Size(batch_dims_A),
                                                     torch.Size(batch_dims_B)))
            expand_A = np.broadcast_to(A, expand_dims + single_dim_A)
            expand_B = np.broadcast_to(B, expand_dims + single_dim_B)
            flat_A = expand_A.reshape((-1,) + single_dim_A)
            flat_B = expand_B.reshape((-1,) + single_dim_B)
            flat_X = np.vstack([tri_solve(a, b, lower=(not upper), trans=int(trans), unit_diagonal=diag)
                                for a, b in zip(flat_A, flat_B)])
            return flat_X.reshape(expand_B.shape)

        def run_test(A_dims, b_dims, device, upper, transpose, unitriangular):
            b, A = self.triangular_solve_test_helper(A_dims, b_dims, upper,
                                                     unitriangular, device, dtype)
            x_exp = torch.as_tensor(scipy_tri_solve_batched(A.cpu().numpy(), b.cpu().numpy(),
                                                            upper, transpose, unitriangular))
            x = torch.triangular_solve(b, A, upper=upper, transpose=transpose, unitriangular=unitriangular)[0]

            self.assertEqual(x, x_exp.to(device))

        for upper, transpose, unitriangular in product([True, False], repeat=3):
            # test against scipy.linalg.solve_triangular
            run_test((2, 1, 3, 4, 4), (2, 1, 3, 4, 6), device, upper, transpose, unitriangular)  # no broadcasting
            run_test((2, 1, 3, 4, 4), (4, 6), device, upper, transpose, unitriangular)  # broadcasting b
            run_test((4, 4), (2, 1, 3, 4, 2), device, upper, transpose, unitriangular)  # broadcasting A
            run_test((1, 3, 1, 4, 4), (2, 1, 3, 4, 5), device, upper, transpose, unitriangular)  # broadcasting A & b

    @onlyCPU
    @skipCPUIfNoLapack
    @dtypes(torch.double)
    def test_triangular_solve_singular(self, device, dtype):
        b = torch.rand(3, 1, device=device)
        A = torch.eye(3, 3, device=device)
        A[-1, -1] = 0  # Now A is singular
        err_str = r"triangular_solve_cpu: U\(3,3\) is zero, singular U\."
        with self.assertRaisesRegex(RuntimeError, err_str):
            torch.triangular_solve(b, A)

    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(torch.double)
    def test_lstsq(self, device, dtype):
        def _test_underdetermined(a, b, expectedNorm):
            # underdetermined systems are only supported on CPU
            if self.device_type != 'cpu':
                return

            m = a.size()[0]
            n = a.size()[1]
            assert(m <= n)

            a_copy = a.clone()
            b_copy = b.clone()
            res1 = torch.lstsq(b, a)[0]
            self.assertEqual(a, a_copy, atol=0, rtol=0)
            self.assertEqual(b, b_copy, atol=0, rtol=0)
            self.assertEqual((torch.mm(a, res1) - b).norm(), expectedNorm, atol=1e-8, rtol=0)

            ta = torch.tensor((), dtype=dtype, device=device)
            tb = torch.tensor((), dtype=dtype, device=device)
            res2 = torch.lstsq(b, a, out=(tb, ta))[0]
            self.assertEqual(a, a_copy, atol=0, rtol=0)
            self.assertEqual(b, b_copy, atol=0, rtol=0)
            self.assertEqual((torch.mm(a, res1) - b).norm(), expectedNorm, atol=1e-8, rtol=0)

            res3 = torch.lstsq(b, a, out=(b, a))[0]
            self.assertEqual((torch.mm(a_copy, b) - b_copy).norm(), expectedNorm, atol=1e-8, rtol=0)
            self.assertEqual(res1, tb, atol=0, rtol=0)
            self.assertEqual(res1, b, atol=0, rtol=0)
            self.assertEqual(res1, res2, atol=0, rtol=0)
            self.assertEqual(res1, res3, atol=0, rtol=0)

        def _test_overdetermined(a, b, expectedNorm):
            m = a.size()[0]
            n = a.size()[1]
            assert(m > n)

            def check_norm(a, b, expected_norm, gels_result):
                # Checks |ax - b| and the residual info from the result

                # The first n rows is the least square solution.
                # Rows n to m-1 contain residual information.
                x = gels_result[:n]
                resid_info = gels_result[n:]

                resid_norm = (torch.mm(a, x) - b).norm()
                self.assertEqual(resid_norm, expectedNorm, atol=1e-8, rtol=0)
                self.assertEqual(resid_info.norm(), resid_norm, atol=1e-8, rtol=0)

            a_copy = a.clone()
            b_copy = b.clone()
            res1 = torch.lstsq(b, a)[0]
            self.assertEqual(a, a_copy, atol=0, rtol=0)
            self.assertEqual(b, b_copy, atol=0, rtol=0)
            check_norm(a, b, expectedNorm, res1)

            ta = torch.tensor((), dtype=dtype, device=device)
            tb = torch.tensor((), dtype=dtype, device=device)
            res2 = torch.lstsq(b, a, out=(tb, ta))[0]
            self.assertEqual(a, a_copy, atol=0, rtol=0)
            self.assertEqual(b, b_copy, atol=0, rtol=0)
            check_norm(a, b, expectedNorm, res2)

            res3 = torch.lstsq(b, a, out=(b, a))[0]
            check_norm(a_copy, b_copy, expectedNorm, res3)

            self.assertEqual(res1, tb, atol=0, rtol=0)
            self.assertEqual(res1, b, atol=0, rtol=0)
            self.assertEqual(res1, res2, atol=0, rtol=0)
            self.assertEqual(res1, res3, atol=0, rtol=0)

        # basic test
        expectedNorm = 0
        a = torch.tensor(((1.44, -9.96, -7.55, 8.34),
                          (-7.84, -0.28, 3.24, 8.09),
                          (-4.39, -3.24, 6.27, 5.28),
                          (4.53, 3.83, -6.64, 2.06)), dtype=dtype, device=device).t()
        b = torch.tensor(((8.58, 8.26, 8.48, -5.28),
                          (9.35, -4.43, -0.70, -0.26)), dtype=dtype, device=device).t()
        _test_underdetermined(a, b, expectedNorm)

        # test overdetermined
        expectedNorm = 17.390200628863
        a = torch.tensor(((1.44, -9.96, -7.55, 8.34, 7.08, -5.45),
                          (-7.84, -0.28, 3.24, 8.09, 2.52, -5.70),
                          (-4.39, -3.24, 6.27, 5.28, 0.74, -1.19),
                          (4.53, 3.83, -6.64, 2.06, -2.47, 4.70)), dtype=dtype, device=device).t()
        b = torch.tensor(((8.58, 8.26, 8.48, -5.28, 5.72, 8.93),
                          (9.35, -4.43, -0.70, -0.26, -7.36, -2.52)), dtype=dtype, device=device).t()
        _test_overdetermined(a, b, expectedNorm)

        # test underdetermined
        expectedNorm = 0
        a = torch.tensor(((1.44, -9.96, -7.55),
                          (-7.84, -0.28, 3.24),
                          (-4.39, -3.24, 6.27),
                          (4.53, 3.83, -6.64)), dtype=dtype, device=device).t()
        b = torch.tensor(((8.58, 8.26, 8.48),
                          (9.35, -4.43, -0.70)), dtype=dtype, device=device).t()
        _test_underdetermined(a, b, expectedNorm)

        # test reuse
        expectedNorm = 0
        a = torch.tensor(((1.44, -9.96, -7.55, 8.34),
                          (-7.84, -0.28, 3.24, 8.09),
                          (-4.39, -3.24, 6.27, 5.28),
                          (4.53, 3.83, -6.64, 2.06)), dtype=dtype, device=device).t()
        b = torch.tensor(((8.58, 8.26, 8.48, -5.28),
                          (9.35, -4.43, -0.70, -0.26)), dtype=dtype, device=device).t()
        ta = torch.tensor((), dtype=dtype, device=device)
        tb = torch.tensor((), dtype=dtype, device=device)
        torch.lstsq(b, a, out=(tb, ta))
        self.assertEqual((torch.mm(a, tb) - b).norm(), expectedNorm, atol=1e-8, rtol=0)
        torch.lstsq(b, a, out=(tb, ta))
        self.assertEqual((torch.mm(a, tb) - b).norm(), expectedNorm, atol=1e-8, rtol=0)
        torch.lstsq(b, a, out=(tb, ta))
        self.assertEqual((torch.mm(a, tb) - b).norm(), expectedNorm, atol=1e-8, rtol=0)

    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @tf32_on_and_off(0.001)
    @dtypes(torch.float32, torch.float64, torch.complex64, torch.complex128)
    def test_qr(self, device, dtype):
        def run_test(tensor_dims, some):
            A = torch.randn(*tensor_dims, device=device)
            Q, R = torch.qr(A, some=some)

            # Check0: Q[-2:] = (m, n_columns), R[-2:] = (n_columns, n)
            m, n = tensor_dims[-2:]
            n_columns = m if (not some) and m > n else min(m, n)
            self.assertEqual(Q.size(-2), m)
            self.assertEqual(R.size(-1), n)
            self.assertEqual(Q.size(-1), n_columns)

            # Check1: A = QR
            self.assertEqual(A, torch.matmul(Q, R))

            # Check2: A = QR (with out)
            Q_out, R_out = torch.Tensor().to(device), torch.Tensor().to(device)
            torch.qr(A, some=some, out=(Q_out, R_out))
            self.assertEqual(A, torch.matmul(Q_out, R_out))

            # Check3: Q == Q_out, R == R_out
            self.assertEqual(Q, Q_out)
            self.assertEqual(R, R_out)

            # Check4: Q^{T}Q = I, triu(R) = R
            self.assertEqual(torch.matmul(Q.transpose(-2, -1), Q),
                             torch.eye(n_columns, device=device).expand(Q.shape[:-2] + (n_columns, n_columns)))
            self.assertEqual(R.triu(), R)

        tensor_dims_list = [(3, 5), (5, 5), (5, 3),  # Single matrix
                            (7, 3, 5), (7, 5, 5), (7, 5, 3),  # 3-dim Tensors
                            (7, 5, 3, 5), (7, 5, 5, 5), (7, 5, 5, 3)]  # 4-dim Tensors
        for tensor_dims, some in product(tensor_dims_list, [True, False]):
            run_test(tensor_dims, some)

    # Ensure that nuclear_norm's out variant gives the same result as the non-out
    @onlyOnCPUAndCUDA
    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(torch.float32, torch.float64)
    def test_nuclear_norm_out(self, device, dtype):
        test_cases = [
            # input size, dim
            ((25, 25), None),
            ((25, 25), (0, 1)),
            ((25, 25), (1, 0)),
            ((25, 25, 25), (2, 0)),
            ((25, 25, 25), (0, 1)),
        ]
        for keepdim in [False, True]:
            for input_size, dim in test_cases:
                msg = f'input_size: {input_size}, dim: {dim}, keepdim: {keepdim}'
                x = torch.randn(*input_size, device=device, dtype=dtype)
                result_out = torch.empty(0, device=device, dtype=dtype)
                if dim is None:
                    result = torch.nuclear_norm(x, keepdim=keepdim)
                    torch.nuclear_norm(x, keepdim=keepdim, out=result_out)
                else:
                    result = torch.nuclear_norm(x, keepdim=keepdim, dim=dim)
                    torch.nuclear_norm(x, keepdim=keepdim, dim=dim, out=result_out)
                self.assertEqual(result, result_out, msg=msg)

    @precisionOverride({torch.float32: 1e-5, torch.complex64: 1e-5})
    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypesIfCPU(torch.float32, torch.float64, torch.complex64, torch.complex128)
    @dtypesIfCUDA(torch.float32, torch.float64)
    def test_symeig(self, device, dtype):
        from torch.testing._internal.common_utils import random_hermitian_matrix

        def run_test(dims, eigenvectors, upper):
            x = random_hermitian_matrix(*dims, dtype=dtype, device=device)
            if dtype.is_complex:
                real_dtype = torch.float32 if dtype is torch.complex64 else torch.float64
            else:
                real_dtype = dtype
            oute = torch.empty(dims[1:] + dims[:1], dtype=real_dtype, device=device)
            outv = torch.empty(dims[1:] + dims[:1] * 2, dtype=dtype, device=device)
            torch.symeig(x, eigenvectors=eigenvectors, upper=upper, out=(oute, outv))

            if eigenvectors:
                x_recon = torch.matmul(torch.matmul(outv, torch.diag_embed(oute.to(dtype))), outv.transpose(-2, -1).conj())
                self.assertEqual(x, x_recon, atol=1e-8, rtol=0, msg='Incorrect reconstruction using V @ diag(e) @ V.T')
            else:
                eigvals, _ = torch.symeig(x, eigenvectors=True, upper=upper)
                self.assertEqual(eigvals, oute, msg='Eigenvalues mismatch')
                self.assertEqual(torch.empty(0, device=device, dtype=dtype), outv, msg='Eigenvector matrix not empty')

            rese, resv = x.symeig(eigenvectors=eigenvectors, upper=upper)
            self.assertEqual(rese, oute, msg="outputs of symeig and symeig with out don't match")
            self.assertEqual(resv, outv, msg="outputs of symeig and symeig with out don't match")

            # test non-contiguous
            x = random_hermitian_matrix(*dims, dtype=dtype, device=device)
            n_dim = len(dims) + 1
            # Reverse the batch dimensions and the matrix dimensions and then concat them
            x = x.permute(tuple(range(n_dim - 3, -1, -1)) + (n_dim - 1, n_dim - 2))
            assert not x.is_contiguous(), "x is intentionally non-contiguous"
            rese, resv = torch.symeig(x, eigenvectors=eigenvectors, upper=upper)
            if eigenvectors:
                x_recon = torch.matmul(torch.matmul(resv, torch.diag_embed(rese.to(dtype))), resv.transpose(-2, -1).conj())
                self.assertEqual(x, x_recon, atol=1e-8, rtol=0, msg='Incorrect reconstruction using V @ diag(e) @ V.T')
            else:
                eigvals, _ = torch.symeig(x, eigenvectors=True, upper=upper)
                self.assertEqual(eigvals, rese, msg='Eigenvalues mismatch')
                self.assertEqual(torch.empty(0, device=device, dtype=dtype), resv, msg='Eigenvector matrix not empty')

        batch_dims_set = [(), (3,), (3, 5), (5, 3, 5)]
        for batch_dims, eigenvectors, upper in product(batch_dims_set, (True, False), (True, False)):
            run_test((5,) + batch_dims, eigenvectors, upper)

    # TODO: once there is more support for complex dtypes on GPU, they shall be added to above test
    # particularly when RuntimeError: _th_bmm_out not supported on CUDAType for ComplexFloat is fixed
    @unittest.expectedFailure
    @onlyCUDA
    @skipCUDAIfNoMagma
    @dtypes(torch.complex64, torch.complex128)
    def test_symeig_complex_xfailed(self, device, dtype):
        from torch.testing._internal.common_utils import random_hermitian_matrix

        dims = (5, 3)
        x = random_hermitian_matrix(*dims, dtype=dtype, device=device)
        real_dtype = torch.float32 if dtype is torch.complex64 else torch.float64
        oute = torch.empty(dims[1:] + dims[:1], dtype=real_dtype, device=device)
        outv = torch.empty(dims[1:] + dims[:1] * 2, dtype=dtype, device=device)
        torch.symeig(x, eigenvectors=eigenvectors, upper=upper, out=(oute, outv))

        x_recon = torch.matmul(torch.matmul(outv, torch.diag_embed(oute.to(dtype))), outv.transpose(-2, -1).conj())
        self.assertEqual(x, x_recon, atol=1e-8, rtol=0)

    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(torch.double)
    def test_svd(self, device, dtype):
        def run_test(dims, some, compute_uv):
            x = torch.randn(*dims, dtype=dtype, device=device)
            outu = torch.tensor((), dtype=dtype, device=device)
            outs = torch.tensor((), dtype=dtype, device=device)
            outv = torch.tensor((), dtype=dtype, device=device)
            torch.svd(x, some=some, compute_uv=compute_uv, out=(outu, outs, outv))

            if compute_uv:
                if some:
                    x_recon = torch.matmul(outu, torch.matmul(outs.diag_embed(), outv.transpose(-2, -1)))
                    self.assertEqual(x, x_recon, atol=1e-8, rtol=0, msg='Incorrect reconstruction using U @ diag(S) @ V.T')
                else:
                    narrow_u = outu[..., :min(*dims[-2:])]
                    narrow_v = outv[..., :min(*dims[-2:])]
                    x_recon = torch.matmul(narrow_u, torch.matmul(outs.diag_embed(), narrow_v.transpose(-2, -1)))
                    self.assertEqual(x, x_recon, atol=1e-8, rtol=0, msg='Incorrect reconstruction using U @ diag(S) @ V.T')
            else:
                _, singvals, _ = torch.svd(x, compute_uv=True)
                self.assertEqual(singvals, outs, msg='Singular values mismatch')
                self.assertEqual(outu, torch.zeros_like(outu), msg='U not zero')
                self.assertEqual(outv, torch.zeros_like(outv), msg='V not zero')

            resu, ress, resv = torch.svd(x, some=some, compute_uv=compute_uv)
            self.assertEqual(resu, outu, msg='outputs of svd and svd with out differ')
            self.assertEqual(ress, outs, msg='outputs of svd and svd with out differ')
            self.assertEqual(resv, outv, msg='outputs of svd and svd with out differ')

            # test non-contiguous
            x = torch.randn(*dims, dtype=dtype, device=device)
            n_dim = len(dims)
            # Reverse the batch dimensions and the matrix dimensions and then concat them
            x = x.permute(tuple(range(n_dim - 3, -1, -1)) + (n_dim - 1, n_dim - 2))
            assert not x.is_contiguous(), "x is intentionally non-contiguous"
            resu, ress, resv = torch.svd(x, some=some, compute_uv=compute_uv)
            if compute_uv:
                if some:
                    x_recon = torch.matmul(resu, torch.matmul(ress.diag_embed(), resv.transpose(-2, -1)))
                    self.assertEqual(x, x_recon, atol=1e-8, rtol=0, msg='Incorrect reconstruction using U @ diag(S) @ V.T')
                else:
                    narrow_u = resu[..., :min(*dims[-2:])]
                    narrow_v = resv[..., :min(*dims[-2:])]
                    x_recon = torch.matmul(narrow_u, torch.matmul(ress.diag_embed(), narrow_v.transpose(-2, -1)))
                    self.assertEqual(x, x_recon, atol=1e-8, rtol=0, msg='Incorrect reconstruction using U @ diag(S) @ V.T')
            else:
                _, singvals, _ = torch.svd(x, compute_uv=True)
                self.assertEqual(singvals, ress, msg='Singular values mismatch')
                self.assertEqual(resu, torch.zeros_like(resu), msg='U not zero')
                self.assertEqual(resv, torch.zeros_like(resv), msg='V not zero')

        shapes = [(3, 3), (5, 3, 3), (7, 5, 3, 3),  # square matrices
                  (7, 3), (5, 7, 3), (7, 5, 7, 3),  # fat matrices
                  (3, 7), (5, 3, 7), (7, 5, 3, 7)]  # thin matrices
        for dims, some, compute_uv in product(shapes, [True, False], [True, False]):
            run_test(dims, some, compute_uv)

    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    def test_svd_no_singularvectors(self, device):
        for size in [(5, 5), (5, 20), (20, 5)]:
            a = torch.randn(*size, device=device)
            u, s_expect, v = torch.svd(a)
            u, s_actual, v = torch.svd(a, compute_uv=False)
            self.assertEqual(s_expect, s_actual, msg="Singular values don't match")

    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    def test_svd_lowrank(self, device):
        import torch
        from torch.testing._internal.common_utils import random_lowrank_matrix, random_sparse_matrix

        dtype = torch.double

        def run_subtest(actual_rank, matrix_size, batches, device, svd_lowrank, **options):
            density = options.pop('density', 1)
            if isinstance(matrix_size, int):
                rows = columns = matrix_size
            else:
                rows, columns = matrix_size
            if density == 1:
                a_input = random_lowrank_matrix(actual_rank, rows, columns, *batches, device=device, dtype=dtype)
                a = a_input
            else:
                assert batches == ()
                a_input = random_sparse_matrix(rows, columns, density, device=device, dtype=dtype)
                a = a_input.to_dense()

            q = min(*size)
            u, s, v = svd_lowrank(a_input, q=q, **options)

            # check if u, s, v is a SVD
            u, s, v = u[..., :q], s[..., :q], v[..., :q]
            A = u.matmul(s.diag_embed()).matmul(v.transpose(-2, -1))
            self.assertEqual(A, a)

            # check if svd_lowrank produces same singular values as torch.svd
            U, S, V = torch.svd(a)
            self.assertEqual(s.shape, S.shape)
            self.assertEqual(u.shape, U.shape)
            self.assertEqual(v.shape, V.shape)
            self.assertEqual(s, S)

            if density == 1:
                # actual_rank is known only for dense inputs
                #
                # check if pairs (u, U) and (v, V) span the same
                # subspaces, respectively
                u, s, v = u[..., :actual_rank], s[..., :actual_rank], v[..., :actual_rank]
                U, S, V = U[..., :actual_rank], S[..., :actual_rank], V[..., :actual_rank]
                self.assertEqual(u.transpose(-2, -1).matmul(U).det().abs(), torch.ones(batches, device=device, dtype=dtype))
                self.assertEqual(v.transpose(-2, -1).matmul(V).det().abs(), torch.ones(batches, device=device, dtype=dtype))

        all_batches = [(), (1,), (3,), (2, 3)]
        for actual_rank, size, all_batches in [
                (2, (17, 4), all_batches),
                (4, (17, 4), all_batches),
                (4, (17, 17), all_batches),
                (10, (100, 40), all_batches),
                (7, (1000, 1000), [()]),
        ]:
            # dense input
            for batches in all_batches:
                run_subtest(actual_rank, size, batches, device, torch.svd_lowrank)
                if size != size[::-1]:
                    run_subtest(actual_rank, size[::-1], batches, device, torch.svd_lowrank)

        # sparse input
        for size in [(17, 4), (4, 17), (17, 17), (100, 40), (40, 100), (1000, 1000)]:
            for density in [0.005, 0.1]:
                run_subtest(None, size, (), device, torch.svd_lowrank, density=density)

        # jitting support
        jitted = torch.jit.script(torch.svd_lowrank)
        actual_rank, size, batches = 2, (17, 4), ()
        run_subtest(actual_rank, size, batches, device, jitted)

    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(torch.double)
    def test_lu_solve_batched_non_contiguous(self, device, dtype):
        from numpy.linalg import solve
        from torch.testing._internal.common_utils import random_fullrank_matrix_distinct_singular_value

        A = random_fullrank_matrix_distinct_singular_value(2, 2, dtype=dtype, device='cpu')
        b = torch.randn(2, 2, 2, dtype=dtype, device='cpu')
        x_exp = torch.as_tensor(solve(A.permute(0, 2, 1).numpy(), b.permute(2, 1, 0).numpy())).to(device)
        A = A.to(device).permute(0, 2, 1)
        b = b.to(device).permute(2, 1, 0)
        assert not A.is_contiguous() and not b.is_contiguous(), "contiguous inputs"
        LU_data, LU_pivots = torch.lu(A)
        x = torch.lu_solve(b, LU_data, LU_pivots)
        self.assertEqual(x, x_exp)

    def lu_solve_test_helper(self, A_dims, b_dims, pivot, device, dtype):
        from torch.testing._internal.common_utils import random_fullrank_matrix_distinct_singular_value

        b = torch.randn(*b_dims, dtype=dtype, device=device)
        A = random_fullrank_matrix_distinct_singular_value(*A_dims, dtype=dtype, device=device)
        LU_data, LU_pivots, info = torch.lu(A, get_infos=True, pivot=pivot)
        self.assertEqual(info, torch.zeros_like(info))
        return b, A, LU_data, LU_pivots

    @skipCPUIfNoLapack
    @skipCUDAIfNoMagma
    @dtypes(torch.double)
    def test_lu_solve(self, device, dtype):
        def sub_test(pivot):
            for k, n in zip([2, 3, 5], [3, 5, 7]):
                b, A, LU_data, LU_pivots = self.lu_solve_test_helper((n,), (n, k), pivot, device, dtype)
                x = torch.lu_solve(b, LU_data, LU_pivots)
                self.assertLessEqual(b.dist(A.mm(x)), 1e-12)

        sub_test(True)
        if self.device_type == 'cuda':
            sub_test(False)

    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(torch.double)
    def test_lu_solve_batched(self, device, dtype):
        def sub_test(pivot):
            def lu_solve_batch_test_helper(A_dims, b_dims, pivot):
                b, A, LU_data, LU_pivots = self.lu_solve_test_helper(A_dims, b_dims, pivot, device, dtype)
                x_exp_list = []
                for i in range(b_dims[0]):
                    x_exp_list.append(torch.lu_solve(b[i], LU_data[i], LU_pivots[i]))
                x_exp = torch.stack(x_exp_list)  # Stacked output
                x_act = torch.lu_solve(b, LU_data, LU_pivots)  # Actual output
                self.assertEqual(x_exp, x_act)  # Equality check
                self.assertLessEqual(b.dist(torch.matmul(A, x_act)), 1e-12)  # Correctness check

            for batchsize in [1, 3, 4]:
                lu_solve_batch_test_helper((5, batchsize), (batchsize, 5, 10), pivot)

        # Tests tensors with 0 elements
        b = torch.randn(3, 0, 3, dtype=dtype, device=device)
        A = torch.randn(3, 0, 0, dtype=dtype, device=device)
        LU_data, LU_pivots = torch.lu(A)
        self.assertEqual(torch.empty_like(b), b.lu_solve(LU_data, LU_pivots))

        sub_test(True)
        if self.device_type == 'cuda':
            sub_test(False)

    @slowTest
    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(torch.double)
    def test_lu_solve_batched_many_batches(self, device, dtype):
        def run_test(A_dims, b_dims):
            b, A, LU_data, LU_pivots = self.lu_solve_test_helper(A_dims, b_dims, True, device, dtype)
            x = torch.lu_solve(b, LU_data, LU_pivots)
            b_ = torch.matmul(A, x)
            self.assertEqual(b_, b.expand_as(b_))

        run_test((5, 65536), (65536, 5, 10))
        run_test((5, 262144), (262144, 5, 10))

    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(torch.double)
    def test_lu_solve_batched_broadcasting(self, device, dtype):
        from numpy.linalg import solve
        from torch.testing._internal.common_utils import random_fullrank_matrix_distinct_singular_value

        def run_test(A_dims, b_dims, pivot=True):
            A_matrix_size = A_dims[-1]
            A_batch_dims = A_dims[:-2]
            A = random_fullrank_matrix_distinct_singular_value(A_matrix_size, *A_batch_dims, dtype=dtype)
            b = torch.randn(*b_dims, dtype=dtype)
            x_exp = torch.as_tensor(solve(A.numpy(), b.numpy())).to(dtype=dtype, device=device)
            A, b = A.to(device), b.to(device)
            LU_data, LU_pivots = torch.lu(A, pivot=pivot)
            x = torch.lu_solve(b, LU_data, LU_pivots)
            self.assertEqual(x, x_exp)

        # test against numpy.linalg.solve
        run_test((2, 1, 3, 4, 4), (2, 1, 3, 4, 6))  # no broadcasting
        run_test((2, 1, 3, 4, 4), (4, 6))  # broadcasting b
        run_test((4, 4), (2, 1, 3, 4, 2))  # broadcasting A
        run_test((1, 3, 1, 4, 4), (2, 1, 3, 4, 5))  # broadcasting A & b

    @dtypes(torch.double)
    def test_chain_matmul(self, device, dtype):
        def product(matrices):
            for mat in matrices[1:]:
                matrices[0] = matrices[0].mm(mat)
            return matrices[0]

        def run_test(p):
            matrices = []
            for (pi, pi_1) in zip(p[:-1], p[1:]):
                matrices.append(torch.randn(pi, pi_1, dtype=dtype, device=device))
            self.assertEqual(torch.chain_matmul(*matrices), product(matrices))

        run_test([10, 20, 30, 5])
        run_test([15, 5, 10, 20, 25])

        with self.assertRaisesRegex(RuntimeError, "chain_matmul: Expected one or more matrices"):
            torch.chain_matmul()

    @slowTest
    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(torch.double)
    def test_det_logdet_slogdet(self, device, dtype):
        def reference_slogdet(M):
            sdet, logabsdet = np.linalg.slogdet(M.detach().cpu().numpy())
            return M.new_tensor(sdet), M.new_tensor(logabsdet)

        def test_single_det(M, target, desc):
            target_sdet, target_logabsdet = target

            det = M.det()
            logdet = M.logdet()
            sdet, logabsdet = M.slogdet()

            # Test det
            self.assertEqual(det, target_sdet * target_logabsdet.exp(),
                             atol=1e-7, rtol=0, msg='{} (det)'.format(desc))

            # Test slogdet
            # Compare the overall value rather than individual parts because of
            # precision issues when det is near zero.
            self.assertEqual(sdet * logabsdet.exp(), target_sdet * target_logabsdet.exp(),
                             atol=1e-7, rtol=0, msg='{} (slogdet)'.format(desc))

            # Test logdet
            # Compare logdet against our own pytorch slogdet because they should
            # be consistent, while it may behave slightly differently with other
            # slogdet implementations when det is near zero due to precision
            # issues.
            if sdet.item() < 0:
                self.assertTrue(logdet.item() != logdet.item(), '{} (logdet negative case)'.format(desc))
            else:
                self.assertEqual(logdet.exp(), target_logabsdet.exp(),
                                 atol=1e-7, rtol=0, msg='{} (logdet non-negative case)'.format(desc))

        eye = torch.eye(5, dtype=dtype, device=device)
        test_single_det(eye, (torch.ones((), dtype=dtype, device=device), torch.zeros((), dtype=dtype, device=device)), 'identity')
        # Testing bug in #34061 (https://github.com/pytorch/pytorch/issues/34061)
        for n in range(250, 551, 100):
            mat = torch.randn(n, n, dtype=dtype, device=device)
            q, _ = torch.qr(mat)
            ref_det, ref_logabsdet = reference_slogdet(q)
            test_single_det(q, (ref_det, ref_logabsdet), 'orthogonal')

        def test(M):
            assert M.size(0) >= 5, 'this helper fn assumes M to be at least 5x5'
            M = M.to(device)

            ref_M_sdet, ref_M_logabsdet = reference_slogdet(M)

            test_single_det(M, (ref_M_sdet, ref_M_logabsdet), 'basic')
            if ref_M_logabsdet.exp().item() >= 1e-6:  # skip singular
                M_inv = M.inverse()
                test_single_det(M_inv, reference_slogdet(M_inv), 'inverse')

            test_single_det(M, (ref_M_sdet, ref_M_logabsdet), 'transpose')

            for x in [0, 2, 4]:
                for scale in [-2, -0.1, 0, 10]:
                    if scale > 0:
                        target = ref_M_sdet, ref_M_logabsdet + math.log(scale)
                    elif scale == 0:
                        target = torch.zeros_like(ref_M_sdet), torch.full_like(ref_M_logabsdet, -inf)
                    else:
                        target = ref_M_sdet.neg(), ref_M_logabsdet + math.log(-scale)

                    # dim 0
                    M_clone = M.clone()
                    M_clone[:, x] *= scale
                    test_single_det(M_clone, target, 'scale a row')
                    # dim 1
                    M_clone = M.clone()
                    M_clone[x, :] *= scale
                    test_single_det(M_clone, target, 'scale a column')

            for x1, x2 in [(0, 3), (4, 1), (3, 2)]:
                assert x1 != x2, 'x1 and x2 needs to be different for this test'
                target = torch.zeros_like(ref_M_sdet), torch.full_like(ref_M_logabsdet, -inf)
                # dim 0
                M_clone = M.clone()
                M_clone[:, x2] = M_clone[:, x1]
                test_single_det(M_clone, target, 'two rows are same')
                # dim 1
                M_clone = M.clone()
                M_clone[x2, :] = M_clone[x1, :]
                test_single_det(M_clone, target, 'two columns are same')

                for scale1, scale2 in [(0.3, -1), (0, 2), (10, 0.1)]:
                    det_scale = scale1 * scale2 * -1
                    if det_scale > 0:
                        target = ref_M_sdet, ref_M_logabsdet + math.log(det_scale)
                    elif det_scale == 0:
                        target = torch.zeros_like(ref_M_sdet), torch.full_like(ref_M_logabsdet, -inf)
                    else:
                        target = ref_M_sdet.neg(), ref_M_logabsdet + math.log(-det_scale)

                    # dim 0
                    M_clone = M.clone()
                    t = M_clone[:, x1] * scale1
                    M_clone[:, x1] += M_clone[:, x2] * scale2
                    M_clone[:, x2] = t
                    test_single_det(M_clone, target, 'exchanging rows')
                    # dim 1
                    M_clone = M.clone()
                    t = M_clone[x1, :] * scale1
                    M_clone[x1, :] += M_clone[x2, :] * scale2
                    M_clone[x2, :] = t
                    test_single_det(M_clone, target, 'exchanging columns')

        def get_random_mat_scale(n):
            # For matrices with values i.i.d. with 0 mean, unit variance, and
            # subexponential tail, we have:
            #   E[log det(A^2)] \approx log((n-1)!)
            #
            # Notice:
            #   log Var[det(A)] = log E[det(A^2)] >= E[log det(A^2)]
            #
            # So:
            #   stddev[det(A)] >= sqrt( (n-1)! )
            #
            # We use this as an intuitive guideline to scale random generated
            # matrices so our closeness tests can work more robustly:
            #   scale by sqrt( (n-1)! )^(-1/n) = ( (n-1)! )^(-1/(2n))
            #
            # source: https://arxiv.org/pdf/1112.0752.pdf

            # TODO: technically we need subexponential distn for this to hold,
            #       but we mostly use gaussian entries below. Consider switching
            #       to Chi-sq if this turns out not stable enough, since Chi-sq
            #       is easy enough to sample from.
            return math.factorial(n - 1) ** (-1.0 / (2 * n))

        for n in [5, 10, 25]:
            scale = get_random_mat_scale(n)
            test(torch.randn(n, n, dtype=dtype, device=device) * scale)
            r = torch.randn(n, n, dtype=dtype, device=device) * scale
            # symmetric psd
            test(r.mm(r.t()))
            # symmetric pd
            r = torch.randn(n, n, dtype=dtype, device=device) * scale
            test(r.mm(r.t()) + torch.eye(n, dtype=dtype, device=device) * 1e-6)
            # symmetric
            r = torch.randn(n, n, dtype=dtype, device=device) * scale
            for i in range(n):
                for j in range(i):
                    r[i, j] = r[j, i]
            test(r)
            # non-contiguous
            test((torch.randn(n, n, n + 1, dtype=dtype, device=device) * scale)[:, 2, 1:])
            # det = 0
            r = torch.randn(n, n, dtype=dtype, device=device) * scale
            u, s, v = r.svd()
            if reference_slogdet(u)[0] < 0:
                u = -u
            if reference_slogdet(v)[0] < 0:
                v = -v
            s[0] *= -1
            s[-1] = 0
            test(u.mm(s.diag()).mm(v))

        # Small values to test numerical stability. Note that we don't scale
        # this matrix.
        r = torch.randn(512, 512, dtype=dtype, device=device)
        u, s, v = r.svd()
        s.fill_(1. / (100 * s.numel()))
        test(u.mm(s.diag()).mm(v))

    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(torch.double)
    def test_det_logdet_slogdet_batched(self, device, dtype):
        from torch.testing._internal.common_utils import (random_symmetric_matrix, random_symmetric_psd_matrix,
                                                          random_symmetric_pd_matrix, random_square_matrix_of_rank)

        # mat_chars denotes matrix characteristics
        # possible values are: sym, sym_psd, sym_pd, sing, non_sym
        def run_test(matsize, batchdims, mat_chars):
            num_matrices = reduce(lambda x, y: x * y, batchdims, 1)
            list_of_matrices = []

            for idx in range(num_matrices):
                mat_type = idx % len(mat_chars)
                if mat_chars[mat_type] == 'sym':
                    list_of_matrices.append(random_symmetric_matrix(matsize, dtype=dtype, device=device))
                elif mat_chars[mat_type] == 'sym_psd':
                    list_of_matrices.append(random_symmetric_psd_matrix(matsize, dtype=dtype, device=device))
                elif mat_chars[mat_type] == 'sym_pd':
                    list_of_matrices.append(random_symmetric_pd_matrix(matsize, dtype=dtype, device=device))
                elif mat_chars[mat_type] == 'sing':
                    list_of_matrices.append(torch.ones(matsize, matsize, dtype=dtype, device=device))
                elif mat_chars[mat_type] == 'non_sing':
                    list_of_matrices.append(random_square_matrix_of_rank(matsize, matsize, dtype=dtype, device=device))
            full_tensor = torch.stack(list_of_matrices, dim=0).reshape(batchdims + (matsize, matsize))
            # Scaling adapted from `get_random_mat_scale` in _test_det_logdet_slogdet
            full_tensor *= (math.factorial(matsize - 1) ** (-1.0 / (2 * matsize)))

            for fn in [torch.det, torch.logdet, torch.slogdet]:
                expected_value = []
                actual_value = fn(full_tensor)
                for full_idx in product(*map(lambda x: list(range(x)), batchdims)):
                    expected_value.append(fn(full_tensor[full_idx]))

                if fn == torch.slogdet:
                    sign_value = torch.stack([tup[0] for tup in expected_value], dim=0).reshape(batchdims)
                    expected_value = torch.stack([tup[1] for tup in expected_value], dim=0).reshape(batchdims)
                    self.assertEqual(sign_value, actual_value[0])
                    self.assertEqual(expected_value, actual_value[1])
                else:
                    expected_value = torch.stack(expected_value, dim=0).reshape(batchdims)
                    self.assertEqual(actual_value, expected_value)

        for matsize, batchdims in product([3, 5], [(3,), (5, 3)]):
            run_test(matsize, batchdims, mat_chars=['sym_pd'])
            run_test(matsize, batchdims, mat_chars=['sing'])
            run_test(matsize, batchdims, mat_chars=['non_sing'])
            run_test(matsize, batchdims, mat_chars=['sym', 'sym_pd', 'sym_psd'])
            run_test(matsize, batchdims, mat_chars=['sing', 'non_sing'])

    def solve_test_helper(self, A_dims, b_dims, device, dtype):
        from torch.testing._internal.common_utils import random_fullrank_matrix_distinct_singular_value

        b = torch.randn(*b_dims, dtype=dtype, device=device)
        A = random_fullrank_matrix_distinct_singular_value(*A_dims, dtype=dtype, device=device)
        return b, A

    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(torch.double)
    def test_solve(self, device, dtype):
        for (k, n) in zip([2, 3, 5], [3, 5, 7]):
            b, A = self.solve_test_helper((n,), (n, k), device, dtype)
            x = torch.solve(b, A)[0]
            self.assertLessEqual(b.dist(A.mm(x)), 1e-12)

    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(torch.double)
    def test_solve_batched(self, device, dtype):
        def solve_batch_helper(A_dims, b_dims):
            b, A = self.solve_test_helper(A_dims, b_dims, device, dtype)
            x_exp_list = []
            for i in range(b_dims[0]):
                x_exp_list.append(torch.solve(b[i], A[i])[0])
            x_exp = torch.stack(x_exp_list)  # Stacked output
            x_act = torch.solve(b, A)[0]  # Actual output
            self.assertEqual(x_exp, x_act)  # Equality check
            self.assertLessEqual(b.dist(torch.matmul(A, x_act)), 1e-12)  # Correctness check

        for batchsize in [1, 3, 4]:
            solve_batch_helper((5, batchsize), (batchsize, 5, 10))

    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(torch.double)
    def test_solve_batched_non_contiguous(self, device, dtype):
        from numpy.linalg import solve
        from torch.testing._internal.common_utils import random_fullrank_matrix_distinct_singular_value
        A = random_fullrank_matrix_distinct_singular_value(2, 2, dtype=dtype,
                                                           device=device).permute(1, 0, 2)
        b = torch.randn(2, 2, 2, dtype=dtype, device=device).permute(2, 1, 0)
        x, _ = torch.solve(b, A)
        x_exp = torch.Tensor(solve(A.cpu().numpy(), b.cpu().numpy())).to(dtype=dtype, device=device)
        self.assertEqual(x, x_exp)

    @slowTest
    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(torch.double)
    def test_solve_batched_many_batches(self, device, dtype):
        b, A = self.solve_test_helper((5, 256, 256), (5, 1), device, dtype)
        x, _ = torch.solve(b, A)
        self.assertEqual(torch.matmul(A, x), b.expand(A.shape[:-2] + (5, 1)))

        b, A = self.solve_test_helper((3,), (512, 512, 3, 1), device, dtype)
        x, _ = torch.solve(b, A)
        self.assertEqual(torch.matmul(A, x), b)

    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(torch.double)
    def test_solve_batched_broadcasting(self, device, dtype):
        from numpy.linalg import solve

        def run_test(A_dims, b_dims):
            A_matrix_size = A_dims[-1]
            A_batch_dims = A_dims[:-2]
            b, A = self.solve_test_helper((A_matrix_size,) + A_batch_dims, b_dims, device, dtype)
            x, _ = torch.solve(b, A)
            x_exp = torch.Tensor(solve(A.cpu().numpy(), b.cpu().numpy())).to(dtype=dtype, device=device)
            self.assertEqual(x, x_exp)

        # test against numpy.linalg.solve
        run_test((2, 1, 3, 4, 4), (2, 1, 3, 4, 6))  # no broadcasting
        run_test((2, 1, 3, 4, 4), (4, 6))  # broadcasting b
        run_test((4, 4), (2, 1, 3, 4, 2))  # broadcasting A
        run_test((1, 3, 1, 4, 4), (2, 1, 3, 4, 5))  # broadcasting A & b

    def cholesky_solve_test_helper(self, A_dims, b_dims, upper, device, dtype):
        from torch.testing._internal.common_utils import random_symmetric_pd_matrix

        b = torch.randn(*b_dims, dtype=dtype, device=device)
        A = random_symmetric_pd_matrix(*A_dims, dtype=dtype, device=device)
        L = torch.cholesky(A, upper=upper)
        return b, A, L

    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(torch.double)
    def test_cholesky_solve(self, device, dtype):
        for (k, n), upper in product(zip([2, 3, 5], [3, 5, 7]), [True, False]):
            b, A, L = self.cholesky_solve_test_helper((n,), (n, k), upper, device, dtype)
            x = torch.cholesky_solve(b, L, upper=upper)
            self.assertLessEqual(b.dist(A.mm(x)), 1e-12)

    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(torch.double)
    def test_cholesky_solve_batched(self, device, dtype):
        def cholesky_solve_batch_helper(A_dims, b_dims, upper):
            b, A, L = self.cholesky_solve_test_helper(A_dims, b_dims, upper, device, dtype)
            x_exp_list = []
            for i in range(b_dims[0]):
                x_exp_list.append(torch.cholesky_solve(b[i], L[i], upper=upper))
            x_exp = torch.stack(x_exp_list)  # Stacked output
            x_act = torch.cholesky_solve(b, L, upper=upper)  # Actual output
            self.assertEqual(x_act, x_exp)  # Equality check
            self.assertLessEqual(b.dist(torch.matmul(A, x_act)), 2e-12)  # Correctness check

        for upper, batchsize in product([True, False], [1, 3, 4]):
            cholesky_solve_batch_helper((5, batchsize), (batchsize, 5, 10), upper)

    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(torch.double)
    def test_cholesky_solve_batched_non_contiguous(self, device, dtype):
        from numpy.linalg import solve
        from torch.testing._internal.common_utils import random_symmetric_pd_matrix

        for upper in [True, False]:
            A = random_symmetric_pd_matrix(2, 2, dtype=dtype, device='cpu')
            b = torch.randn(2, 2, 2, dtype=dtype, device='cpu')
            x_exp = torch.Tensor(solve(A.permute(0, 2, 1).numpy(), b.permute(2, 1, 0).numpy())).to(dtype=dtype, device=device)
            A = A.to(device).permute(0, 2, 1)
            b = b.to(device).permute(2, 1, 0)
            assert not A.is_contiguous() and not b.is_contiguous(), "contiguous inputs"
            L = torch.cholesky(A, upper)
            x = torch.cholesky_solve(b, L, upper=upper)
            self.assertEqual(x, x_exp)

    @slowTest
    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(torch.double)
    def test_cholesky_solve_batched_many_batches(self, device, dtype):
        for upper in [True, False]:
            b, A, L = self.cholesky_solve_test_helper((5, 256, 256), (5, 10), upper, device, dtype)
            x = torch.cholesky_solve(b, L, upper)
            self.assertEqual(torch.matmul(A, x), b.expand(A.shape[:-2] + (5, 10)))

            b, A, L = self.cholesky_solve_test_helper((5,), (512, 512, 5, 10), upper, device, dtype)
            x = torch.cholesky_solve(b, L, upper)
            self.assertEqual(torch.matmul(A, x), b)

    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(torch.double)
    def test_cholesky_solve_batched_broadcasting(self, device, dtype):
        from numpy.linalg import solve
        from torch.testing._internal.common_utils import random_symmetric_pd_matrix

        def run_test(A_dims, b_dims, upper):
            A_matrix_size = A_dims[-1]
            A_batch_dims = A_dims[:-2]
            A = random_symmetric_pd_matrix(A_matrix_size, *A_batch_dims,
                                           dtype=dtype, device='cpu')
            b = torch.randn(*b_dims, dtype=dtype, device='cpu')
            x_exp = torch.tensor(solve(A.numpy(), b.numpy()), dtype=dtype, device=device)
            A, b = A.to(dtype=dtype, device=device), b.to(dtype=dtype, device=device)
            L = torch.cholesky(A, upper)
            x = torch.cholesky_solve(b, L, upper=upper)
            self.assertEqual(x, x_exp)
            # issue gh-42695
            x = torch.cholesky_solve(b, L, upper=upper, out=x)
            self.assertEqual(x, x_exp)

        # test against numpy.linalg.solve
        for upper in [True, False]:
            run_test((2, 1, 3, 4, 4), (2, 1, 3, 4, 6), upper)  # no broadcasting
            run_test((2, 1, 3, 4, 4), (4, 6), upper)  # broadcasting b
            run_test((4, 4), (2, 1, 3, 4, 2), upper)  # broadcasting A
            run_test((1, 3, 1, 4, 4), (2, 1, 3, 4, 5), upper)  # broadcasting A & b

    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(torch.double)
    def test_cholesky_inverse(self, device, dtype):
        from torch.testing._internal.common_utils import random_symmetric_pd_matrix
        a = random_symmetric_pd_matrix(5, dtype=dtype, device=device)

        # compute inverse directly
        inv0 = torch.inverse(a)

        # default case
        chol = torch.cholesky(a)
        inv1 = torch.cholesky_inverse(chol, False)
        self.assertLessEqual(inv0.dist(inv1), 1e-12)

        # upper Triangular Test
        chol = torch.cholesky(a, True)
        inv1 = torch.cholesky_inverse(chol, True)
        self.assertLessEqual(inv0.dist(inv1), 1e-12)

        # lower Triangular Test
        chol = torch.cholesky(a, False)
        inv1 = torch.cholesky_inverse(chol, False)
        self.assertLessEqual(inv0.dist(inv1), 1e-12)

    @slowTest
    @skipCUDAIf(True, "See issue #26789.")
    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(torch.double)
    def test_cholesky_batched_many_batches(self, device, dtype):
        from torch.testing._internal.common_utils import random_symmetric_pd_matrix

        def cholesky_test_helper(n, batchsize, device, upper):
            A = random_symmetric_pd_matrix(n, batchsize, dtype=dtype, device=device)
            chol_fact = torch.cholesky(A, upper=upper)
            if upper:
                # Correctness check
                self.assertEqual(A, chol_fact.transpose(-2, -1).matmul(chol_fact))
                # Upper triangular check
                self.assertEqual(chol_fact, chol_fact.triu())
            else:
                # Correctness check
                self.assertEqual(A, chol_fact.matmul(chol_fact.transpose(-2, -1)))
                # Lower triangular check
                self.assertEqual(chol_fact, chol_fact.tril())

        for upper, batchsize in product([True, False], [262144, 524288]):
            cholesky_test_helper(2, batchsize, device, upper)

    @precisionOverride({torch.float32: 1e-4, torch.complex64: 1e-4})
    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(torch.float32, torch.float64, torch.complex64, torch.complex128)
    def test_cholesky_batched(self, device, dtype):
        from torch.testing._internal.common_utils import \
            (random_symmetric_pd_matrix,
             random_fullrank_matrix_distinct_singular_value)

        def cholesky_test_helper(n, batch_dims, upper):
            # This is a workaround while there is no support for complex random_symmetric_pd_matrix
            if dtype.is_complex:
                real_dtype = torch.float32 if dtype is torch.complex64 else torch.float64
                A_real = random_fullrank_matrix_distinct_singular_value(n, *batch_dims, dtype=real_dtype, device=device)
                A_imag = random_fullrank_matrix_distinct_singular_value(n, *batch_dims, dtype=real_dtype, device=device)
                A = A_real + 1j * A_imag
                # There is no support for complex batched matmul yet
                matmul_list = []
                for mat in A.contiguous().view(-1, n, n):
                    matmul_list.append(mat @ mat.t().conj())
                A = torch.stack(matmul_list).view(*batch_dims, n, n)
            else:
                A = random_symmetric_pd_matrix(n, *batch_dims, dtype=dtype, device=device)
            cholesky_exp = torch.stack([m.cholesky(upper=upper) for m in A.reshape(-1, n, n)])
            cholesky_exp = cholesky_exp.reshape_as(A)
            self.assertEqual(cholesky_exp, torch.cholesky(A, upper=upper))

        for upper, batchsize in product([True, False], [(3,), (3, 4), (2, 3, 4)]):
            cholesky_test_helper(3, batchsize, upper)

    @precisionOverride({torch.float32: 1e-4, torch.complex64: 1e-4})
    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(torch.float32, torch.float64, torch.complex64, torch.complex128)
    @tf32_on_and_off(0.01)
    def test_cholesky(self, device, dtype):
        from torch.testing._internal.common_utils import \
            (random_symmetric_pd_matrix,
             random_fullrank_matrix_distinct_singular_value)

        # This is a workaround while there is no support for complex random_symmetric_pd_matrix
        if dtype.is_complex:
            real_dtype = torch.float32 if dtype is torch.complex64 else torch.float64
            A_real = random_fullrank_matrix_distinct_singular_value(10, dtype=real_dtype, device=device)
            A_imag = random_fullrank_matrix_distinct_singular_value(10, dtype=real_dtype, device=device)
            A = A_real + 1j * A_imag
            A = A @ A.t().conj()
        else:
            A = random_symmetric_pd_matrix(10, dtype=dtype, device=device)

        # default Case
        C = torch.cholesky(A)
        B = torch.mm(C, C.t().conj())
        self.assertEqual(A, B, atol=1e-14, rtol=0)

        # test Upper Triangular
        U = torch.cholesky(A, True)
        B = torch.mm(U.t().conj(), U)
        self.assertEqual(A, B, atol=1e-14, rtol=0, msg='cholesky (upper) did not allow rebuilding the original matrix')

        # test Lower Triangular
        L = torch.cholesky(A, False)
        B = torch.mm(L, L.t().conj())
        self.assertEqual(A, B, atol=1e-14, rtol=0, msg='cholesky (lower) did not allow rebuilding the original matrix')

    # Tests torch.outer, and its alias, torch.ger, vs. NumPy
    @precisionOverride({torch.bfloat16: 1e-1})
    @dtypes(*(torch.testing.get_all_dtypes()))
    def test_outer(self, device, dtype):
        def run_test_case(a, b):
            if dtype == torch.bfloat16:
                a_np = a.to(torch.double).cpu().numpy()
                b_np = b.to(torch.double).cpu().numpy()
            else:
                a_np = a.cpu().numpy()
                b_np = b.cpu().numpy()
            expected = np.outer(a_np, b_np)

            self.assertEqual(torch.outer(a, b), expected)
            self.assertEqual(torch.Tensor.outer(a, b), expected)

            self.assertEqual(torch.ger(a, b), expected)
            self.assertEqual(torch.Tensor.ger(a, b), expected)

            # test out variant
            out = torch.empty(a.size(0), b.size(0), device=device, dtype=dtype)
            torch.outer(a, b, out=out)
            self.assertEqual(out, expected)

            out = torch.empty(a.size(0), b.size(0), device=device, dtype=dtype)
            torch.ger(a, b, out=out)
            self.assertEqual(out, expected)

        a = torch.randn(50).to(device=device, dtype=dtype)
        b = torch.randn(50).to(device=device, dtype=dtype)
        run_test_case(a, b)

        # test 0 strided tensor
        zero_strided = torch.randn(1).to(device=device, dtype=dtype).expand(50)
        run_test_case(zero_strided, b)
        run_test_case(a, zero_strided)

    @precisionOverride({torch.bfloat16: 1e-1})
    @dtypes(*(torch.testing.get_all_dtypes()))
    def test_addr(self, device, dtype):
        def run_test_case(m, a, b, beta=1, alpha=1):
            if dtype == torch.bfloat16:
                a_np = a.to(torch.double).cpu().numpy()
                b_np = b.to(torch.double).cpu().numpy()
                m_np = m.to(torch.double).cpu().numpy()
            else:
                a_np = a.cpu().numpy()
                b_np = b.cpu().numpy()
                m_np = m.cpu().numpy()

            if beta == 0:
                expected = alpha * np.outer(a_np, b_np)
            else:
                expected = beta * m_np + alpha * np.outer(a_np, b_np)

            self.assertEqual(torch.addr(m, a, b, beta=beta, alpha=alpha), expected)
            self.assertEqual(torch.Tensor.addr(m, a, b, beta=beta, alpha=alpha), expected)

            result_dtype = torch.addr(m, a, b, beta=beta, alpha=alpha).dtype
            out = torch.empty_like(m, dtype=result_dtype)
            torch.addr(m, a, b, beta=beta, alpha=alpha, out=out)
            self.assertEqual(out, expected)

        a = torch.randn(50).to(device=device, dtype=dtype)
        b = torch.randn(50).to(device=device, dtype=dtype)
        m = torch.randn(50, 50).to(device=device, dtype=dtype)

        # when beta is zero
        run_test_case(m, a, b, beta=0., alpha=2)

        # when beta is not zero
        run_test_case(m, a, b, beta=0.5, alpha=2)

        # test transpose
        m_transpose = torch.transpose(m, 0, 1)
        run_test_case(m_transpose, a, b, beta=0.5, alpha=2)

        # test 0 strided tensor
        zero_strided = torch.randn(1).to(device=device, dtype=dtype).expand(50)
        run_test_case(m, zero_strided, b, beta=0.5, alpha=2)

        # test scalar
        m_scalar = torch.tensor(1, device=device, dtype=dtype)
        run_test_case(m_scalar, a, b)

    @dtypes(*itertools.product(torch.testing.get_all_dtypes(),
                               torch.testing.get_all_dtypes()))
    def test_outer_type_promotion(self, device, dtypes):
        a = torch.randn(5).to(device=device, dtype=dtypes[0])
        b = torch.randn(5).to(device=device, dtype=dtypes[1])
        for op in (torch.outer, torch.Tensor.outer, torch.ger, torch.Tensor.ger):
            result = op(a, b)
            self.assertEqual(result.dtype, torch.result_type(a, b))

    @dtypes(*itertools.product(torch.testing.get_all_dtypes(),
                               torch.testing.get_all_dtypes()))
    def test_addr_type_promotion(self, device, dtypes):
        a = torch.randn(5).to(device=device, dtype=dtypes[0])
        b = torch.randn(5).to(device=device, dtype=dtypes[1])
        m = torch.randn(5, 5).to(device=device,
                                 dtype=torch.result_type(a, b))
        for op in (torch.addr, torch.Tensor.addr):
            # pass the integer 1 to the torch.result_type as both
            # the default values of alpha and beta are integers (alpha=1, beta=1)
            desired_dtype = torch.result_type(m, 1)
            result = op(m, a, b)
            self.assertEqual(result.dtype, desired_dtype)

            desired_dtype = torch.result_type(m, 2.)
            result = op(m, a, b, beta=0, alpha=2.)
            self.assertEqual(result.dtype, desired_dtype)

    # Tests migrated from test_torch.py
    # 1) test the shape of the result tensor when there is empty input tensor
    # 2) test the Runtime Exception when there is scalar input tensor
    def test_outer_ger_addr_legacy_tests(self, device):
        for size in ((0, 0), (0, 5), (5, 0)):
            a = torch.rand(size[0], device=device)
            b = torch.rand(size[1], device=device)

            self.assertEqual(torch.outer(a, b).shape, size)
            self.assertEqual(torch.ger(a, b).shape, size)

            m = torch.empty(size, device=device)
            self.assertEqual(torch.addr(m, a, b).shape, size)

        m = torch.randn(5, 6, device=device)
        a = torch.randn(5, device=device)
        b = torch.tensor(6, device=device)
        self.assertRaises(RuntimeError, lambda: torch.outer(a, b))
        self.assertRaises(RuntimeError, lambda: torch.outer(b, a))
        self.assertRaises(RuntimeError, lambda: torch.ger(a, b))
        self.assertRaises(RuntimeError, lambda: torch.ger(b, a))
        self.assertRaises(RuntimeError, lambda: torch.addr(m, a, b))
        self.assertRaises(RuntimeError, lambda: torch.addr(m, b, a))

    # Tests torch.det and its alias, torch.linalg.det, vs. NumPy
    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(torch.double)
    def test_det(self, device, dtype):
        tensors = (
            torch.randn((2, 2), device=device, dtype=dtype),
            torch.randn((129, 129), device=device, dtype=dtype),
            torch.randn((3, 52, 52), device=device, dtype=dtype),
            torch.randn((4, 2, 26, 26), device=device, dtype=dtype))


        ops = (torch.det, torch.Tensor.det,
               torch.linalg.det)
        for t in tensors:
            expected = np.linalg.det(t.cpu().numpy())
            for op in ops:
                actual = op(t)
                self.assertEqual(actual, expected)

        # NOTE: det requires a 2D+ tensor
        t = torch.randn(1, device=device, dtype=dtype)
        with self.assertRaises(RuntimeError):
            op(t)

    @dtypes(torch.float32, torch.float64, torch.complex64, torch.complex128)
    def test_kron(self, device, dtype):

        def run_test_case(a_shape, b_shape):
            a = torch.rand(a_shape, dtype=dtype, device=device)
            b = torch.rand(b_shape, dtype=dtype, device=device)

            expected = np.kron(a.cpu().numpy(), b.cpu().numpy())
            result = torch.kron(a, b)
            self.assertEqual(result, expected)

            # check the out= variant
            out = torch.empty_like(result)
            ans = torch.kron(a, b, out=out)
            self.assertEqual(ans, out)
            self.assertEqual(ans, result)

        shapes = [(4,), (2, 2), (1, 2, 3), (1, 2, 3, 3)]
        for a_shape, b_shape in itertools.product(shapes, reversed(shapes)):
            run_test_case(a_shape, b_shape)

    @dtypes(torch.float32, torch.float64, torch.complex64, torch.complex128)
    def test_kron_non_contiguous(self, device, dtype):

        def run_test_transposed(a_shape, b_shape):
            # check for transposed case
            a = torch.rand(a_shape, dtype=dtype, device=device).transpose(-2, -1)
            b = torch.rand(b_shape, dtype=dtype, device=device).transpose(-2, -1)
            self.assertFalse(a.is_contiguous())
            self.assertFalse(b.is_contiguous())

            expected = np.kron(a.cpu().numpy(), b.cpu().numpy())
            result = torch.kron(a, b)
            self.assertEqual(result, expected)

            # check the out= variant
            out = torch.empty(result.transpose(-2, -1).shape, dtype=dtype, device=device).transpose(-2, -1)
            self.assertFalse(out.is_contiguous())
            ans = torch.kron(a, b, out=out)
            self.assertEqual(ans, out)
            self.assertEqual(ans, result)

        def run_test_skipped_elements(a_shape, b_shape):
            # check for transposed case
            a = torch.rand(2 * a_shape[0], *a_shape[1:], dtype=dtype, device=device)[::2]
            b = torch.rand(2 * b_shape[0], *b_shape[1:], dtype=dtype, device=device)[::2]
            self.assertFalse(a.is_contiguous())
            self.assertFalse(b.is_contiguous())

            expected = np.kron(a.cpu().numpy(), b.cpu().numpy())
            result = torch.kron(a, b)
            self.assertEqual(result, expected)

            # check the out= variant
            out = torch.empty(2 * result.shape[0], *result.shape[1:], dtype=dtype, device=device)[::2]
            self.assertFalse(out.is_contiguous())
            ans = torch.kron(a, b, out=out)
            self.assertEqual(ans, out)
            self.assertEqual(ans, result)

        shapes = [(2, 2), (2, 2, 3), (2, 2, 3, 3)]
        for a_shape, b_shape in itertools.product(shapes, reversed(shapes)):
            # run_test_transposed(a_shape, b_shape)
            run_test_skipped_elements(a_shape, b_shape)

    @dtypes(torch.float32, torch.float64, torch.complex64, torch.complex128)
    def test_kron_empty(self, device, dtype):

        def run_test_case(empty_shape):
            a = torch.eye(3, dtype=dtype, device=device)
            b = torch.empty(empty_shape, dtype=dtype, device=device)
            result = torch.kron(a, b)
            expected = np.kron(a.cpu().numpy(), b.cpu().numpy())
            self.assertEqual(result, expected)

            # NumPy doesn't work if the first argument is empty
            result = torch.kron(b, a)
            self.assertEqual(result.shape, expected.shape)

        empty_shapes = [(0,), (2, 0), (1, 0, 3)]
        for empty_shape in empty_shapes:
            run_test_case(empty_shape)

    @dtypes(torch.float32, torch.float64, torch.complex64, torch.complex128)
    def test_kron_errors_and_warnings(self, device, dtype):
        # if non-empty out tensor with wrong shape is passed a warning is given
        a = torch.eye(3, dtype=dtype, device=device)
        b = torch.ones((2, 2), dtype=dtype, device=device)
        out = torch.empty_like(a)
        with warnings.catch_warnings(record=True) as w:
            # Trigger warning
            torch.kron(a, b, out=out)
            # Check warning occurs
            self.assertEqual(len(w), 1)
            self.assertTrue("An output with one or more elements was resized" in str(w[-1].message))

        # dtypes should match
        out = torch.empty_like(a).to(torch.int)
        with self.assertRaisesRegex(RuntimeError, "result dtype Int does not match self dtype"):
            torch.kron(a, b, out=out)

    # This test confirms that torch.linalg.norm's dtype argument works
    # as expected, according to the function's documentation
    @skipCUDAIfNoMagma
    def test_norm_dtype(self, device):
        def run_test_case(input_size, ord, keepdim, from_dtype, to_dtype, compare_dtype):
            msg = (
                f'input_size={input_size}, ord={ord}, keepdim={keepdim}, '
                f'from_dtype={from_dtype}, to_dtype={to_dtype}')
            input = torch.randn(*input_size, dtype=from_dtype, device=device)
            result = torch.linalg.norm(input, ord, keepdim=keepdim, dtype=from_dtype)
            self.assertEqual(result.dtype, from_dtype, msg=msg)
            result_converted = torch.linalg.norm(input, ord, keepdim=keepdim, dtype=to_dtype)
            self.assertEqual(result_converted.dtype, to_dtype, msg=msg)
            self.assertEqual(result.to(compare_dtype), result_converted.to(compare_dtype), msg=msg)

            result_out_converted = torch.empty_like(result_converted)
            torch.linalg.norm(input, ord, keepdim=keepdim, dtype=to_dtype, out=result_out_converted)
            self.assertEqual(result_out_converted.dtype, to_dtype, msg=msg)
            self.assertEqual(result_converted, result_out_converted, msg=msg)

        ord_vector = [0, 1, -1, 2, -2, 3, -3, 4.5, -4.5, inf, -inf, None]
        ord_matrix = ['fro', 'nuc', 1, -1, 2, -2, inf, -inf, None]
        S = 10
        test_cases = [
            ((S, ), ord_vector),
            ((S, S), ord_matrix),
        ]
        for keepdim in [True, False]:
            for input_size, ord_settings in test_cases:
                for ord in ord_settings:
                    # float to double
                    run_test_case(input_size, ord, keepdim, torch.float, torch.double, torch.float)
                    # double to float
                    run_test_case(input_size, ord, keepdim, torch.double, torch.double, torch.float)

        # Make sure that setting dtype != out.dtype raises an error
        dtype_pairs = [
            (torch.float, torch.double),
            (torch.double, torch.float),
        ]
        for keepdim in [True, False]:
            for input_size, ord_settings in test_cases:
                for ord in ord_settings:
                    for dtype, out_dtype in dtype_pairs:
                        input = torch.rand(*input_size)
                        result = torch.Tensor().to(out_dtype)
                        with self.assertRaisesRegex(RuntimeError, r'provided dtype must match dtype of result'):
                            torch.linalg.norm(input, ord=ord, keepdim=keepdim, dtype=dtype, out=result)

    # This test compares torch.linalg.norm and numpy.linalg.norm to ensure that
    # their vector norm results match
    @dtypes(torch.float, torch.double)
    def test_norm_vector(self, device, dtype):
        def run_test_case(input, p, dim, keepdim):
            result = torch.linalg.norm(input, ord, dim, keepdim)
            input_numpy = input.cpu().numpy()
            result_numpy = np.linalg.norm(input_numpy, ord, dim, keepdim)

            msg = f'input.size()={input.size()}, ord={ord}, dim={dim}, keepdim={keepdim}, dtype={dtype}'
            self.assertEqual(result, result_numpy, msg=msg)

            result_out = torch.empty_like(result)
            torch.linalg.norm(input, ord, dim, keepdim, out=result_out)
            self.assertEqual(result, result_out, msg=msg)

        ord_vector = [0, 1, -1, 2, -2, 3, -3, 4.5, -4.5, inf, -inf, None]
        S = 10
        test_cases = [
            # input size, p settings, dim
            ((S, ), ord_vector, None),
            ((S, ), ord_vector, 0),
            ((S, S, S), ord_vector, 0),
            ((S, S, S), ord_vector, 1),
            ((S, S, S), ord_vector, 2),
            ((S, S, S), ord_vector, -1),
            ((S, S, S), ord_vector, -2),
        ]
        L = 1_000_000
        if dtype == torch.double:
            test_cases.append(((L, ), ord_vector, None))
        for keepdim in [True, False]:
            for input_size, ord_settings, dim in test_cases:
                input = torch.randn(*input_size, dtype=dtype, device=device)
                for ord in ord_settings:
                    run_test_case(input, ord, dim, keepdim)

    # This test compares torch.linalg.norm and numpy.linalg.norm to ensure that
    # their matrix norm results match
    @skipCUDAIfNoMagma
    @dtypes(torch.float, torch.double)
    def test_norm_matrix(self, device, dtype):
        def run_test_case(input, p, dim, keepdim):
            result = torch.linalg.norm(input, ord, dim, keepdim)
            input_numpy = input.cpu().numpy()
            result_numpy = np.linalg.norm(input_numpy, ord, dim, keepdim)

            msg = f'input.size()={input.size()}, ord={ord}, dim={dim}, keepdim={keepdim}, dtype={dtype}'
            self.assertEqual(result, result_numpy, msg=msg)

            result_out = torch.empty_like(result)
            torch.linalg.norm(input, ord, dim, keepdim, out=result_out)
            self.assertEqual(result, result_out, msg=msg)

        ord_matrix = [1, -1, 2, -2, inf, -inf, 'nuc', 'fro', None]
        S = 10
        test_cases = [
            # input size, p settings, dim
            ((S, S), ord_matrix, None),
            ((S, S), ord_matrix, (0, 1)),
            ((S, S), ord_matrix, (1, 0)),
            ((S, S, S, S), ord_matrix, (2, 0)),
            ((S, S, S, S), ord_matrix, (-1, -2)),
            ((S, S, S, S), ord_matrix, (-1, -3)),
            ((S, S, S, S), ord_matrix, (-3, 2)),
        ]
        L = 1_000
        if dtype == torch.double:
            test_cases.append(((L, L), ord_matrix, None))
        for keepdim in [True, False]:
            for input_size, ord_settings, dim in test_cases:
                input = torch.randn(*input_size, dtype=dtype, device=device)
                for ord in ord_settings:
                    run_test_case(input, ord, dim, keepdim)

    # Test autograd and jit functionality for linalg functions.
    # TODO: Once support for linalg functions is added to method_tests in common_methods_invocations.py,
    #       the `test_cases` entries below should be moved there. These entries are in a similar format,
    #       so they should work with minimal changes.
    @dtypes(torch.float, torch.double)
    def test_autograd_and_jit(self, device, dtype):
        torch.manual_seed(0)
        S = 10
        NO_ARGS = None  # NOTE: refer to common_methods_invocations.py if you need this feature
        test_cases = [
            # NOTE: Not all the features from common_methods_invocations.py are functional here, since this
            #       is only a temporary solution.
            # (
            #   method name,
            #   input size/constructing fn,
            #   args (tuple represents shape of a tensor arg),
            #   test variant name (will be used at test name suffix),    // optional
            #   (should_check_autodiff[bool], nonfusible_nodes, fusible_nodes) for autodiff, // optional
            #   indices for possible dim arg,                            // optional
            #   fn mapping output to part that should be gradcheck'ed,   // optional
            #   kwargs                                                   // optional
            # )
            ('norm', (S,), (), 'default_1d'),
            ('norm', (S, S), (), 'default_2d'),
            ('norm', (S, S, S), (), 'default_3d'),
            ('norm', (S,), (inf,), 'vector_inf'),
            ('norm', (S,), (3.5,), 'vector_3_5'),
            ('norm', (S,), (0.5,), 'vector_0_5'),
            ('norm', (S,), (2,), 'vector_2'),
            ('norm', (S,), (1,), 'vector_1'),
            ('norm', (S,), (0,), 'vector_0'),
            ('norm', (S,), (-inf,), 'vector_neg_inf'),
            ('norm', (S,), (-3.5,), 'vector_neg_3_5'),
            ('norm', (S,), (-0.5,), 'vector_neg_0_5'),
            ('norm', (S,), (2,), 'vector_neg_2'),
            ('norm', (S,), (1,), 'vector_neg_1'),
            ('norm', (S, S), (inf,), 'matrix_inf'),
            ('norm', (S, S), (2,), 'matrix_2', (), NO_ARGS, [skipCPUIfNoLapack, skipCUDAIfNoMagma]),
            ('norm', (S, S), (1,), 'matrix_1'),
            ('norm', (S, S), (-inf,), 'matrix_neg_inf'),
            ('norm', (S, S), (-2,), 'matrix_neg_2', (), NO_ARGS, [skipCPUIfNoLapack, skipCUDAIfNoMagma]),
            ('norm', (S, S), (-1,), 'matrix_neg_1'),
            ('norm', (S, S), ('fro',), 'fro'),
            ('norm', (S, S), ('fro', [0, 1]), 'fro_dim'),
            ('norm', (S, S), ('nuc',), 'nuc', (), NO_ARGS, [skipCPUIfNoLapack, skipCUDAIfNoMagma]),
            ('norm', (S, S), ('nuc', [0, 1]), 'nuc_dim', (), NO_ARGS, [skipCPUIfNoLapack, skipCUDAIfNoMagma]),
        ]
        for test_case in test_cases:
            func_name = test_case[0]
            func = getattr(torch.linalg, func_name)
            input_size = test_case[1]
            args = list(test_case[2])
            test_case_name = test_case[3] if len(test_case) >= 4 else None
            mapping_funcs = list(test_case[6]) if len(test_case) >= 7 else None

            # Skip a test if a decorator tells us to
            if mapping_funcs is not None:
                def decorated_func(self, device, dtype):
                    pass
                for mapping_func in mapping_funcs:
                    decorated_func = mapping_func(decorated_func)
                try:
                    decorated_func(self, device, dtype)
                except unittest.SkipTest:
                    continue

            msg = f'function name: {func_name}, case name: {test_case_name}'

            # Test JIT
            input = torch.randn(*input_size, dtype=dtype, device=device)
            input_script = input.clone().detach()
            script_method, tensors = gen_script_fn_and_args("linalg.norm", "functional", input_script, *args)
            self.assertEqual(
                func(input, *args),
                script_method(input_script),
                msg=msg)

            # Test autograd
            # gradcheck is only designed to work with torch.double inputs
            if dtype == torch.double:
                input = torch.randn(*input_size, dtype=dtype, device=device, requires_grad=True)

                def run_func(input):
                    return func(input, *args)
                self.assertTrue(gradcheck(run_func, input), msg=msg)

    # This test calls torch.linalg.norm and numpy.linalg.norm with illegal arguments
    # to ensure that they both throw errors
    @dtypes(torch.float, torch.double)
    def test_norm_errors(self, device, dtype):
        def run_error_test_case(input, ord, dim, keepdim, error_type, error_regex):
            test_case_info = (
                f'test case input.size()={input.size()}, ord={ord}, dim={dim}, '
                f'keepdim={keepdim}, dtype={dtype}')

            with self.assertRaisesRegex(error_type, error_regex, msg=test_case_info):
                torch.linalg.norm(input, ord, dim, keepdim)

            input_numpy = input.cpu().numpy()

            msg = f'numpy does not raise error but pytorch does, for case "{test_case_info}"'
            with self.assertRaises(Exception, msg=test_case_info):
                np.linalg.norm(input_numpy, ord, dim, keepdim)

        S = 10
        error_test_cases = [
            # input size, p settings, dim, error type, error regex
            ((S, ), ['fro'], None, RuntimeError, r'order "fro" can only be used if either len\(dim\) == 2'),
            ((S, ), ['nuc'], None, RuntimeError, r'order "nuc" can only be used if either len\(dim\) == 2'),
            ((S, S), [3.5], None, RuntimeError, r'Order 3.5 not supported for matrix norm'),
            ((S, S), [0], None, RuntimeError, r'Order 0 not supported for matrix norm'),
            ((S, S), ['nuc'], 0, RuntimeError, r'order "nuc" can only be used if either len\(dim\) == 2'),
            ((S, S), ['fro'], 0, RuntimeError, r'order "fro" can only be used if either len\(dim\) == 2'),
            ((S, S), ['nuc'], (0, 0), RuntimeError, r'duplicate or invalid dimensions'),
            ((S, S), ['fro', 0], (0, 0), RuntimeError, r'Expected dims to be different'),
            ((S, S), ['fro', 'nuc', 0], (0, 4), IndexError, r'Dimension out of range'),
            ((S, ), [0], (4, ), IndexError, r'Dimension out of range'),
            ((S, ), [None], (0, 0), RuntimeError, r'Expected dims to be different, got this instead'),
            ((S, S, S), [1], (0, 1, 2), RuntimeError, r"'dim' must specify 1 or 2 dimensions"),
            ((S, S, S), [1], None, RuntimeError, r"'dim' must specify 1 or 2 dimensions"),
            ((S, S), ['garbage'], (0, 1), RuntimeError, r'Invalid norm order: garbage'),
        ]
        for keepdim in [True, False]:
            for input_size, ord_settings, dim, error_type, error_regex in error_test_cases:
                input = torch.randn(*input_size, dtype=dtype, device=device)
                for ord in ord_settings:
                    run_error_test_case(input, ord, dim, keepdim, error_type, error_regex)

    # Test complex number inputs for linalg.norm. Some cases are not supported yet, so
    # this test also verifies that those cases raise an error.
    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(torch.cfloat, torch.cdouble)
    def test_norm_complex(self, device, dtype):
        def gen_error_message(input_size, ord, keepdim, dim=None):
            return "complex norm failed for input size %s, ord=%s, keepdim=%s, dim=%s" % (
                input_size, ord, keepdim, dim)

        if self.device_type == 'cpu':
            supported_vector_ords = [0, 1, 3, inf, -1, -2, -3, -inf]
            supported_matrix_ords = ['nuc', 1, 2, inf, -1, -2, -inf]
            unsupported_vector_ords = [
                (2, r'norm with p=2 not supported for complex tensors'),
                (None, r'norm with p=2 not supported for complex tensors'),
            ]
            unsupported_matrix_ords = [
                ('fro', r'frobenius norm not supported for complex tensors'),
                (None, r'norm with p=2 not supported for complex tensors'),
            ]

        elif self.device_type == 'cuda':
            supported_vector_ords = [inf, -inf]
            supported_matrix_ords = [1, inf, -1, -inf]
            unsupported_vector_ords = [
                (0, r'norm_cuda" not implemented for \'Complex'),
                (1, r'norm_cuda" not implemented for \'Complex'),
                (2, r'norm with p=2 not supported for complex tensors'),
                (-1, r'norm_cuda" not implemented for \'Complex'),
                (-2, r'norm_cuda" not implemented for \'Complex'),
                (None, r'norm with p=2 not supported for complex tensors'),
            ]
            unsupported_matrix_ords = [
                (None, r'norm with p=2 not supported for complex tensors'),
                ('fro', r'frobenius norm not supported for complex tensors'),
            ]

        # Test supported ords
        for keepdim in [False, True]:
            # vector norm
            x = torch.randn(25, device=device, dtype=dtype)
            xn = x.cpu().numpy()
            for ord in supported_vector_ords:
                res = torch.linalg.norm(x, ord, keepdim=keepdim).cpu()
                expected = np.linalg.norm(xn, ord, keepdims=keepdim)
                msg = gen_error_message(x.size(), ord, keepdim)
                self.assertEqual(res.shape, expected.shape, msg=msg)
                self.assertEqual(res, expected, msg=msg)

            # matrix norm
            x = torch.randn(25, 25, device=device, dtype=dtype)
            xn = x.cpu().numpy()
            for ord in supported_matrix_ords:
                # TODO: Need to fix abort when nuclear norm is given cdouble input:
                #       "double free or corruption (!prev) Aborted (core dumped)"
                if ord == 'nuc' and dtype == torch.cdouble:
                    continue
                res = torch.linalg.norm(x, ord, keepdim=keepdim).cpu()
                expected = np.linalg.norm(xn, ord, keepdims=keepdim)
                msg = gen_error_message(x.size(), ord, keepdim)
                self.assertEqual(res.shape, expected.shape, msg=msg)
                self.assertEqual(res, expected, msg=msg)

        # Test unsupported ords
        # vector norm
        x = torch.randn(25, device=device, dtype=dtype)
        for ord, error_msg in unsupported_vector_ords:
            with self.assertRaisesRegex(RuntimeError, error_msg):
                torch.linalg.norm(x, ord)

        # matrix norm
        x = torch.randn(25, 25, device=device, dtype=dtype)
        for ord, error_msg in unsupported_matrix_ords:
            with self.assertRaisesRegex(RuntimeError, error_msg):
                torch.linalg.norm(x, ord)

    # Test that linal.norm gives the same result as numpy when inputs
    # contain extreme values (inf, -inf, nan)
    @unittest.skipIf(IS_WINDOWS, "Skipped on Windows!")
    @unittest.skipIf(IS_MACOS, "Skipped on MacOS!")
    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    def test_norm_extreme_values(self, device):
        vector_ords = [0, 1, 2, 3, inf, -1, -2, -3, -inf]
        matrix_ords = ['fro', 'nuc', 1, 2, inf, -1, -2, -inf]
        vectors = []
        matrices = []
        for pair in itertools.product([inf, -inf, 0.0, nan, 1.0], repeat=2):
            vectors.append(list(pair))
            matrices.append([[pair[0], pair[1]]])
            matrices.append([[pair[0]], [pair[1]]])
        for vector in vectors:
            x = torch.tensor(vector).to(device)
            x_n = x.cpu().numpy()
            for ord in vector_ords:
                msg = f'ord={ord}, vector={vector}'
                result = torch.linalg.norm(x, ord=ord)
                result_n = np.linalg.norm(x_n, ord=ord)
                self.assertEqual(result, result_n, msg=msg)

        # TODO: Remove this function once the broken cases are fixed
        def is_broken_matrix_norm_case(ord, x):
            if self.device_type == 'cuda':
                if x.size() == torch.Size([1, 2]):
                    if ord in ['nuc', 2, -2] and isnan(x[0][0]) and x[0][1] == 1:
                        # These cases are broken because of an issue with svd
                        # https://github.com/pytorch/pytorch/issues/43567
                        return True
            return False

        for matrix in matrices:
            x = torch.tensor(matrix).to(device)
            x_n = x.cpu().numpy()
            for ord in matrix_ords:
                msg = f'ord={ord}, matrix={matrix}'
                result = torch.linalg.norm(x, ord=ord)
                result_n = np.linalg.norm(x_n, ord=ord)

                if is_broken_matrix_norm_case(ord, x):
                    continue
                else:
                    self.assertEqual(result, result_n, msg=msg)

    # Test degenerate shape results match numpy for linalg.norm vector norms
    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @unittest.skipIf(TEST_WITH_ASAN, "Skipped on ASAN since it checks for undefined behavior.")
    @dtypes(torch.float, torch.double, torch.cfloat, torch.cdouble)
    def test_norm_vector_degenerate_shapes(self, device, dtype):
        def run_test_case(input, ord, dim, keepdim, should_error):
            msg = f'input.size()={input.size()}, ord={ord}, dim={dim}, keepdim={keepdim}, dtype={dtype}'
            input_numpy = input.cpu().numpy()
            if should_error:
                with self.assertRaises(ValueError):
                    np.linalg.norm(input_numpy, ord, dim, keepdim)
                with self.assertRaises(RuntimeError):
                    torch.linalg.norm(input, ord, dim, keepdim)
            else:
                if dtype in [torch.cfloat, torch.cdouble] and ord in [2, None]:
                    # TODO: Once these ord values have support for complex numbers,
                    #       remove this error test case
                    with self.assertRaises(RuntimeError):
                        torch.linalg.norm(input, ord, dim, keepdim)
                    return
                result_numpy = np.linalg.norm(input_numpy, ord, dim, keepdim)
                result = torch.linalg.norm(input, ord, dim, keepdim)
                self.assertEqual(result, result_numpy, msg=msg)

        ord_vector = [0, 0.5, 1, 2, 3, inf, -0.5, -1, -2, -3, -inf, None]
        S = 10
        test_cases = [
            # input size, p settings that cause error, dim
            ((0, ), [inf, -inf], None),
            ((0, S), [inf, -inf], 0),
            ((0, S), [], 1),
            ((S, 0), [], 0),
            ((S, 0), [inf, -inf], 1),
        ]
        for keepdim in [True, False]:
            for input_size, error_ords, dim in test_cases:
                input = torch.randn(*input_size, dtype=dtype, device=device)
                for ord in ord_vector:
                    run_test_case(input, ord, dim, keepdim, ord in error_ords)

    # Test degenerate shape results match numpy for linalg.norm matrix norms
    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(torch.float, torch.double, torch.cfloat, torch.cdouble)
    def test_norm_matrix_degenerate_shapes(self, device, dtype):
        def run_test_case(input, ord, dim, keepdim, should_error):
            if dtype in [torch.cfloat, torch.cdouble] and ord in ['fro', None]:
                # TODO: Once these ord values have support for complex numbers,
                #       remove this error test case
                with self.assertRaises(RuntimeError):
                    torch.linalg.norm(input, ord, dim, keepdim)
                return
            msg = f'input.size()={input.size()}, ord={ord}, dim={dim}, keepdim={keepdim}, dtype={dtype}'
            input_numpy = input.cpu().numpy()
            if should_error:
                with self.assertRaises(ValueError):
                    np.linalg.norm(input_numpy, ord, dim, keepdim)
                with self.assertRaises(RuntimeError):
                    torch.linalg.norm(input, ord, dim, keepdim)
            else:
                result_numpy = np.linalg.norm(input_numpy, ord, dim, keepdim)
                result = torch.linalg.norm(input, ord, dim, keepdim)
                self.assertEqual(result, result_numpy, msg=msg)

        ord_matrix = ['fro', 'nuc', 1, 2, inf, -1, -2, -inf, None]
        S = 10
        test_cases = [
            # input size, p settings that cause error, dim
            ((0, 0), [1, 2, inf, -1, -2, -inf], None),
            ((0, S), [2, inf, -2, -inf], None),
            ((S, 0), [1, 2, -1, -2], None),
            ((S, S, 0), [], (0, 1)),
            ((1, S, 0), [], (0, 1)),
            ((0, 0, S), [1, 2, inf, -1, -2, -inf], (0, 1)),
            ((0, 0, S), [1, 2, inf, -1, -2, -inf], (1, 0)),
        ]
        for keepdim in [True, False]:
            for input_size, error_ords, dim in test_cases:
                input = torch.randn(*input_size, dtype=dtype, device=device)
                for ord in ord_matrix:
                    run_test_case(input, ord, dim, keepdim, ord in error_ords)

    def test_norm_fastpaths(self, device):
        x = torch.randn(3, 5, device=device)

        # slow path
        result = torch.linalg.norm(x, 4.5, 1)
        expected = torch.pow(x.abs().pow(4.5).sum(1), 1.0 / 4.5)
        self.assertEqual(result, expected)

        # fast 0-norm
        result = torch.linalg.norm(x, 0, 1)
        expected = (x != 0).type_as(x).sum(1)
        self.assertEqual(result, expected)

        # fast 1-norm
        result = torch.linalg.norm(x, 1, 1)
        expected = x.abs().sum(1)
        self.assertEqual(result, expected)

        # fast 2-norm
        result = torch.linalg.norm(x, 2, 1)
        expected = torch.sqrt(x.pow(2).sum(1))
        self.assertEqual(result, expected)

        # fast 3-norm
        result = torch.linalg.norm(x, 3, 1)
        expected = torch.pow(x.pow(3).abs().sum(1), 1.0 / 3.0)
        self.assertEqual(result, expected)

    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    def test_norm_old(self, device):
        def gen_error_message(input_size, p, keepdim, dim=None):
            return "norm failed for input size %s, p=%s, keepdim=%s, dim=%s" % (
                input_size, p, keepdim, dim)

        for keepdim in [False, True]:
            # full reduction
            x = torch.randn(25, device=device)
            xn = x.cpu().numpy()
            for p in [0, 1, 2, 3, 4, inf, -inf, -1, -2, -3, 1.5]:
                res = x.norm(p, keepdim=keepdim).cpu()
                expected = np.linalg.norm(xn, p, keepdims=keepdim)
                self.assertEqual(res, expected, atol=1e-5, rtol=0, msg=gen_error_message(x.size(), p, keepdim))

            # one dimension
            x = torch.randn(25, 25, device=device)
            xn = x.cpu().numpy()
            for p in [0, 1, 2, 3, 4, inf, -inf, -1, -2, -3]:
                dim = 1
                res = x.norm(p, dim, keepdim=keepdim).cpu()
                expected = np.linalg.norm(xn, p, dim, keepdims=keepdim)
                msg = gen_error_message(x.size(), p, keepdim, dim)
                self.assertEqual(res.shape, expected.shape, msg=msg)
                self.assertEqual(res, expected, msg=msg)

            # matrix norm
            for p in ['fro', 'nuc']:
                res = x.norm(p, keepdim=keepdim).cpu()
                expected = np.linalg.norm(xn, p, keepdims=keepdim)
                msg = gen_error_message(x.size(), p, keepdim)
                self.assertEqual(res.shape, expected.shape, msg=msg)
                self.assertEqual(res, expected, msg=msg)

            # zero dimensions
            x = torch.randn((), device=device)
            xn = x.cpu().numpy()
            res = x.norm(keepdim=keepdim).cpu()
            expected = np.linalg.norm(xn, keepdims=keepdim)
            msg = gen_error_message(x.size(), None, keepdim)
            self.assertEqual(res.shape, expected.shape, msg=msg)
            self.assertEqual(res, expected, msg=msg)

            # larger tensor sanity check
            self.assertEqual(
                2 * torch.norm(torch.ones(10000), keepdim=keepdim),
                torch.norm(torch.ones(40000), keepdim=keepdim))

            # matrix norm with non-square >2-D tensors, all combinations of reduction dims
            x = torch.randn(5, 6, 7, 8, device=device)
            xn = x.cpu().numpy()
            for p in ['fro', 'nuc']:
                for dim in itertools.product(*[list(range(4))] * 2):
                    if dim[0] == dim[1]:
                        continue
                    res = x.norm(p=p, dim=dim, keepdim=keepdim).cpu()
                    expected = np.linalg.norm(xn, ord=p, axis=dim, keepdims=keepdim)
                    msg = gen_error_message(x.size(), p, keepdim, dim)
                    self.assertEqual(res.shape, expected.shape, msg=msg)
                    self.assertEqual(res, expected, msg=msg)

    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    def test_norm_complex_old(self, device):
        def gen_error_message(input_size, p, keepdim, dim=None):
            return "complex norm failed for input size %s, p=%s, keepdim=%s, dim=%s" % (
                input_size, p, keepdim, dim)

        if device == 'cpu':
            for keepdim in [False, True]:
                # vector norm
                x = torch.randn(25, device=device) + 1j * torch.randn(25, device=device)
                xn = x.cpu().numpy()
                for p in [0, 1, 3, inf, -1, -2, -3, -inf]:
                    res = x.norm(p, keepdim=keepdim).cpu()
                    expected = np.linalg.norm(xn, p, keepdims=keepdim)
                    msg = gen_error_message(x.size(), p, keepdim)
                    self.assertEqual(res.shape, expected.shape, msg=msg)
                    self.assertEqual(res, expected, msg=msg)

                # matrix norm
                x = torch.randn(25, 25, device=device) + 1j * torch.randn(25, 25, device=device)
                xn = x.cpu().numpy()
                for p in ['nuc']:
                    res = x.norm(p, keepdim=keepdim).cpu()
                    expected = np.linalg.norm(xn, p, keepdims=keepdim)
                    msg = gen_error_message(x.size(), p, keepdim)
                    self.assertEqual(res.shape, expected.shape, msg=msg)
                    self.assertEqual(res, expected, msg=msg)

            # TODO: remove error test and add functionality test above when 2-norm support is added
            with self.assertRaisesRegex(RuntimeError, r'norm with p=2 not supported for complex tensors'):
                x = torch.randn(2, device=device, dtype=torch.complex64).norm(p=2)

            # TODO: remove error test and add functionality test above when frobenius support is added
            with self.assertRaisesRegex(RuntimeError, r'frobenius norm not supported for complex tensors'):
                x = torch.randn(2, 2, device=device, dtype=torch.complex64).norm(p='fro')

        elif device == 'cuda':
            with self.assertRaisesRegex(RuntimeError, r'"norm_cuda" not implemented for \'ComplexFloat\''):
                (1j * torch.randn(25)).norm()

    # Ensure torch.norm with p='fro' and p=2 give the same results for mutually supported input combinations
    @dtypes(torch.float)
    def test_norm_fro_2_equivalence_old(self, device, dtype):
        input_sizes = [
            (0,),
            (10,),
            (0, 0),
            (4, 30),
            (0, 45),
            (100, 0),
            (45, 10, 23),
            (0, 23, 59),
            (23, 0, 37),
            (34, 58, 0),
            (0, 0, 348),
            (0, 3434, 0),
            (0, 0, 0),
            (5, 3, 8, 1, 3, 5)]

        for input_size in input_sizes:
            a = make_tensor(input_size, device, dtype, low=-9, high=9)

            # Try full reduction
            dim_settings = [None]

            # Try all possible 1-D reductions
            dim_settings += list(range(-a.dim(), a.dim()))

            def wrap_dim(dim, ndims):
                assert (dim < ndims) and (dim >= -ndims)
                if dim >= 0:
                    return dim
                else:
                    return dim + ndims

            # Try all possible 2-D reductions
            dim_settings += [
                (d0, d1) for d0, d1 in itertools.combinations(range(-a.dim(), a.dim()), 2)
                if wrap_dim(d0, a.dim()) != wrap_dim(d1, a.dim())]

            for dim in dim_settings:
                for keepdim in [True, False]:
                    a_norm_2 = torch.norm(a, p=2, dim=dim, keepdim=keepdim)
                    a_norm_fro = torch.norm(a, p='fro', dim=dim, keepdim=keepdim)
                    self.assertEqual(a_norm_fro, a_norm_2)

    @skipCUDAIfNoMagma
    def test_nuclear_norm_axes_small_brute_force_old(self, device):
        def check_single_nuclear_norm(x, axes):
            if self.device_type != 'cpu' and randrange(100) < 95:
                return  # too many cpu <==> device copies

            a = np.array(x.cpu(), copy=False)
            expected = np.linalg.norm(a, "nuc", axis=axes)

            ans = torch.norm(x, "nuc", dim=axes)
            self.assertTrue(ans.is_contiguous())
            self.assertEqual(ans.shape, expected.shape)
            self.assertEqual(ans.cpu(), expected, rtol=1e-02, atol=1e-03, equal_nan=True)

            out = torch.zeros(expected.shape, dtype=x.dtype, device=x.device)
            ans = torch.norm(x, "nuc", dim=axes, out=out)
            self.assertIs(ans, out)
            self.assertTrue(ans.is_contiguous())
            self.assertEqual(ans.shape, expected.shape)
            self.assertEqual(ans.cpu(), expected, rtol=1e-02, atol=1e-03, equal_nan=True)

        for n in range(1, 3):
            for m in range(1, 3):
                for axes in itertools.permutations([0, 1], 2):
                    # 2d, inner dimensions C
                    x = torch.randn(n, m, device=device)
                    check_single_nuclear_norm(x, axes)

                    # 2d, inner dimensions Fortran
                    x = torch.randn(m, n, device=device).transpose(-1, -2)
                    check_single_nuclear_norm(x, axes)

                    # 2d, inner dimensions non-contiguous
                    x = torch.randn(n, 2 * m, device=device)[:, ::2]
                    check_single_nuclear_norm(x, axes)

                    # 2d, all dimensions non-contiguous
                    x = torch.randn(7 * n, 2 * m, device=device)[::7, ::2]
                    check_single_nuclear_norm(x, axes)

                for o in range(1, 3):
                    for axes in itertools.permutations([0, 1, 2], 2):
                        # 3d, inner dimensions C
                        x = torch.randn(o, n, m, device=device)
                        check_single_nuclear_norm(x, axes)

                        # 3d, inner dimensions Fortran
                        x = torch.randn(o, m, n, device=device).transpose(-1, -2)
                        check_single_nuclear_norm(x, axes)

                        # 3d, inner dimensions non-contiguous
                        x = torch.randn(o, n, 2 * m, device=device)[:, :, ::2]
                        check_single_nuclear_norm(x, axes)

                        # 3d, all dimensions non-contiguous
                        x = torch.randn(7 * o, 5 * n, 2 * m, device=device)[::7, ::5, ::2]
                        check_single_nuclear_norm(x, axes)

                    for r in range(1, 3):
                        for axes in itertools.permutations([0, 1, 2, 3], 2):
                            # 4d, inner dimensions C
                            x = torch.randn(r, o, n, m, device=device)
                            check_single_nuclear_norm(x, axes)

                            # 4d, inner dimensions Fortran
                            x = torch.randn(r, o, n, m, device=device).transpose(-1, -2)
                            check_single_nuclear_norm(x, axes)

                            # 4d, inner dimensions non-contiguous
                            x = torch.randn(r, o, n, 2 * m, device=device)[:, :, :, ::2]
                            check_single_nuclear_norm(x, axes)

                            # 4d, all dimensions non-contiguous
                            x = torch.randn(7 * r, 5 * o, 11 * n, 2 * m, device=device)[::7, ::5, ::11, ::2]
                            check_single_nuclear_norm(x, axes)

    @skipCUDAIfNoMagma
    def test_nuclear_norm_exceptions_old(self, device):
        for lst in [], [1], [1, 2]:
            x = torch.tensor(lst, dtype=torch.double, device=device)
            for axes in (), (0,):
                self.assertRaises(RuntimeError, torch.norm, x, "nuc", axes)
            self.assertRaises(IndexError, torch.norm, x, "nuc", (0, 1))

        x = torch.tensor([[0, 1, 2], [3, 4, 5]], dtype=torch.double, device=device)
        self.assertRaisesRegex(RuntimeError, "duplicate or invalid", torch.norm, x, "nuc", (0, 0))
        self.assertRaisesRegex(IndexError, "Dimension out of range", torch.norm, x, "nuc", (0, 2))

    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(torch.float, torch.double, torch.cfloat, torch.cdouble)
    @dtypesIfCUDA(torch.float, torch.double)
    @precisionOverride({torch.float: 1e-4, torch.cfloat: 1e-4})
    def test_tensorsolve(self, device, dtype):
        def run_test(a_shape, dims):
            a = torch.randn(a_shape, dtype=dtype, device=device)
            b = torch.randn(a_shape[:2], dtype=dtype, device=device)
            result = torch.linalg.tensorsolve(a, b, dims=dims)
            expected = np.linalg.tensorsolve(a.cpu().numpy(), b.cpu().numpy(), axes=dims)
            self.assertEqual(result, expected)

            # check the out= variant
            out = torch.empty_like(result)
            ans = torch.linalg.tensorsolve(a, b, dims=dims, out=out)
            self.assertEqual(ans, out)
            self.assertEqual(ans, result)

        a_shapes = [(2, 3, 6), (3, 4, 4, 3)]
        dims = [None, (0, 2)]
        for a_shape, d in itertools.product(a_shapes, dims):
            run_test(a_shape, d)

    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(torch.float, torch.double, torch.cfloat, torch.cdouble)
    @dtypesIfCUDA(torch.float, torch.double)
    def test_tensorsolve_empty(self, device, dtype):
        # Check for empty inputs. NumPy does not work for these cases.
        a = torch.empty(0, 0, 1, 2, 3, 0, dtype=dtype, device=device)
        b = torch.empty(a.shape[:2], dtype=dtype, device=device)
        x = torch.linalg.tensorsolve(a, b)
        self.assertEqual(torch.tensordot(a, x, dims=len(x.shape)), b)

    # TODO: once "solve_cuda" supports complex dtypes, they shall be added to above tests
    @unittest.expectedFailure
    @onlyCUDA
    @skipCUDAIfNoMagma
    @dtypes(torch.cfloat, torch.cdouble)
    def test_tensorsolve_xfailed(self, device, dtype):
        a_shape = (2, 3, 6)
        a = torch.randn(a_shape, dtype=dtype, device=device)
        b = torch.randn(a_shape[:2], dtype=dtype, device=device)
        result = torch.linalg.tensorsolve(a, b)
        expected = np.linalg.tensorsolve(a.cpu().numpy(), b.cpu().numpy())
        self.assertEqual(result, expected)

    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(torch.float, torch.double, torch.cfloat, torch.cdouble)
    @dtypesIfCUDA(torch.float, torch.double)
    @precisionOverride({torch.float: 1e-4, torch.cfloat: 1e-4})
    def test_tensorsolve_non_contiguous(self, device, dtype):
        def run_test_permuted(a_shape, dims):
            # check for permuted / transposed inputs
            a = torch.randn(a_shape, dtype=dtype, device=device)
            a = a.movedim((0, 2), (-2, -1))
            self.assertFalse(a.is_contiguous())
            b = torch.randn(a.shape[:2], dtype=dtype, device=device)
            b = b.t()
            self.assertFalse(b.is_contiguous())
            result = torch.linalg.tensorsolve(a, b, dims=dims)
            expected = np.linalg.tensorsolve(a.cpu().numpy(), b.cpu().numpy(), axes=dims)
            self.assertEqual(result, expected)

        def run_test_skipped_elements(a_shape, dims):
            # check for inputs with skipped elements
            a = torch.randn(a_shape, dtype=dtype, device=device)
            a = a[::2]
            self.assertFalse(a.is_contiguous())
            b = torch.randn(a_shape[:2], dtype=dtype, device=device)
            b = b[::2]
            self.assertFalse(b.is_contiguous())
            result = torch.linalg.tensorsolve(a, b, dims=dims)
            expected = np.linalg.tensorsolve(a.cpu().numpy(), b.cpu().numpy(), axes=dims)
            self.assertEqual(result, expected)

            # check non-contiguous out
            out = torch.empty(2 * result.shape[0], *result.shape[1:], dtype=dtype, device=device)[::2]
            self.assertFalse(out.is_contiguous())
            ans = torch.linalg.tensorsolve(a, b, dims=dims, out=out)
            self.assertEqual(ans, out)
            self.assertEqual(ans, result)

        a_shapes = [(2, 3, 6), (3, 4, 4, 3)]
        dims = [None, (0, 2)]
        for a_shape, d in itertools.product(a_shapes, dims):
            run_test_permuted(a_shape, d)

        a_shapes = [(4, 3, 6), (6, 4, 4, 3)]
        dims = [None, (0, 2)]
        for a_shape, d in itertools.product(a_shapes, dims):
            run_test_skipped_elements(a_shape, d)

    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(torch.float32)
    def test_tensorsolve_errors_and_warnings(self, device, dtype):
        # tensorsolve expects the input that can be reshaped to a square matrix
        a = torch.eye(2 * 3 * 4).reshape((2 * 3, 4, 2, 3, 4))
        b = torch.randn(8, 4)
        self.assertTrue(np.prod(a.shape[2:]) != np.prod(b.shape))
        with self.assertRaisesRegex(RuntimeError, r'Expected self to satisfy the requirement'):
            torch.linalg.tensorsolve(a, b)

        # if non-empty out tensor with wrong shape is passed a warning is given
        out = torch.empty_like(a)
        b = torch.randn(6, 4)
        with warnings.catch_warnings(record=True) as w:
            # Trigger warning
            torch.linalg.tensorsolve(a, b, out=out)
            # Check warning occurs
            self.assertEqual(len(w), 1)
            self.assertTrue("An output with one or more elements was resized" in str(w[-1].message))

        # dtypes should match
        out = torch.empty_like(a).to(torch.int)
        with self.assertRaisesRegex(RuntimeError, "result dtype Int does not match self dtype"):
            torch.linalg.tensorsolve(a, b, out=out)

instantiate_device_type_tests(TestLinalg, globals())

if __name__ == '__main__':
    run_tests()
