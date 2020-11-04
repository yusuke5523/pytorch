import math
from itertools import product, chain
from numbers import Number
import random

import unittest

import torch

from torch.testing._internal.common_utils import \
    (TestCase, run_tests, torch_to_numpy_dtype_dict, suppress_warnings,
     TEST_NUMPY, make_tensor)
from torch.testing._internal.common_methods_invocations import \
    (unary_ufuncs)
from torch.testing._internal.common_device_type import \
    (instantiate_device_type_tests, ops, dtypes, onlyOnCPUAndCUDA)
from torch.testing import \
    (floating_types_and, integral_types, all_types_and_complex_and)

if TEST_NUMPY:
    import numpy as np

# Tests for unary "universal functions (ufuncs)" that accept a single
# tensor and have common properties like:
#   - they are elementwise functions
#   - the input shape is the output shape
#   - they typically have method and inplace variants
#   - they typically support the out kwarg
#   - they typically have NumPy or SciPy references

# See NumPy's universal function documentation
# (https://numpy.org/doc/1.18/reference/ufuncs.html) for more details
# about the concept of ufuncs.

# Functions tested here:
#

# Interesting values and extremal values for different dtypes
_unsigned_int_vals = (0, 1, 55, 127)
_int_vals = (0, -1, 1, -55, 55, -127, 127, -128, 128)
_large_int_vals = (-1113, 1113, -10701, 10701)
_float_vals = (0.,
               -.001, .001,
               -.25, .25,
               -1., 1.,
               -math.pi / 2, math.pi / 2,
               -math.pi + .00001, math.pi - .00001,
               -math.pi, math.pi,
               -math.pi - .00001, math.pi + .00001)
_large_float_vals = (-501, 501,
                     -1001.2, 1001.2,
                     -13437.7, 13437.7,
                     -4988429.2, 4988429.2,
                     -1e20, 1e20)
_float_extremals = (float('inf'), float('-inf'), float('nan'))


# Returns an iterable of contiguous tensors with the same storage on the requested
#   device and with the requested dtype.
#
# This function is intended to test the non-vectorized and vectorized code
#   paths of unary functions, as well as their handling of odd tensor
#   sizes (like zero-dim tensors and tensors with zero elements).
#
# The iterable will include an empty tensor, tensors with no elements,
#   zero dim (scalar) tensors, small 1D tensors, a medium 1D tensor, and
#   a large 2D tensor.
#
# These tensors will include interesting values. If include_large_values
#   is true they will include larger values (>500), too, and if
#   include_extremal_values is true they will include extremal values
#   like -inf, inf, and nan.
#
# The randomly generated values can be constracted by the domain
#   argument.
def generate_numeric_tensors(device, dtype, *,
                             domain=(None, None),
                             include_large_values=True,
                             include_extremal_values=True):
    medium_length = 812
    large_size = (1029, 917)
    offset = 63

    assert large_size[1] > (medium_length + offset)
    assert medium_length % 4 == 0

    # Special-cases bool
    if dtype is torch.bool:
        tensors = (torch.empty(0, device=device, dtype=torch.bool),
                   torch.tensor(True, device=device),
                   torch.tensor(False, device=device),
                   torch.tensor((True, False), device=device),
                   make_tensor((medium_length,), device=device, dtype=dtype, low=None, high=None),
                   make_tensor(large_size, device=device, dtype=dtype, low=None, high=None))
        return tensors

    # Acquires dtype-specific vals
    if dtype.is_floating_point or dtype.is_complex:
        large_vals = _large_float_vals if include_large_values else tuple()
        extremals = _float_extremals if include_extremal_values else tuple()
        vals = _float_vals + large_vals + extremals

        # Converts float -> complex vals if dtype is complex
        if dtype.is_complex:
            vals = tuple(complex(x, y) for x, y in product(vals, vals))
    elif dtype is torch.uint8:
        vals = _unsigned_int_vals
    else:  # dtypes is a signed integer type
        assert dtype in (torch.int8, torch.int16, torch.int32, torch.int64)
        large_vals = _large_int_vals if include_large_values else tuple()
        vals = _int_vals + large_vals

    assert len(vals) < medium_length

    # Constructs the large tensor containing vals
    large_tensor = make_tensor(large_size, device=device, dtype=dtype, low=domain[0], high=domain[1])

    # Inserts the vals at an odd place
    large_tensor[57][offset:offset + len(vals)] = torch.tensor(vals, device=device, dtype=dtype)

    # Takes a medium sized copy of the large tensor containing vals
    medium_tensor = large_tensor[57][offset:offset + medium_length]

    # Constructs small tensors (4 elements)
    small_tensors = (t for t in torch.split(medium_tensor, 4))

    # Constructs scalar tensors
    scalar_tensors = (t.squeeze() for t in torch.split(medium_tensor, 1))

    # Tensors with no elements
    empty_sizes = ((0,), (0, 3, 3), (1, 0, 5), (6, 0, 0, 0), (3, 0, 1, 0))
    empty_tensors = (torch.empty(size, device=device, dtype=dtype) for size in empty_sizes)

    return chain(empty_tensors, scalar_tensors, small_tensors, (medium_tensor,), (large_tensor,))

# TODO: port test_unary_out_op_mem_overlap
# TODO: add out= tests (different devices, dtypes, mismatched sizes,
#                       correct sizes, 0 size, broadcasted out)
# TODO: add test for inplace variants erroring on broadcasted inputs
class TestUnaryUfuncs(TestCase):
    exact_dtype = True

    def unary_check_input_output_mem_overlap(self, data, sz, op,
                                             expected_failure=False):

        def _test(op, output, input):
            output_exp = torch.empty_like(output)
            op(input, out=output_exp)
            self.assertEqual(op(input, out=output), output_exp, msg=op.__name__)

        # output is identical to input:
        _test(op, output=data[0:sz], input=data[0:sz])
        # output and input are independent:
        _test(op, output=data[0:sz], input=data[sz:2 * sz])
        # output partially overlaps with input:
        if not expected_failure:
            with self.assertRaisesRegex(RuntimeError, 'unsupported operation'):
                _test(op, data[0:sz], data[1:sz + 1])
        else:
            with self.assertRaises(AssertionError):
                with self.assertRaisesRegex(RuntimeError, 'unsupported operation'):
                    _test(op, data[0:sz], data[1:sz + 1])

    def check_internal_mem_overlap(self, inplace_op, num_inputs,
                                   dtype, device,
                                   expected_failure=False):
        if isinstance(inplace_op, str):
            inplace_op = getattr(torch.Tensor, inplace_op)
        input = torch.randn(1, dtype=dtype, device=device).expand(3, 3)
        inputs = [input] + [torch.randn_like(input)
                            for i in range(num_inputs - 1)]
        if not expected_failure:
            with self.assertRaisesRegex(RuntimeError, 'single memory location'):
                inplace_op(*inputs)
        else:
            with self.assertRaises(AssertionError):
                with self.assertRaisesRegex(RuntimeError, 'single memory location'):
                    inplace_op(*inputs)

    # TODO: run on non-native device types
    @onlyOnCPUAndCUDA
    @dtypes(torch.double)
    def test_unary_out_op_mem_overlap(self, device, dtype):
        sz = 3
        doubles = torch.randn(2 * sz, dtype=dtype, device=device)
        positives = torch.randint(1, 100, (2 * sz,), device=device).double()
        ints = torch.randint(-100, 100, (2 * sz,), device=device)
        unary_mem_overlap_cases = [
            ("abs", doubles, True, True, 'cpu'),
            ("abs", doubles, True, True, 'cuda'),
            ("acos", doubles, True, True, 'cpu'),
            ("acos", doubles, True, True, 'cuda'),
            ("asin", doubles, True, True, 'cpu'),
            ("asin", doubles, True, True, 'cuda'),
            ("atan", doubles, True, True, 'cpu'),
            ("atan", doubles, True, True, 'cuda'),
            ("acosh", doubles, True, True, 'cpu'),
            ("acosh", doubles, True, True, 'cuda'),
            ("asinh", doubles, True, True, 'cpu'),
            ("asinh", doubles, True, True, 'cuda'),
            ("atanh", doubles, True, True, 'cpu'),
            ("atanh", doubles, True, True, 'cuda'),
            ("bitwise_not", ints, True, True, 'cpu'),
            ("bitwise_not", ints, True, True, 'cuda'),
            ("ceil", doubles, True, True, 'cpu'),
            ("ceil", doubles, True, True, 'cuda'),
            ("cos", doubles, True, True, 'cpu'),
            ("cos", doubles, True, True, 'cuda'),
            ("cosh", doubles, True, True, 'cpu'),
            ("cosh", doubles, True, True, 'cuda'),
            ("digamma", doubles, True, True, 'cpu'),
            ("erf", doubles, True, True, 'cpu'),
            ("erf", doubles, True, True, 'cuda'),
            ("erfc", doubles, True, True, 'cpu'),
            ("erfc", doubles, True, True, 'cuda'),
            ("erfinv", doubles, True, True, 'cpu'),
            ("erfinv", doubles, True, True, 'cuda'),
            ("exp", doubles, True, True, 'cpu'),
            ("exp", doubles, True, True, 'cuda'),
            ("exp2", doubles, True, True, 'cpu'),
            ("exp2", doubles, True, True, 'cuda'),
            ("expm1", doubles, True, True, 'cpu'),
            ("expm1", doubles, True, True, 'cuda'),
            ("floor", doubles, True, True, 'cpu'),
            ("floor", doubles, True, True, 'cuda'),
            ("frac", doubles, True, True, 'cpu'),
            ("frac", doubles, True, True, 'cuda'),
            ("i0", doubles, True, True, 'cpu'),
            ("i0", doubles, True, True, 'cuda'),
            ("log", positives, True, True, 'cpu'),
            ("log", positives, True, True, 'cuda'),
            ("log10", positives, True, True, 'cpu'),
            ("log10", positives, True, True, 'cuda'),
            ("log1p", positives, True, True, 'cpu'),
            ("log1p", positives, True, True, 'cuda'),
            ("log2", positives, True, True, 'cpu'),
            ("log2", positives, True, True, 'cuda'),
            ("neg", doubles, True, True, 'cpu'),
            ("neg", doubles, True, True, 'cuda'),
            ("reciprocal", doubles, True, True, 'cpu'),
            ("reciprocal", doubles, True, True, 'cuda'),
            ("round", doubles, True, True, 'cpu'),
            ("round", doubles, True, True, 'cuda'),
            ("rsqrt", positives, True, True, 'cpu'),
            ("rsqrt", positives, True, True, 'cuda'),
            ("sin", doubles, True, True, 'cpu'),
            ("sin", doubles, True, True, 'cuda'),
            ("sinh", doubles, True, True, 'cpu'),
            ("sinh", doubles, False, True, 'cuda'),
            ("sigmoid", doubles, True, True, 'cpu'),
            ("sigmoid", doubles, True, True, 'cuda'),
            ("logit", doubles, True, True, 'cpu'),
            ("logit", doubles, True, True, 'cuda'),
            ("sqrt", doubles, True, True, 'cpu'),
            ("sqrt", doubles, False, True, 'cuda'),
            ("tan", doubles, True, True, 'cpu'),
            ("tan", doubles, True, True, 'cuda'),
            ("tanh", doubles, True, True, 'cpu'),
            ("tanh", doubles, True, True, 'cuda'),
            ("trunc", doubles, True, True, 'cpu'),
            ("trunc", doubles, True, True, 'cuda')
        ]

        for (fn, inputs, has_input_output_mem_overlap_check,
             has_internal_mem_overlap_check, dev) in unary_mem_overlap_cases:
            if dev != device:
                continue
            out_fn = getattr(torch, fn)
            in_fn = getattr(torch.Tensor, fn + '_')

            self.unary_check_input_output_mem_overlap(inputs, sz, out_fn,
                                                      expected_failure=not has_input_output_mem_overlap_check)

            self.check_internal_mem_overlap(in_fn, 1, dtype, dev,
                                            expected_failure=not has_internal_mem_overlap_check)

    # This function tests that a nan value is returned for input values not in domain
    @dtypes(torch.float32, torch.float64)
    def test_acosh_domain_float(self, device, dtype):
        # Domain of acosh is [1, inf), for values outside the domain - output is mapped
        # to NaN, except for input value `inf` - output is mapped to `inf`
        sample = torch.tensor([float('-inf'), 1.00, -1.23, -0.06, 0.98, float('inf')],
                              device=device, dtype=dtype)
        nan_mask = torch.tensor([True, False, True, True, True, False], device=device)
        inf_mask = torch.tensor([False, False, False, False, False, True], device=device)
        self.assertEqual(torch.isnan(torch.acosh(sample)), nan_mask)
        self.assertEqual(torch.isnan(sample.acosh()), nan_mask)
        self.assertEqual(torch.isinf(torch.acosh(sample)), inf_mask)
        self.assertEqual(torch.isinf(sample.acosh()), inf_mask)

    # This function tests that a nan value is returned for input values not in domain
    @dtypes(torch.float32, torch.float64)
    def test_atanh_domain_float(self, device, dtype):
        # Domain of atanh is (-1, 1), for edge values (-1 and 1) - output is mapped
        # to inf and for other values outside this range - output is mapped to NaN
        sample = torch.tensor([float('-inf'), -1.00, 1.00, -1.23, 1.06, float('inf')],
                              device=device, dtype=dtype)
        nan_mask = torch.tensor([True, False, False, True, True, True], device=device)
        inf_mask = torch.tensor([False, True, True, False, False, False], device=device)
        # For values not in domain (except -1.0 and 1.0), atanh should return nan
        self.assertEqual(torch.isnan(torch.atanh(sample)), nan_mask)
        self.assertEqual(torch.isnan(sample.atanh()), nan_mask)
        # For values -1.0 and 1.0, atanh should return -inf and inf respectively
        self.assertEqual(torch.isinf(torch.atanh(sample)), inf_mask)
        self.assertEqual(torch.isinf(sample.atanh()), inf_mask)

    # Tests bool tensor negation raises the correct error
    def test_neg_error_message(self, device):
        msg = ("Negation, the `\\-` operator, on a bool tensor is not supported."
               " If you are trying to invert a mask, use the `\\~` or"
               " `logical_not\\(\\)` operator instead.")

        t = torch.tensor((False, True), device=device)

        with self.assertRaisesRegex(RuntimeError, msg):
            torch.neg(t)

    @dtypes(*floating_types_and(torch.bfloat16, torch.half))
    @ops((_fn for _fn in unary_ufuncs if _fn.domain != (None, None)))
    def test_float_domains(self, device, dtype, op):
        if not op.supports_dtype(dtype, torch.device(device).type):
            raise unittest.SkipTest('unsupported dtype')

        eps = (1e-5, 1e-3, 1e-1, 1, 2, 10, 20, 50, 100)

        low, high = op.domain
        # NOTE: the following two loops are separated for readability
        if low is not None:
            low_tensor = torch.tensor(low, device=device, dtype=dtype)
            for epsilon in eps:
                lower_tensor = low_tensor - epsilon

                # Skips the test if the difference is not representable,
                #   which can occur if, for example, the difference is small
                #   and the dtype is imprecise (like bfloat16 is)
                if lower_tensor.item() == low_tensor.item():
                    continue

                result = op(lower_tensor)
                self.assertEqual(result.item(), float('nan'),
                                 msg=("input of {0} outside lower domain boundary"
                                      " {1} produced {2}, not nan!").format(lower_tensor.item(),
                                                                            low,
                                                                            result.item()))

        if high is not None:
            high_tensor = torch.tensor(high, device=device, dtype=dtype)
            for epsilon in eps:
                higher_tensor = high_tensor + epsilon

                # See above comment
                if higher_tensor.item() == high_tensor.item():
                    continue

                result = op(higher_tensor)
                self.assertEqual(result.item(), float('nan'),
                                 msg=("input of {0} outside upper domain boundary"
                                      " {1} produced {2}, not nan!").format(higher_tensor.item(),
                                                                            high,
                                                                            result.item()))

    # Tests that fn == method == inplace == jit on a simple single tensor input
    # TODO: should this jitting the method and inplace variants, too?
    @ops(unary_ufuncs)
    def test_variant_consistency(self, device, dtype, op):
        def _fn(t):
            return op(t)

        t = make_tensor((5, 5), device, dtype, low=op.domain[0], high=op.domain[1])
        expected = op(t)

        for alt, inplace in ((op.get_method(), False), (op.get_inplace(), True),
                             (torch.jit.script(_fn), False)):
            if alt is None:
                with self.assertRaises(RuntimeError):
                    alt(t.clone())

            if inplace and op.promotes_integers_to_float and dtype in integral_types() + (torch.bool,):
                # Assert that RuntimeError is raised
                # for inplace variant of Operators that
                # promote integer input to floating dtype.
                with self.assertRaises(RuntimeError):
                    alt(t.clone())
                continue

            actual = alt(t.clone())
            self.assertEqual(actual, expected, rtol=0, atol=0)

    # Helper for comparing torch tensors and numpy arrays
    # TODO: should this or assertEqual also validate that strides are equal?
    def assertEqualHelper(self, actual, expected, msg, *, dtype, exact_dtype=True, **kwargs):
        assert isinstance(actual, torch.Tensor)

        # Some NumPy functions return scalars, not arrays
        if isinstance(expected, Number):
            self.assertEqual(actual.item(), expected, **kwargs)
        elif isinstance(expected, np.ndarray):
            # Handles exact dtype comparisons between arrays and tensors
            if exact_dtype:
                # Allows array dtype to be float32 when comparing with bfloat16 tensors
                #   since NumPy doesn't support the bfloat16 dtype
                if expected.dtype == np.float32:
                    assert actual.dtype in (torch.bfloat16, torch.float32)
                else:
                    assert expected.dtype == torch_to_numpy_dtype_dict[actual.dtype]

            self.assertEqual(actual,
                             torch.from_numpy(expected).to(actual.dtype),
                             msg,
                             exact_device=False,
                             **kwargs)
        else:
            self.assertEqual(actual, expected, msg, exact_device=False, **kwargs)

    # Tests that the function and its (array-accepting) reference produce the same
    #   values on a range of tensors, including empty tensors, scalar tensors,
    #   1D tensors and a large 2D tensor with interesting and extremal values
    #   and discontiguities.
    @unittest.skipIf(not TEST_NUMPY, "NumPy not found")
    @suppress_warnings
    @ops(unary_ufuncs)
    def test_reference_numerics(self, device, dtype, op):
        include_extremals = (op.handles_complex_extremals if
                             dtype in (torch.cfloat, torch.cdouble) else op.handles_extremals)

        tensors = generate_numeric_tensors(device, dtype,
                                           domain=op.domain,
                                           include_large_values=op.handles_large_floats,
                                           include_extremal_values=include_extremals)
        for t in tensors:
            if dtype is torch.bfloat16:
                a = t.cpu().to(torch.float32).numpy()
            else:
                a = t.cpu().numpy()

            actual = op(t)
            expected = op.ref(a)

            # Crafts a custom error message for smaller, printable tensors
            if t.numel() < 10:
                msg = ("Failed to produce expected results! Input tensor was"
                       " {0}, torch result is {1}, and reference result is"
                       " {2}.").format(t, actual, expected)
            else:
                msg = None

            exact_dtype = True
            if op.promotes_integers_to_float and dtype in integral_types() + (torch.bool,):
                exact_dtype = False

                if dtype in [torch.uint8, torch.int8, torch.bool]:
                    # NOTE: For these dtypes, PyTorch computes in the default scalar type (float)
                    # while NumPy computes in float16
                    self.assertEqualHelper(actual, expected, msg, dtype=dtype,
                                           exact_dtype=exact_dtype, rtol=1e-3, atol=1e-2)
                    continue

            self.assertEqualHelper(actual, expected, msg, dtype=dtype, exact_dtype=exact_dtype)

    # Tests for testing (dis)contiguity consistency

    @ops(unary_ufuncs)
    def test_contig_vs_every_other(self, device, dtype, op):
        contig = make_tensor((1026,), device=device, dtype=dtype,
                             low=op.domain[0], high=op.domain[1])
        non_contig = contig[::2]

        self.assertTrue(contig.is_contiguous())
        self.assertFalse(non_contig.is_contiguous())

        self.assertEqual(op(contig)[::2], op(non_contig))

    @ops(unary_ufuncs)
    def test_contig_vs_transposed(self, device, dtype, op):
        contig = make_tensor((789, 357), device=device, dtype=dtype,
                             low=op.domain[0], high=op.domain[1])
        non_contig = contig.T

        self.assertTrue(contig.is_contiguous())
        self.assertFalse(non_contig.is_contiguous())

        self.assertEqual(op(contig).T, op(non_contig))

    @ops(unary_ufuncs)
    def test_non_contig(self, device, dtype, op):
        shapes = [(5, 7), (1024,)]
        for shape in shapes:
            contig = make_tensor(shape, device, dtype,
                                 low=op.domain[0], high=op.domain[1])
            non_contig = torch.empty(shape + (2,), device=device, dtype=dtype)[..., 0]
            non_contig.copy_(contig)

            self.assertTrue(contig.is_contiguous())
            self.assertFalse(non_contig.is_contiguous())

            self.assertEqual(op(contig), op(non_contig))

    @ops(unary_ufuncs)
    def test_non_contig_index(self, device, dtype, op):
        contig = make_tensor((2, 2, 1, 2), device, dtype,
                             low=op.domain[0], high=op.domain[1])
        non_contig = contig[:, 1, ...]
        contig = non_contig.contiguous()

        self.assertTrue(contig.is_contiguous())
        self.assertFalse(non_contig.is_contiguous())

        self.assertEqual(op(contig), op(non_contig))

    @ops(unary_ufuncs)
    def test_non_contig_expand(self, device, dtype, op):
        shapes = [(1, 3), (1, 7), (5, 7)]
        for shape in shapes:
            contig = make_tensor(shape, device, dtype,
                                 low=op.domain[0], high=op.domain[1])
            non_contig = contig.clone().expand(3, -1, -1)

            self.assertTrue(contig.is_contiguous())
            self.assertFalse(non_contig.is_contiguous())

            contig = op(contig)
            non_contig = op(non_contig)
            for i in range(3):
                self.assertEqual(contig, non_contig[i],
                                 msg='non-contiguous expand[' + str(i) + ']')

    @ops(unary_ufuncs)
    def test_contig_size1(self, device, dtype, op):
        contig = make_tensor((5, 100), device, dtype,
                             low=op.domain[0], high=op.domain[1])
        contig = contig[:1, :50]
        contig2 = torch.empty(contig.size(), device=device, dtype=dtype)
        contig2.copy_(contig)

        self.assertTrue(contig.is_contiguous())
        self.assertTrue(contig2.is_contiguous())

        self.assertEqual(op(contig), op(contig2))

    @ops(unary_ufuncs)
    def test_contig_size1_large_dim(self, device, dtype, op):
        contig = make_tensor((5, 2, 3, 1, 4, 5, 3, 2, 1, 2, 3, 4), device, dtype,
                             low=op.domain[0], high=op.domain[1])
        contig = contig[:1, :, :, :, :, :, :, :, :, :, :, :]
        contig2 = torch.empty(contig.size(), device=device, dtype=dtype)
        contig2.copy_(contig)

        self.assertTrue(contig.is_contiguous())
        self.assertTrue(contig2.is_contiguous())

        self.assertEqual(op(contig), op(contig2))

    # Tests that computation on a multiple batches is the same as
    # per-batch computation.
    @ops(unary_ufuncs)
    def test_batch_vs_slicing(self, device, dtype, op):
        input = make_tensor((1024, 512), dtype=dtype, device=device,
                            low=op.domain[0], high=op.domain[1])

        actual = op(input)
        expected = torch.stack([op(slice) for slice in input])

        self.assertEqual(actual, expected)

    def _test_out_arg(self, op, input, output):
        dtype = input.dtype
        out_dtype = output.dtype
        if dtype is out_dtype:
            expected = op(input)
            op(input, out=output)
            self.assertEqual(output, expected)
        else:
            with self.assertRaises(RuntimeError):
                op(input, out=output)

    def _test_out_promote_int_to_float_op(self, op, input, output):
        def compare_out(op, input, out):
            out_dtype = out.dtype
            expected = op(input)
            op(input, out=out)
            self.assertEqual(out, expected.to(out_dtype))

        dtype = input.dtype
        out_dtype = output.dtype
        if out_dtype.is_floating_point and not dtype.is_complex:
            compare_out(op, input, output)
        elif out_dtype.is_floating_point and dtype.is_complex:
            # Can't cast complex to float
            with self.assertRaises(RuntimeError):
                op(input, out=output)
        elif out_dtype.is_complex:
            compare_out(op, input, output)
        else:
            # Can't cast to Integral types
            with self.assertRaises(RuntimeError):
                op(input, out=output)

    @ops(unary_ufuncs)
    def test_out_arg_all_dtypes(self, device, dtype, op):
        input = make_tensor((64, 64), dtype=dtype, device=device,
                            low=op.domain[0], high=op.domain[1])

        for out_dtype in all_types_and_complex_and(torch.bool, torch.half):
            out = torch.empty_like(input, dtype=out_dtype)
            if op.promotes_integers_to_float:
                self._test_out_promote_int_to_float_op(op, input, out)
            else:
                self._test_out_arg(op, input, out)

    @dtypes(*(torch.testing.get_all_int_dtypes() + [torch.bool] +
              torch.testing.get_all_fp_dtypes(include_bfloat16=False)))
    def test_nan_to_num(self, device, dtype):
        for contiguous in [False, True]:
            x = make_tensor((64, 64), low=0., high=100., dtype=dtype, device=device)

            if dtype.is_floating_point:
                # Add extremal values.
                extremals = [float('nan'), float('inf'), -float('inf')]
                for idx, extremal in zip(torch.randint(0, 63, (3,)), extremals):
                    x[idx, :] = extremal

            if not contiguous:
                x = x.T

            # With args
            nan = random.random()
            posinf = random.random() * 5
            neginf = random.random() * 10

            self.compare_with_numpy(lambda x: x.nan_to_num(nan=nan, posinf=posinf),
                                    lambda x: np.nan_to_num(x, nan=nan, posinf=posinf),
                                    x)
            self.compare_with_numpy(lambda x: x.nan_to_num(posinf=posinf, neginf=neginf),
                                    lambda x: np.nan_to_num(x, posinf=posinf, neginf=neginf),
                                    x)

            # Out Variant
            out = torch.empty_like(x)
            result = torch.nan_to_num(x)
            torch.nan_to_num(x, out=out)
            self.assertEqual(result, out)

            result = torch.nan_to_num(x, nan=nan, posinf=posinf, neginf=neginf)
            torch.nan_to_num(x, out=out, nan=nan, posinf=posinf, neginf=neginf)
            self.assertEqual(result, out)

instantiate_device_type_tests(TestUnaryUfuncs, globals())

if __name__ == '__main__':
    run_tests()
