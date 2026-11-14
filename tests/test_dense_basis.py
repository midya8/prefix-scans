"""Coordinate changes test covariance, not another scan implementation."""
from copy import deepcopy
from fractions import Fraction as F
import unittest

from scanlab import dense
from test_dense import case, matmul


def transposed(a):
    return [list(row) for row in zip(*a)]


def vector_product(a, x):
    return tuple(row[0] for row in matmul(a, [[F(v)] for v in x]))


def conjugate(a, left, right):
    return tuple(tuple(row) for row in matmul(matmul(left, a), right))


def parameters(value, basis, inverse):
    return {
        'initial': vector_product(basis, value['initial']),
        'pairs': [
            {'matrix': conjugate(p['matrix'], basis, inverse),
             'write': vector_product(basis, p['write'])}
            for p in value['pairs']
        ],
    }


class DenseBasis(unittest.TestCase):
    def test_nonorthogonal_basis_changes_preserve_derivative_pairings(self):
        bases = [
            ([[1, 1], [0, 1]], [[1, -1], [0, 1]]),
            ([[2, 0], [1, 1]], [[F(1, 2), 0], [F(-1, 2), 1]]),
            ([[1, -2], [1, -1]], [[-1, 2], [-1, 1]]),
        ]
        for length in [0, 1, 2, 5, 8, 32]:
            for seed in [17, 43]:
                inp = case(length, 2, seed + length)
                if length > 1:
                    # Include singular resets; invertibility is needed only
                    # for the coordinate change, not for each transition.
                    inp['pairs'][1]['matrix'] = [[0, 0], [0, 0]]
                before_input = deepcopy(inp)
                before = dense.audit(**inp)
                for basis, inverse in bases:
                    with self.subTest(length=length, seed=seed, basis=basis):
                        inverse_t = transposed(inverse)
                        transformed = parameters(inp, basis, inverse)
                        transformed['direction'] = parameters(inp['direction'], basis, inverse)
                        transformed['cotangents'] = [vector_product(inverse_t, q)
                                                     for q in inp['cotangents']]
                        after = dense.audit(**transformed)
                        for key in ['states', 'state_jvp']:
                            self.assertEqual(after[key], [vector_product(basis, x)
                                                          for x in before[key]])
                        self.assertEqual(after['adjoints'], [vector_product(inverse_t, x)
                                                            for x in before['adjoints']])
                        self.assertEqual(after['vjp']['initial'],
                                         vector_product(inverse_t, before['vjp']['initial']))
                        for old, new in zip(before['vjp']['pairs'], after['vjp']['pairs']):
                            self.assertEqual(new['write'], vector_product(inverse_t, old['write']))
                            self.assertEqual(new['matrix'],
                                conjugate(old['matrix'], inverse_t, transposed(basis)))
                        self.assertEqual(after['transpose'], before['transpose'])
                        self.assertEqual(after['transpose']['exact_gap'], 0)
                self.assertEqual(inp, before_input)


if __name__ == '__main__':
    unittest.main()
