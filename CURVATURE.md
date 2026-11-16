# Dense squared-loss curvature

`curvature_audit(pairs, initial, direction, targets)` differentiates the supplied dense affine recurrence and its half-squared state error. It returns the full parameter gradient and Hessian-vector product, without constructing the dense Hessian. All operations in this module run sequentially. This is not a scan-depth, hardware-speed or trained-model result.

## Forward and reverse equations

Write the supplied recurrence and fixed-target loss as

```text
h_t = A_t h_(t-1) + b_t
L = (1/2) sum_t ||h_t - target_t||^2.
```

The initial state, every matrix entry and every write entry are parameters. The direction supplies their first derivatives; it is fixed, so their second derivatives are zero. With `v_t = D h_t` and `w_t = D^2 h_t`,

```text
v_t = A_t v_(t-1) + dA_t h_(t-1) + db_t
w_t = A_t w_(t-1) + 2 dA_t v_(t-1).
```

The initial tangent is the supplied initial-state direction. The initial second derivative is zero. The report keeps every state, tangent and second directional state derivative, including the previous-state witnesses used for matrix gradients.

Let `r_t=h_t-target_t`. With no contribution after the last step, the loss adjoints and their directional derivatives are

```text
lambda_t = r_t + A_(t+1)^T lambda_(t+1)
dlambda_t = v_t + A_(t+1)^T dlambda_(t+1)
                 + dA_(t+1)^T lambda_(t+1).
```

Both the next-step index and transposes matter. The matrix gradient is the outer product `lambda_t h_(t-1)^T`; the write gradient is `lambda_t`. Their directional derivatives are

```text
HVP_A_t = dlambda_t h_(t-1)^T + lambda_t v_(t-1)^T
HVP_b_t = dlambda_t.
```

For the initial state, the gradient is `A_0^T lambda_0` and its HVP is `dA_0^T lambda_0 + A_0^T dlambda_0`. The empty sequence has zero loss, gradient and HVP, including its initial-state entries.

The implementation checks two exact scalar identities:

```text
gradient . direction = sum_t r_t . v_t
HVP . direction = sum_t [v_t . v_t + r_t . w_t].
```

These checks are useful but not sufficient alone. An independent full-polynomial reference checks every gradient and HVP coordinate in the tests.

## Squared loss need not be convex in all parameters

A one-step scalar control uses `A=0`, `h_initial=0`, `b=0`, target one, and direction `dA=dh_initial=1`, `db=0`. Along the parameter line, the state is `epsilon^2`, so

```text
L(epsilon) = (epsilon^2 - 1)^2 / 2
L(0) = 1/2
L''(0) = -2.
```

The state JVP is zero, its second derivative is two, and the exact HVP paired with the direction is minus two. Keeping only the Gauss-Newton squared-tangent term would give zero. This is a developed finite counterexample, not a model-training experiment. No positive-semidefinite Hessian or optimizer-improvement guarantee is claimed.

## Run and input contract

```python
from scanlab import curvature_audit

report = curvature_audit(
    [{'matrix': [[0]], 'write': [0]}], [0],
    {'pairs': [{'matrix': [[1]], 'write': [0]}], 'initial': [1]},
    [[1]],
)
assert report['directional_second'] == -2
```

```bash
python -m scanlab curvature examples/curvature-input.json
python examples/verify_wheel.py
```

The JSON object contains exactly `pairs`, `initial`, `direction` and `targets`. Each pair has exactly `matrix` and `write`; the direction has exactly `pairs` and `initial` in the same layout. Every target is a vector of the state width, one per time step. The established dense limits remain length 0..32 and width 1..4. Numbers use the established exact parser: integers, Fractions or rational/decimal strings, never floats or bools, with at most 80-character strings and 128-bit input numerator/denominator components. CLI JSON remains capped at 1 MB and rejects duplicate/unknown keys and nonfinite constants. Internal exact values can grow beyond the input caps.

## Verification and scope

Seven added tests compare all 672 parameter coordinates across 36 models against independent bivariate polynomial expansions. Their mixed coefficients give each HVP coordinate without differentiating the production reverse recurrence. Another 24 full directional models check all states, state JVPs, second directional state derivatives and loss coefficients. There are 24 exact Hessian symmetry checks, the negative-curvature control above, empty/single-step/singular/maximum-size and zero-direction cases, strict validation and CLI checks, and input/output/RNG isolation checks.

The full suite has 71 tests. The installed-wheel verifier runs them all outside checkout, checks all ten package origins before and after, compares seven complete module CLI reports and regenerates all three original figures. All six earlier complete CLI outputs and all three original SVG files are unchanged. The earlier dense and diagonal reports keep their existing fields and claims.

This is hand-written forward-over-reverse differentiation of one finite supplied recurrence and fixed squared loss. No general AD engine, stochastic Hessian estimator, learned task, convergence result, dense-Hessian allocation, parallel-span analysis or speed claim is added. The chain rule, tangent and adjoint equations, and forward-over-reverse Hessian-vector construction are standard. The original derivative literature and source notes in [ADJOINT.md](ADJOINT.md) and dense conventions in [DENSE.md](DENSE.md) still apply; this extension does not claim a new differentiation algorithm.
