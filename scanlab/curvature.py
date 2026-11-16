"""Exact squared-loss curvature of small supplied dense affine recurrences.

Hand-written forward-over-reverse derivatives. All recurrences here run
sequentially; no dense Hessian is built and no parallel-latency claim is made.
"""
from fractions import Fraction as F
from .dense import prepare, mv, transpose, plus, _dot
from .exact import vector


def curvature_audit(pairs, initial, direction, targets):
    ps, h = prepare(pairs, initial)
    n, d = len(ps), len(h)
    if not isinstance(direction, dict) or set(direction) != {'pairs', 'initial'}:
        raise ValueError('direction needs exactly pairs and initial')
    ds, dh = prepare(direction['pairs'], direction['initial'])
    if len(ds) != n or len(dh) != d:
        raise ValueError('direction shape mismatch')
    if not isinstance(targets, (list, tuple)) or len(targets) != n:
        raise ValueError('targets length mismatch')
    target = [vector(row, d) for row in targets]
    dr = {'pairs': [{'matrix': a, 'write': b} for a, b in ds], 'initial': dh}
    states, dots, seconds, previous, previous_dots = [], [], [], [], []
    ddh = (F(0),) * d
    for (a, b), (da, db) in zip(ps, ds):
        previous.append(h)
        previous_dots.append(dh)
        ddh = plus(mv(a, ddh), tuple(2 * v for v in mv(da, dh)))
        dh = plus(plus(mv(a, dh), mv(da, h)), db)
        h = plus(mv(a, h), b)
        states.append(h)
        dots.append(dh)
        seconds.append(ddh)
    residual = [tuple(x - y for x, y in zip(state, row)) for state, row in zip(states, target)]
    adjoints, adjoint_dots = [None] * n, [None] * n
    for t in reversed(range(n)):
        adjoints[t] = residual[t]
        adjoint_dots[t] = dots[t]
        if t + 1 < n:
            a, da = ps[t + 1][0], ds[t + 1][0]
            adjoints[t] = plus(adjoints[t], mv(transpose(a), adjoints[t + 1]))
            adjoint_dots[t] = plus(adjoint_dots[t], plus(mv(transpose(a), adjoint_dots[t + 1]), mv(transpose(da), adjoints[t + 1])))
    gradient_pairs, hvp_pairs = [], []
    for lam, dlam, prev, dprev in zip(adjoints, adjoint_dots, previous, previous_dots):
        gradient_pairs.append({'matrix': tuple(tuple(x * y for y in prev) for x in lam), 'write': lam})
        hvp_pairs.append({'matrix': tuple(tuple(dx * y + x * dy for y, dy in zip(prev, dprev)) for x, dx in zip(lam, dlam)), 'write': dlam})
    zero = (F(0),) * d
    gradient = {'pairs': gradient_pairs, 'initial': mv(transpose(ps[0][0]), adjoints[0]) if n else zero}
    hvp = {'pairs': hvp_pairs, 'initial': plus(mv(transpose(ds[0][0]), adjoints[0]), mv(transpose(ps[0][0]), adjoint_dots[0])) if n else zero}
    loss = sum((x * x / 2 for row in residual for x in row), F(0))
    first = sum((x * y for row, tangent in zip(residual, dots) for x, y in zip(row, tangent)), F(0))
    second = sum((v * v + r * w for row, tangent, accel in zip(residual, dots, seconds) for r, v, w in zip(row, tangent, accel)), F(0))
    assert _dot(gradient, dr) == first
    assert _dot(hvp, dr) == second
    return {'states': states, 'state_jvp': dots, 'state_second_directional': seconds,
            'direction': dr, 'targets': target, 'residual': residual,
            'squared_loss': loss, 'gradient': gradient, 'hessian_vector': hvp,
            'directional_first': first, 'directional_second': second,
            'witness': {'previous_states': previous, 'previous_state_jvp': previous_dots,
                        'loss_adjoints': adjoints, 'loss_adjoint_jvp': adjoint_dots,
                        'gradient_dot_direction': _dot(gradient, dr), 'hvp_dot_direction': _dot(hvp, dr),
                        'first_identity_gap': _dot(gradient, dr) - first,
                        'second_identity_gap': _dot(hvp, dr) - second},
            'contract': 'exact squared-loss derivatives of supplied dense affine pairs along a fixed parameter direction; sequential Python, no dense Hessian construction, trained model, speed or convexity guarantee'}
