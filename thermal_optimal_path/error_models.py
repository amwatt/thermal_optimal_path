import numpy as np
from numba import jit


@jit
def _sign_error(a, b, sign, sqrt):
    e = (a + sign * b) ** 2
    if sqrt:
        e = np.sqrt(e)
    return e


@jit
def error(a, b, sqrt=False, both_signs=True):
    """ Basic mean squared error model.

    Parameters
    ----------
    a: float
        The first value
    b: float
        The second value
    sqrt: bool, opt
        If true, computes the root mean squared error.
    both_signs: bool, opt
        If True, the error with uses both +b (anti-correlation) and -b (correlation). If
        false, only -b (correlation).

    Returns
    -------
    A float for the error between the two values.
    """
    e = _sign_error(a, b, -1, sqrt)
    if both_signs:
        plus = _sign_error(a, b, 1, sqrt)
        e = min(e, plus)
    return e
