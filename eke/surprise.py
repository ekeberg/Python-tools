import numpy as _numpy
from . import log_factorial as _log_factorial
from scipy.special import factorial as _factorial

def surprise(expectation, outcome, k_max=None):
    if not _numpy.issubdtype(expectation.dtype, _numpy.float):
        raise ValueError("expectation must be of float type")
    if not _numpy.issubdtype(outcome.dtype, _numpy.integer):
        raise ValueError("outcome must be of integer type")
    
    if k_max is None:
        k_max = outcome.max()*2
    max_outcome = outcome.max()
    surprise_pix = _numpy.zeros(expectation.shape)
    k_mask = outcome < k_max
    # surprise_pix[k_mask] =
    # -_numpy.log(expectation[k_mask]**outcome[k_mask]
    #  *_numpy.exp(-expectation[k_mask])
    # / _factorial(outcome[k_mask]))
    surprise_pix[k_mask] = -(outcome[k_mask]*_numpy.log(expectation[k_mask])
                             + -expectation[k_mask]
                             - _numpy.log(_factorial(outcome[k_mask])))
    surprise_pix[~k_mask] = (-(expectation[~k_mask] - outcome[~k_mask])**2
                             / (2*expectation[~k_mask]))
    surprise = surprise_pix.sum()

    
    surprise_expectation_pix = _numpy.zeros(surprise_pix.shape)
    surprise_expectation = 0
    for k in range(max_outcome):
        if k < k_max:
            fac_k = _factorial(k)
            this_exp = (expectation**k*_numpy.exp(-expectation)/fac_k
                        * _numpy.log(expectation**k
                                     * _numpy.exp(-expectation)
                                     / fac_k))
            this_exp[_numpy.isnan(this_exp)] = 0

            surprise_expectation_pix -= this_exp
        else:
            this_exp = (_numpy.exp(-(expectation - k)**2 / (2 * expectation))
                        * (-(expectation - k)**2 / (2 * expectation)))
            surprise_expectation_pix -= this_exp

    surprise_expectation = surprise_expectation_pix.sum()

    surprise_variance_pix = _numpy.zeros(surprise_pix.shape)
    for k in range(max_outcome):
        if k < k_max:
            fac_k = _factorial(k)
            surprise_variance_pix += (
                expectation**k * _numpy.exp(-expectation) / fac_k
                * (k*_numpy.log(expectation)
                   - expectation
                   - _numpy.log(fac_k))**2)
        else:
            surprise_variance_pix += (
                _numpy.exp(-(expectation - k)**2 / (2*expectation))
                * (-(expectation - k)**2 / (2*expectation))**2)
    surprise_variance_pix -= surprise_expectation_pix**2
    surprise_variance = surprise_variance_pix.sum()

    score = abs(surprise - surprise_expectation) / surprise_variance

    return score
