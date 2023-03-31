# Generate filter coefficients of raised cosine
def raisedCosineDesign(alpha, span, L):
    import numpy as np

    """
    Raised cosine FIR filter design
    Parameters:
        alpha: roll-off factor
        span: filter span in symbols
        L: oversampling factor (i.e. each symbol contains L samples)
    Returns:
        p: filter coefficients b of the designed raised cosine filter
    """
    t = np.arange(-span / 2, span / 2 + 1 / L, 1 / L)  # +/- discrete-time base

    with np.errstate(divide="ignore", invalid="ignore"):
        A = np.divide(np.sin(np.pi * t), (np.pi * t))  # Assume Tsym=1
        B = np.divide(np.cos(np.pi * alpha * t), 1 - (2 * alpha * t) ** 2)
        p = A * B

    # Handle singularities
    p[np.argwhere(np.isnan(p))] = 1  # Singularity at p(t=0)
    p[np.argwhere(np.isinf(p))] = (alpha / 2) * np.sin(np.divide(np.pi, (2 * alpha)))

    return p


# Generate filter coefficients of Gaussian LPF
def gaussianLPF(BT, Tb, L, k):
    import numpy as np

    """
    Generate filter coefficients of Gaussian low pass filter (used in gmsk_mod)
    Parameters:
        BT: BT product -- Bandwidth * bit period
        Tb: bit period
        L: oversampling factor (number of samples per bit)
        k: span length of the pulse (bit interval)
    Returns:
        h_norm: normalized filter coefficients of Gaussian LPF
    """
    B = BT / Tb  # bandwidth of the filter
    # Truncated time limits for the filter
    t = np.arange(-k * Tb, k * Tb + Tb / L, step=Tb / L)
    h = (
        B
        * np.sqrt(2 * np.pi / (np.log(2)))
        * np.exp(-2 * (t * np.pi * B) ** 2 / (np.log(2)))
    )
    h_norm = h / np.sum(h)

    return h_norm
