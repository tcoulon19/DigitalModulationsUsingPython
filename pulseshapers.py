def raisedCosineDesign(alpha, span,L):

    import numpy as np

    '''
    Raised cosine FIR filter design
    Parameters:
        alpha: roll-off factor
        span: filter span in symbols
        L: oversampling factor (i.e. each symbol contains L samples)
    Returns:
        p: filter coefficients b of the designed raised cosine filter
    '''
    t = np.arange(-span/2, span/2+1/L, 1/L) # +/- discrete-time base
    
    with np.errstate(divide = 'ignore', invalid = 'ignore'):
        A = np.divide(np.sin(np.pi*t),(np.pi*t)) # Assume Tsym=1
        B = np.divide(np.cos(np.pi*alpha*t),1-(2*alpha*t)**2)
        p = A*B

    # Handle singularities
    p[np.argwhere(np.isnan(p))] = 1 # Singularity at p(t=0)
    p[np.argwhere(np.isinf(p))] = (alpha/2)*np.sin(np.divide(np.pi,(2*alpha)))

    return p

    
    
     