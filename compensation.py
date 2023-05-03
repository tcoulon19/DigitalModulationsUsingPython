import numpy as np
from numpy import mean,real,imag,sign,abs,sqrt,sum

def dc_compensation(z):
    """Function to estimate and remove DC impairments in the IQ branch.

    Args:
        z: DC impaired signal sequence (numpy format)
    Returns:
        v: DC removed signal sequence
    """

    iDCest = mean(real(z)) # Estimated DC on I branch
    qDCest = mean(imag(z)) # Estimated DC on Q branch
    v = z - (iDCest+1j*qDCest) # Remove estimated DCs

    return v
