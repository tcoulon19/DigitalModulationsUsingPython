import numpy as np


# Sine wave
def sine_wave(f, overSampRate, phase, nCyl):

    '''
    f : frequency of sine wave in Hertz
    overSampRate : oversampling rate (integer)
    phase : desired phase shift in radians
    nCyl : number of cycles of sine wave to generate
    '''

    fs = overSampRate * f # sampling frequency
    t = np.arange(0, nCyl*1/f-1/fs, 1/fs) # time base
    g = np.sin(2*np.pi*f*t + phase)
    
    return (t, g)


# Square wave
def square_wave(f, overSampRate, nCyl):

    fs = overSampRate*f # sampling frequency
    t = np.arange(0, nCyl*1/f-1/fs, 1/fs) # time base
    g = np.sign(np.sin(2*np.pi*f*t))

    return (t,g)


# Rectangular pulse
def rect_pulse(A, fs, T):

    """
    Generate isolated rectangular pulse with the following parameters
    Parameters:
    A : amplitude of the rectangular pulse
    fs : sampling frequency in Hz
    T : duration of the pulse in seconds
    """

    t = np.arange(-.5,.5,1/fs) # time base
    rect = (t > -T/2)*(t < T/2) + .5*(t == T/2) + .5*(T==-T/2)
    g = A* rect
    return (t,g)


# Gaussian pulse
def gaussian_pulse(fs, sigma):

    t = np.arange(-.5, .5, 1/fs)
    g = 1/(np.sqrt(2*np.pi)*sigma)*(np.exp(-t**2/(2*sigma**2)))
    return(t,g)

