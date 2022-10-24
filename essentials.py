# PSD estimation using Welch method
def plotWelchPSD(x, fs, fc, ax = None, color = 'b', label = None):

    from scipy.signal import welch
    from numpy import log10, hanning
    import matplotlib.pyplot as plt

    nx = max(x.shape)
    na = 16 # averaging factor to plot averaged welch spectrum
    w = hanning(nx//na)
    f, Pxx = welch(x, fs, window = w, noverlap = 0)
    indices = (f>=fc) & (f<4*fc) # to plot PSD from Fc to 4*Fc
    Pxx = Pxx[indices]/Pxx[indices][0] # normalize PSD w.r.t. Fc

    plt.figure(1)
    plt.plot(f[indices]-fc, 10*log10(Pxx))
    plt.title('Welch plot')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('PSD (dB/Hz)')
    plt.savefig('welch_example_im2.png')


# Convolution by brute force
def conv_brute_force(x, h):

    import numpy as np
    
    '''
    x: input, numpy vector
    h: impulse response, numpy vector
    y: convolution of x and h
    '''

    N = len(x)
    M = len(h)
    y = np.zeros(N+M-1) # Array filled with zeros

    for i in np.arange(0,N):
        for j in np.arange(0,M):
            y[i+j] = y[i+j] + x[i] * h[j]
    
    return y


# Create Toeplitz matrix using h and length of x
def convMatrix(h,p):

    '''
    Construct convolution matrix of size (N+p-1) x p from input matrix h of size N
    h: numpy vector of length N
    p: scalar value
    H: convolution matrix of size (N+p-1) x p
    '''

    import numpy as np

    col = np.hstack((h,np.zeros(p-1)))
    row = np.hstack((h[0],np.zeros(p-1)))

    from scipy.linalg import toeplitz
    H = toeplitz(col,row)

    return H


# Convolution using Toeplitz matrix
def my_convolve(h,x):

    H = convMatrix(h, len(x))
    y = H @ x.transpose()

    return y



