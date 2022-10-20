def square_wave(f,overSampRate,nCyl):
    """
    Generate square wave signal with the following parameters
    Parameters:
    f : frequency of square wave in Hertz
    overSampRate : oversampling rate (integer)
    nCyl : number of cycles of square wave to generate
    Returns:
    (t,g) : time base (t) and the signal g(t) as tuple
    Example:
    f=10; overSampRate=30;nCyl = 5;
    (t,g) = square_wave(f,overSampRate,nCyl)
    """
    fs = overSampRate*f # sampling frequency
    t = np.arange(0,nCyl*1/f-1/fs,1/fs) # time base
    g = np.sign(np.sin(2*np.pi*f*t)) # replace with cos if a cosine wave is desired

    return (t,g) # return time base and signal g(t) as tuple

def chirp_demo():
    """
    Generating and plotting a chirp signal
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.signal import chirp
    fs = 500 # sampling frequency in Hz
    t =np.arange(start = 0, stop = 1,step = 1/fs) #total time base from 0 to 1 second
    g = chirp(t, f0=1, t1=0.5, f1=20, phi=0, method='linear')
    plt.plot(t,g); plt.show()

def fft_example_1():

    from scipy.fftpack import fft, ifft
    import numpy as np
    import matplotlib.pyplot as plt

    fc=10 # frequency of the carrier
    fs=32*fc # sampling frequency with oversampling factor=32
    t=np.arange(start = 0,stop = 2,step = 1/fs) # 2 seconds duration
    x=np.cos(2*np.pi*fc*t) # time domain signal (real number)

    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1)
    ax1.plot(t,x) #plot the signal
    ax1.set_title('$x[n]= cos(2 \pi 10 t)$')
    ax1.set_xlabel('$t=nT_s$')
    ax1.set_ylabel('$x[n]$')

    N=256 # FFT size
    X = fft(x,N) # N-point complex DFT, output contains DC at index 0
    # Nyquist frequency at N/2 th index positive frequencies from
    # index 2 to N/2-1 and negative frequencies from index N/2 to N-1

    # calculate frequency bins with FFT
    df=fs/N # frequency resolution
    sampleIndex = np.arange(start = 0,stop = N) # raw index for FFT plot
    f=sampleIndex*df # x-axis index converted to frequencies

    ax2.stem(sampleIndex,abs(X),use_line_collection=True) # sample values on x-axis
    ax2.set_title('X[k]');ax2.set_xlabel('k');ax2.set_ylabel('|X(k)|');
    ax3.stem(f,abs(X),use_line_collection=True); # x-axis represent frequencies
    ax3.set_title('X[f]');ax3.set_xlabel('frequencies (f)');ax3.set_ylabel('|X(f)|');
    fig.show()

    from scipy.fftpack import fftshift
    #re-order the index for emulating fftshift
    sampleIndex = np.arange(start = -N//2,stop = N//2) # // for integer division
    X1 = X[sampleIndex] #order frequencies without using fftShift
    X2 = fftshift(X) # order frequencies by using fftshift
    df=fs/N # frequency resolution
    f=sampleIndex*df # x-axis index converted to frequencies

    #plot ordered spectrum using the two methods
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1)#subplots creation
    ax1.stem(sampleIndex,abs(X1), use_line_collection=True)# result without fftshift
    ax1.stem(sampleIndex,abs(X2),'r',use_line_collection=True) #result with fftshift
    ax1.set_xlabel('k');ax1.set_ylabel('|X(k)|')
    ax2.stem(f,abs(X1), use_line_collection=True)
    ax2.stem(f,abs(X2),'r' , use_line_collection=True)
    ax2.set_xlabel('frequencies (f)'),ax2.set_ylabel('|X(f)|'); fig.show()


def fft_example_2():

    from scipy.fftpack import fft, ifft, fftshift, ifftshift
    import numpy as np
    import matplotlib.pyplot as plt

    A = 0.5 # amplitude of the cosine wave
    fc=10 # frequency of the cosine wave in Hz
    phase=30 # desired phase shift of the cosine in degrees
    fs=32*fc # sampling frequency with oversampling factor 32
    t=np.arange(start = 0,stop = 2,step = 1/fs) # 2 seconds duration
    phi = phase*np.pi/180; # convert phase shift in degrees in radians
    x=A*np.cos(2*np.pi*fc*t+phi) # time domain signal with phase shift

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=4, ncols=1)
    ax1.plot(t,x) # plot time domain representation
    ax1.set_title(r'$x(t) = 0.5 cos (2 \pi 10 t + \pi/6)$')
    ax1.set_xlabel('time (t seconds)');ax1.set_ylabel('x(t)')

    N=256 # FFT size
    X = 1/N*fftshift(fft(x,N)) # N-point complex DFT

    df=fs/N # frequency resolution
    sampleIndex = np.arange(start = -N//2,stop = N//2) # // for integer division
    f=sampleIndex*df # x-axis index converted to ordered frequencies

    ax2.stem(f,abs(X), use_line_collection=True) # magnitudes vs frequencies
    ax2.set_xlim(-30, 30)
    ax2.set_title('Amplitude spectrum')
    ax2.set_xlabel('f (Hz)');ax2.set_ylabel(r'$ \left| X(k) \right|$')

    phase=np.arctan2(np.imag(X),np.real(X))*180/np.pi # phase information
    ax3.plot(f,phase) # phase vs frequencies

    X2=X #store the FFT results in another array
    # detect noise (very small numbers (eps)) and ignore them
    threshold = max(abs(X))/10000; # tolerance threshold
    X2[abs(X)<threshold]=0 # maskout values below the threshold
    phase=np.arctan2(np.imag(X2),np.real(X2))*180/np.pi # phase information

    ax4.stem(f,phase, use_line_collection=True) # phase vs frequencies
    ax4.set_xlim(-30, 30); ax4.set_title('Phase spectrum')
    ax4.set_ylabel(r"$\angle$ X[k]");ax4.set_xlabel('f(Hz)')
    fig.show()

    x_recon = N*ifft(ifftshift(X),N) # reconstructed signal
    t = np.arange(start = 0,stop = len(x_recon))/fs # recompute time index
    
    fig2, ax5 = plt.subplots()
    ax5.plot(t,np.real(x_recon)) # reconstructed signal
    ax5.set_title('reconstructed signal')
    ax1.set_xlabel('time (t seconds)');ax1.set_ylabel('x(t)');
    fig2.show()

