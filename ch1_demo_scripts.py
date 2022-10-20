def sine_wave_demo():

    """
    Simulate a sinusoidal signal with given sampling rate
    """
    import numpy as np
    import matplotlib.pyplot as plt # library for plotting
    from signalgen import sine_wave # import the function

    f = 10 #frequency = 10 Hz
    overSampRate = 30 #oversammpling rate
    phase = 1/3*np.pi #phase shift in radians
    nCyl = 5 # desired number of cycles of the sine wave
    (t,g) = sine_wave(f,overSampRate,phase,nCyl) #function call

    plt.plot(t,g) # plot using pyplot library from matplotlib package
    plt.title('Sine wave f='+str(f)+' Hz') # plot title
    plt.xlabel('Time (s)') # x-axis label
    plt.ylabel('Amplitude') # y-axis label
    plt.savefig('Sine wave.png') # display the figure


def scipy_square_wave():

    """
    Generate a square wave with given sampling rate
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy import signal

    f = 10
    overSampRate = 30 # oversampling rate
    nCyl = 5 # number of cycles to generate
    fs = overSampRate*f # sampling frequency
    t = np.arange(start=0,stop=nCyl*1/f,step=1/fs) # time base
    g = signal.square(2 * np.pi * f * t, duty = 0.2)
    plt.plot(t,g); plt.show()

    plt.plot(t,g) # plot using pyplot library from matplotlib package
    plt.title('Square wave f='+str(f)+' Hz') # plot title
    plt.xlabel('Time (s)') # x-axis label
    plt.ylabel('Amplitude') # y-axis label
    plt.savefig('Square wave.png') # display the figure


def rectangular_pulse_demo():

    from signalgen import rect_pulse
    import matplotlib.pyplot as plt 

    A = 1
    fs = 500
    T = .2
    (t,g) = rect_pulse(A, fs, T)

    plt.plot(t,g) # plot using pyplot library from matplotlib package
    plt.title('Rectangular pulse') # plot title
    plt.xlabel('Time (s)') # x-axis label
    plt.ylabel('Amplitude') # y-axis label
    plt.savefig('Rect pulse.png') # display the figure


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

    plt.plot(t,g) # plot using pyplot library from matplotlib package
    plt.title('Chirp') # plot title
    plt.xlabel('Time (s)') # x-axis label
    plt.ylabel('Amplitude') # y-axis label
    plt.savefig('Chirp signal.png') # display the figure


def fft_example_1():

    from scipy.fftpack import fft, ifft, fftshift
    import numpy as np
    import matplotlib.pyplot as plt

    fc=10 # frequency of the carrier
    fs=32*fc # sampling frequency with oversampling factor=32
    t=np.arange(start = 0,stop = 2,step = 1/fs) # 2 seconds duration
    x=np.cos(2*np.pi*fc*t) # time domain signal (real number)

    plt.figure(0)
    plt.plot(t,x) # plot using pyplot library from matplotlib package
    plt.title('Time domain') # plot title
    plt.xlabel('Time (s)') # x-axis label
    plt.ylabel('Amplitude') # y-axis label
    plt.savefig('fft_example_1_im1.png') # display the figure

    # # fft without fftshift
    # N=fs # FFT size
    # X = fft(x,N) # N-point complex DFT, output contains DC at index 0
    # # Nyquist frequency at N/2 th index positive frequencies from
    # # index 2 to N/2-1 and negative frequencies from index N/2 to N-1
    # # calculate frequency bins with FFT
    # df=fs/N # frequency resolution
    # sampleIndex = np.arange(start = 0,stop = N) # raw index for FFT plot
    # f=sampleIndex*df # x-axis index converted to frequencies

    # fft with fftshift
    N=fs # FFT size
    X = fftshift(fft(x,N)) # N-point complex DFT, output contains DC at index 0
    df=fs/N # frequency resolution
    sampleIndex = np.arange(start = -N//2,stop = N//2) # raw index for FFT plot
    f=sampleIndex*df # x-axis index converted to frequencies

    plt.figure(1)
    plt.stem(f,abs(X)) # plot using pyplot library from matplotlib package
    plt.title('Freuency domain') # plot title
    plt.xlabel('f (Hz)') # x-axis label
    plt.ylabel('Amplitude') # y-axis label
    plt.savefig('fft_example_1_im2.png') # display the figure


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

