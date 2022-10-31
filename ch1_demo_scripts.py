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
    plt.savefig('Ch1_images/Sine wave.png') # display the figure


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
    plt.savefig('Ch1_images/Square wave.png') # display the figure


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
    plt.savefig('Ch1_images/Rect pulse.png') # display the figure


def gaussian_pulse_demo():

    from signalgen import gaussian_pulse
    import matplotlib.pyplot as plt

    fs = 500
    sigma = .1
    (t,g) = gaussian_pulse(fs, sigma)

    plt.plot(t,g) # plot using pyplot library from matplotlib package
    plt.title('Gaussian pulse') # plot title
    plt.xlabel('Time (s)') # x-axis label
    plt.ylabel('Amplitude') # y-axis label
    plt.savefig('Ch1_images/Gaussian pulse.png') # display the figure


def chirp_demo():
    """
    Generating and plotting a chirp signal
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.signal import chirp

    '''
    f0: starting frequency
    t1: time at end of T
    f1: frequency at end of T
    phi: starting phase
    '''

    fs = 500 # sampling frequency in Hz
    t =np.arange(start = 0, stop = 1,step = 1/fs) #total time base from 0 to 1 second
    g = chirp(t, f0=1, t1=0.5, f1=20, phi=0, method='linear')

    plt.plot(t,g) # plot using pyplot library from matplotlib package
    plt.title('Chirp') # plot title
    plt.xlabel('Time (s)') # x-axis label
    plt.ylabel('Amplitude') # y-axis label
    plt.savefig('Ch1_images/Chirp signal.png') # display the figure


# For this fft example, you have option to use fftshift. Comment/uncomment accordingly.
# FFT size should be 2^L and large enough to cover sample size
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
    plt.savefig('Ch1_images/fft_example_1_im1.png') # display the figure

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
    plt.savefig('Ch1_images/fft_example_1_im2.png') # display the figure


# Take FFT, extract amplitude and phase info, re-construct in time domain
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

    plt.figure(0)
    plt.plot(t,x) # plot using pyplot library from matplotlib package
    plt.title('Time domain') # plot title
    plt.xlabel('t (s)') # x-axis label
    plt.ylabel('Amplitude') # y-axis label
    plt.savefig('Ch1_images/fft_example_2_im1.png') # display the figure

    # Calculate fft and extract amplitude info
    N=fs # FFT size
    X = 1/N*fftshift(fft(x,N)) # N-point complex DFT

    df=fs/N # frequency resolution
    sampleIndex = np.arange(start = -N//2,stop = N//2) # // for integer division
    f=sampleIndex*df # x-axis index converted to ordered frequencies

    plt.figure(1)
    plt.stem(f,abs(X)) # plot using pyplot library from matplotlib package
    plt.title('Amplitude vs Frequency') # plot title
    plt.xlabel('f (Hz)') # x-axis label
    plt.ylabel('Amplitude') # y-axis label
    plt.savefig('Ch1_images/fft_example_2_im2.png') # display the figure

    # Extract phase info.
    # Note: Because arctan is sensitive to small ammounts of noise, zero noise below threshold
    X2=X #store the FFT results in another array
    # detect noise (very small numbers (eps)) and ignore them
    threshold = max(abs(X))/10000; # tolerance threshold
    X2[abs(X)<threshold]=0 # maskout values below the threshold
    phase=np.arctan2(np.imag(X2),np.real(X2))*180/np.pi # phase information

    plt.figure(2)
    plt.stem(f,phase) # plot using pyplot library from matplotlib package
    plt.title('Phase vs Frequency') # plot title
    plt.xlabel('f (Hz)') # x-axis label
    plt.ylabel('Phase') # y-axis label
    plt.savefig('Ch1_images/fft_example_2_im3.png') # display the figure

    x_recon = N*ifft(ifftshift(X),N) # reconstructed signal
    t = np.arange(start = 0,stop = len(x_recon))/fs # recompute time index
    
    plt.figure(3)
    plt.plot(t,x_recon) # plot using pyplot library from matplotlib package
    plt.title('Time domain, reconstructed') # plot title
    plt.xlabel('t (s)') # x-axis label
    plt.ylabel('Amplitude') # y-axis label
    plt.savefig('Ch1_images/fft_example_2_im4.png') # display the figure


# Estimate PSD with Welch method
def welch_demo():

    import matplotlib.pyplot as plt
    import numpy as np
    from essentials import plotWelchPSD

    # Create sample signal
    A = 0.5 # amplitude of the cosine wave
    fc=4000 # frequency of the cosine wave in Hz
    phase=30 # desired phase shift of the cosine in degrees
    fs=300*fc # sampling frequency with oversampling factor 32
    t=np.arange(start = 0,stop = 2,step = 1/fs) # 2 seconds duration

    phi = phase*np.pi/180; # convert phase shift in degrees in radians
    x=A*np.cos(2*np.pi*fc*t+phi) # time domain signal with phase shift

    plt.figure(0)
    plt.plot(t,x) # plot using pyplot library from matplotlib package
    plt.title('Time domain') # plot title
    plt.xlabel('t (s)') # x-axis label
    plt.ylabel('Amplitude') # y-axis label
    plt.savefig('Ch1_images/welch_example_im1.png') # display the figure

    plotWelchPSD(x, fs, fc)
welch_demo()

# Compute total power using norm function and verify total power in frequency domain
def power_using_norm():

    import numpy as np
    from numpy.linalg import norm

    A=1 #Amplitude of sine wave
    fc=100 #Frequency of sine wave
    fs=3000 # Sampling frequency - oversampled by the rate of 30
    nCyl=3 # Number of cycles of the sinewave
    t=np.arange(start = 0,stop = nCyl/fc,step = 1/fs) #Time base
    x=-A*np.sin(2*np.pi*fc*t) # Sinusoid

    L = len(x)
    P = (norm(x)**2)/L
    print('Power of the Signal from Time domain {:0.4f}'.format(P))

    from scipy.fftpack import fft, fftshift
    NFFT = L
    X = fftshift(fft(x,NFFT))
    Px = X*np.conj(X)/(L**2) # Power of each frequency component
    print(sum(abs(Px)))


# Compare three convolution methods
def compare_convolutions():

    import numpy as np
    from scipy.fftpack import fft, ifft
    from essentials import my_convolve

    x = np.random.normal(size = 7) + 1j*np.random.normal(size = 7) # normal random complex vectors
    h = np.random.normal(size = 3) + 1j*np.random.normal(size = 3) # normal random complex vectors
    L = len(x) + len(h) - 1 # length of convolution output

    y1 = my_convolve(h,x) # Convolution using Toeplitz matrix
    y2 = ifft(fft(x,L)*(fft(h,L))).T # Convolution using FFT
    y3 = np.convolve(h,x)
    print(f' y1 : {y1} \n y2 : {y2} \n y3 : {y3} \n')


# Take analysic signal from real-valued signal, investigate analytic signal components
# Note: Can also use scipy's hilbert function. Note that the hilbert function full analytic signal, so take imag() to get actual hilbert transformation
def analytic_signal_demo():

    import numpy as np
    import matplotlib.pyplot as plt
    from essentials import analytic_signal

    t = np.arange(0,.5,.001)
    x = np.sin(2*np.pi*10*t) # real-valued f = 10 Hz

    plt.figure(0)
    plt.plot(t,x)
    plt.title('x[n] - real-valued signal')
    plt.xlabel('n')
    plt.ylabel('x[n]')
    plt.savefig('Ch1_images/analytic_signal_demo_im1.png')

    z = analytic_signal(x)

    plt.figure(1)
    plt.plot(t, np.real(z), 'k', label='Real(z[n])')
    plt.plot(t, np.imag(z), 'r', label='Imag(z[n])')
    plt.title('Components of analytic signal')
    plt.xlabel('n')
    plt.ylabel('$z_r[n]$ and $z_i[n]$')
    plt.legend()
    plt.savefig('Ch1_images/analytic_signal_demo_im2.png')


# Demonstrate extraction of instantaneous amplitude and phase from analytic signal constructed from real-valued modulated signal
def extract_envelope_phase():

    import numpy as np
    from scipy.signal import chirp
    import matplotlib.pyplot as plt
    from essentials import analytic_signal

    fs = 600 # sampling frequency Hz
    t = np.arange(0,1,1/fs)
    a_t = 1.0 + .7 * np.sin(2.0*np.pi*3.0*t) # information signal
    c_t = chirp(t, f0=20, t1=t[-1], f1=80, phi=0, method='linear') # index of -1 means last index
    x = a_t * c_t # modulated signal

    z = analytic_signal(x) # form the analytical signal
    inst_amplitude = abs(z) # envelope extraction - calculing mag from complex plane
    inst_phase = np.unwrap(np.angle(z)) # inst phase - arctan
    inst_freq = np.diff(inst_phase)/(2*np.pi)*fs # inst frequency 

    extracted_carrier = np.cos(inst_phase) # Regenerate carrier from instantaneous phase

    plt.figure(0)
    plt.plot(t,x)
    plt.plot(t,inst_amplitude,'r')
    plt.title('Modulated signal and extracted envelope')
    plt.xlabel('n')
    plt.ylabel('x(t) and |z(t)|')
    plt.savefig('Ch1_images/extract_envelope_phase_im1.png')

    plt.figure(1)
    plt.plot(t,extracted_carrier)
    plt.title('Extracted carrier or TFS')
    plt.xlabel('n')
    plt.ylabel('$cos[\omega(t)]$')
    plt.savefig('Ch1_images/extract_envelope_phase_im2.png')


# Demonstrate phase demodulation using Hilbert transformation
def hilbert_phase_demod():

    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.signal import hilbert

    fc = 210 # carrier frequency
    fm = 10 # frequency of modulating signal
    alpha = 1 # amplitude of modulating signal
    theta = np.pi/4 # phase offset of modulating signal
    beta = np.pi/5 # constant carrier phase offset
    # Set true if receiver knows carrier frequency and phase offset
    receiverKnowsCarrier = False

    fs = 8*fc # sampling frequency
    duration = .5 # duration of signal
    t = np.arange(0,duration,1/fs)

    # Phase modulation
    m_t = alpha*np.sin(2*np.pi*fm*t + theta) # modulating signal
    x = np.cos(2*np.pi*fc*t + beta + m_t) # modulated signal

    plt.figure(0)
    plt.plot(t, m_t)
    plt.title('Modulating signal')
    plt.xlabel('t')
    plt.ylabel('m(t)')
    plt.savefig('Ch1_images/hilbert_phase_demod_im1.png')

    plt.figure(1)
    plt.plot(t, x)
    plt.title('Modulated signal')
    plt.xlabel('t')
    plt.ylabel('x(t)')
    plt.savefig('Ch1_images/hilbert_phase_demod_im2.png')

    # Add AWGN noise to the transmitted signal
    mu = 0; sigma = .1 # noise mean and variance
    # Note: for normal function, must specify "size="
    n = mu + sigma*np.random.normal(size=len(t)) # awgn noise
    r = x + n # noisy received signal

    # Demodulation of the noisy Phase Modulated signal
    z = hilbert(r) # form analytical signal from received vector
    inst_phase = np.unwrap(np.angle(z)) # instantaneous phase

    if receiverKnowsCarrier: # If receiver knows the carrier freq/phase perfectly
        offsetTerm = 2*np.pi*fc*t + beta
    else: # else, estimate the subtraction term
        p = np.polyfit(t, inst_phase, 1) # linear fit instantaneous phase
        estimated = np.polyval(p,t)
        offsetTerm = estimated

    demodulated = inst_phase - offsetTerm

    plt.figure(2)
    plt.plot(t, demodulated)
    plt.title('demodulated')
    plt.xlabel('t')
    plt.ylabel('demodulated')
    plt.savefig('Ch1_images/hilbert_phase_demod_im3.png')







