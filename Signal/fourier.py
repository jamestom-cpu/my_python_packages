import numpy as np
import matplotlib.pyplot as plt
import scipy.fft as sc_fft
import pandas as pd


class Signal:
    
    # group all the elaboration tools in one class. 
    # this class is built to handle the data processing of a single measurement. 

    def __init__(self, observation_time, number_of_samples, signal=None):
        self.T = observation_time
        self.N = number_of_samples

        ##
        self.Ts = self.T / self.N
        self.fs = 1 / self.Ts
        self.f_Nyquist = self.fs / 2
        self.t = np.linspace(0, self.T, self.N)

        # two sided
        self.f2 = np.linspace(-self.f_Nyquist, self.f_Nyquist, int(self.N))

        # single sided
        self.f1 = np.linspace(0, self.f_Nyquist, int((self.N//2)+1))

        if signal is not None:
            self.signal = signal
            # perform fft
            self.raw_spectrum_2, self.raw_spectrum_1 = self.fft(signal, sides=0)

    def fft(self, signal=None, normalize=True, sides=1):
        if signal is None:
            signal = self.signal
        
        if normalize:
            norm="forward"
        else:
            norm="backward"

        fft1 = 2*sc_fft.rfft(signal, norm=norm)
        fft1[0] = fft1[0]/2
        fft2 = sc_fft.fftshift(sc_fft.fft(signal, norm=norm))


        if sides == 1:
            return fft1
        if sides == 2:
            return fft2
        if sides == 0:
            return fft1, fft2
    
    def get_fft_table(self, sides):
        if sides==1:
            f=self.f1
        if sides==2:
            f=self.f2        
        fft = self.fft(sides=sides)

        return pd.DataFrame(np.stack([f, fft], axis=1), columns=["f", "fft"]).astype({"f":"float32", "fft":"complex128"}).set_index("f")



    def apply_conv_FIR_filter(self, signal, filt):
        self.FIR_spectr = self.fft(filt, sides=2)

        #         self.FIR_coherent_gain = np.average(self.FIR_spectr)
        #         self.FIR_coherent_power_gain = np.sqrt(np.average(self.FIR_spectr**2))
        self.filt_signal = np.convolve(filt, signal, "same")

        #         self.filt_signal_amp = self.filt_signal/self.FIR_coherent_gain
        #         self.filt_signal_pwr = self.filt_signal/self.FIR_coherent_power_gain

        return self.filt_signal

    def apply_window(self, signal, window, coherence="amplitude"):
        self.coherent_gain = np.average(window)
        self.coherent_power_gain = np.sqrt(np.average(window ** 2))

        raw_windowed_signal = signal * window

        if coherence == "amplitude":
            windowed_signal = raw_windowed_signal / self.coherent_gain
        elif coherence == "power":
            windowed_signal = raw_windowed_signal / self.coherent_power_gain
        else:
            print("retuning raw window")
            windowed_signal = raw_windowed_signal

        return windowed_signal

    def inspect_filter(self, signal, FIR, window=None, window_coherence="amplitude", ax=None):
        if ax == None:
            fig, ax = plt.subplots(1, 2, figsize=(18, 6), dpi=120)

        ntaps = FIR.shape[0]
        f_FIR = np.linspace(0, self.f_Nyquist, int((ntaps + 1) // 2))
        FIR_fz = self.fft(FIR, sides=2, normalize=False)[-f_FIR.shape[0]:]

        filt = self.apply_conv_FIR_filter(self.signal, FIR)

        ax[0].plot(self.f1, self.fft(signal), label="raw signal")
        ax[0].plot(self.f1, self.fft(filt), label="filtered")
        ax[0].plot(f_FIR, FIR_fz, '-k', label="filter frequency response")

        ax[0].set_xscale("log")
        ax[0].set_xlim([10e6, 2.5e9])
        ax[0].grid(True, which="both")
        ax[0].set_xlabel("f[Hz]")
        ax[0].set_ylabel("Spectrum Magnitude [V]")
        ax[0].set_ylim([0, 1.5])

        ax[1].plot(self.f1, 20 * np.log10(self.fft(signal)), label="raw signal")
        ax[1].plot(self.f1, 20 * np.log10(self.fft(filt)), label="filtered")
        ax[1].plot(f_FIR, 20 * np.log10(FIR_fz), '-k', label="filter frequency response")

        ax[1].set_ylim([-40, 5])
        ax[1].set_ylabel("Spectrum Magnitude[dB]")
        ax[1].set_xscale("log")
        ax[1].set_xlim([10e6, 2.5e9])
        ax[1].grid(True, which="both")
        ax[1].set_xlabel("f[Hz]")

        if window is not None:
            windowed = self.fft(self.apply_window(filt, window, coherence=window_coherence))

            ax[0].plot(self.f1, windowed, label="windowed and filtered")
            ax[1].plot(self.f1, 20 * np.log10(windowed), label="windowed and filtered")

        ax[0].legend()
        ax[1].legend()





def ifft(fft, normalized_input=True, sides=2):

    if normalized_input:
        norm="forward"
    else:
        norm="backward"

    if sides==1:
        my_fft = fft.copy()
        my_fft *= 0.5
        my_fft[0] = 2*my_fft[0]
        return np.real(sc_fft.irfft(my_fft, norm="forward"))

    else:
        return sc_fft.ifft((sc_fft.ifftshift(fft)))
