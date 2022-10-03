import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Signal import fourier, from_HFSS





## assumption the input signal consists of N samples evenly spaced over a period T

def get_output_signal_td(input_signal, sparamcsvfile, Sparam_dir='.', plot=False, return_all = False):
     """ the input signal should be a pandas table with time values as index - in seconds."""
     """ there should only be one signal insider the table as input """

     #get information on the time
     input_time=input_signal.index
     N=len(input_time)
     T=list(input_time)[-1]

     # get input signal spectrum
     fourier_signal = fourier.Signal(
          observation_time=T,
          number_of_samples=N,
          signal=input_signal.squeeze().to_numpy()
     )

     fft_table = fourier_signal.get_fft_table(1)

     # get Sparameters from a csvfile
     sprm = from_HFSS.Sparam(csvfile=sparamcsvfile, directory=Sparam_dir)

     # resample
     in_range_input_spectrum = fft_table.loc[sprm.table.index.min():sprm.table.index.max()]
     sprm.resample(in_range_input_spectrum.index.to_numpy())

     # # plot characteristics
     if plot:
          plot_signal_properties_compare_with_S_params(input_signal, fft_table, sprm)

     # find the spectrum at the output
     # find the input components at the edge of the probe output

     data = np.array([
               fft_table.multiply(sprm.table[S], axis="index")
               .astype("complex128")
               .fillna(0)
               .to_numpy()
               .squeeze() 
               for S in sprm.table.columns
               ]).T
     
     output_fft = pd.DataFrame(
          columns=sprm.table.columns, 
          dtype="complex128", 
          index=fft_table.index,
          data= data
          )

     output_signals = \
     output_fft.copy().reset_index() \
          .drop(columns="f") \
          .apply(lambda x: fourier.ifft(x.to_numpy(), sides=1)) \
          .set_index(input_signal.index) \
          .add_prefix("signal_over_")


     if return_all:
        return dict(input_fft_table=fft_table, S=sprm, output_fft_table=output_fft, output_signals=output_signals)

     return output_signals

def plot_signal_properties_compare_with_S_params(input_signal, fft_table, sprm):
    f = plt.figure(figsize=(16,8))
    ax = []
    ax.append(f.add_subplot(221))
    input_signal.plot(legend=False, ax=ax[-1])
    ax[-1].set_xlabel("t[s]")
    ax[-1].set_ylabel("Amp[V]")
    ax[-1].set_title("Input Signal")

    # fft_table_in_range = fft_table.copy().loc[sprm.table.index.min():sprm.table.index.max()]

    ax.append(f.add_subplot(222))
    ax[-1].stem(fft_table.index, fft_table.apply(abs));
    ax[-1].set_xscale("log")
    ax[-1].grid()
    ax[-1].set_xlabel("f[Hz]")
    ax[-1].set_ylabel("Spectrum[V]")
    ax[-1].set_ylim([0,1.1])
    ax[-1].set_title("Input Signal Spectrum")

    ax = f.add_subplot(212)
    sprm.mag().plot(ax=ax, logx=True, grid=True)
    ax.stem(fft_table.index, fft_table.apply(abs), label="Input Spectrum")
    ax.set_ylim([0,1.1])
    ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
    ax.set_title("Compare Input Spectrum With S parameters")

def plot_timedomain_results(input_signal, output_signals, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(12,5))
    output_signals.plot(ax=ax)
    input_signal.rename(columns={"value":"input_signal"}).plot(ax=ax)
    ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
    ax.grid()