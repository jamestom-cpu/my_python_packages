import os
import numpy as np
import pandas as pd

from scipy import interpolate


class Sparam():
    def __init__(self, csvfile, directory="."):
        self.filepath=os.path.join(directory, csvfile)
        self.table=self.import_S_params()
        self.format = "amp"


    def import_S_params(self):
        tmp_table = pd.read_csv(self.filepath)

        rename_dict = {
                    tmp_table.columns[0]: "f",
                    tmp_table.columns[1]: "S11_r",
                    tmp_table.columns[2]: "S11_im",
                    tmp_table.columns[3]: "S21_r",
                    tmp_table.columns[4]: "S21_im",
                    tmp_table.columns[5]: "S31_r",
                    tmp_table.columns[6]: "S31_im",
                }

        tmp_table.rename(rename_dict, inplace=True, axis="columns")

        table = pd.DataFrame(columns=["S11", "S21", "S31"])
        table["S11"]=pd.Series(tmp_table["S11_r"]+1j*tmp_table["S11_im"], dtype="complex64")
        table["S21"]=pd.Series(tmp_table["S21_r"]+1j*tmp_table["S21_im"], dtype="complex64")
        table["S31"]=pd.Series(tmp_table["S31_r"]+1j*tmp_table["S31_im"], dtype="complex64")
        table.set_index(pd.Index(tmp_table["f"]*1e6, dtype="float64",name="f"), inplace=True)
        return table
    
    def resample(self, new_f):
        old_f = self.table.index.to_numpy()

        self.table = self.table.reset_index().drop("f", axis=1).apply(lambda x: resample_function(x, old_f, new_f))
        self.table = self.table.set_index(pd.Index(new_f, dtype="float32", name="f"))
        return self
        

    def to_dB(self):
        self.table = self.table.transform(lambda x: 20*np.log10(x))
        self.format="dB"
        return self

    def to_ampl(self):
        self.table = self.table.transform(lambda x: 10**(x/20))
        self.format="amp"
        return self
        
    def mag(self):
        format = self.format
        if format=="dB":
            self.to_ampl()
            mags = self.table.transform(lambda x: np.abs(x))
            mags = 20*np.log10(mags)
            self.to_dB()
        else:
            mags = self.table.transform(lambda x: np.abs(x))

        return mags

    def phase(self):
        return self.table.transform(lambda x: np.angle(x))




def resample_function(series, old_f, new_f):
    # get the interpolation coefficients
    interpolation_coeff_r = interpolate.splrep(old_f, np.real(series))
    interpolation_coeff_i = interpolate.splrep(old_f, np.imag(series))

    # interpolate at the values of f_sim
    series_inter_r = interpolate.splev(new_f, interpolation_coeff_r)
    series_interp_i = interpolate.splev(new_f, interpolation_coeff_i)

    series_interp = series_inter_r + series_interp_i*1j

    return series_interp