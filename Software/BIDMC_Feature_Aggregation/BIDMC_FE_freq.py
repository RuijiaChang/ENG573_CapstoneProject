from scipy.fft import fft, fftfreq
import numpy as np
from librosa.feature import mfcc

def frequency_features(PLETH: list[float], freq=125, delf=0.25) -> dict[float]:
    """
    Extract frequency domian features from preprocessed PPG signal.
    
    Input:
      - PLETH: preprocessed PPG signal
      - freq: sampling frequency
      - delf: peak frequency radius

    Output:
      - dict of frequency domain features
    """
    
    xf, yf = performe_fft(PLETH)

    # get results
    delN = int(30*delf)
    Maxfreq, Maxval, Maxratio = find_max_ratio(xf, yf, delN)

    # get mfcc results
    signal_length = len(PLETH)
    mfccs_24 = mfcc(y=np.array(PLETH), sr=freq, n_mfcc=24, n_fft = signal_length, hop_length = signal_length, center=False)

    FE = {
        "Maxfreq"   :Maxfreq,
        "Maxval"    :Maxval,
        "Maxratio"  :Maxratio
    }
    
    for i in range(24):
        FE[f"mfcc_{i}"] = mfccs_24[i][0]

    return  FE


def performe_fft(PLETH, freq=125):
    """conventional fft"""

    N = len(PLETH)
    T = float(1/freq)

    yf = 2.0/N * np.abs(fft(PLETH)[0:N//2])
    yf[0] /= 2
    xf = fftfreq(N, T)[:N//2]

    return xf, yf

def find_max_ratio(xf, yf, delN=15):
    """Find max value, its frequency and peak to total energy ratio """

    # find max amplitude, freq and its index
    max_index = np.argmax(yf)
    Maxfreq = xf[max_index]
    Maxval = yf[max_index]

    # compute energy ratio
    peak_E = np.sum(yf[max(0,max_index-delN):max_index+delN])
    total_E = np.sum(yf)
    Maxratio = peak_E/total_E

    return Maxfreq, Maxval, Maxratio


# testing code
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import pandas as pd
    import librosa

    path = r"D:\桌面\ENG573\data\BIDMC_Denoised\04\seg_0_d.csv"

    df = pd.read_csv(path)
    t = df['Time [s]']
    PLETH = df[' PLETH']
    freq = 125

    xf, yf = performe_fft(PLETH)

    FE = frequency_features(PLETH)

    Maxfreq = FE["Maxfreq"]
    Maxval = FE["Maxval"]
    Maxratio = FE["Maxratio"]

    plt.plot(xf,yf)
    plt.plot(Maxfreq,Maxval, 'ro')
    plt.plot(Maxfreq+0.25, Maxval, 'ro')
    plt.xlim(-1, 5)
    plt.show()
    print(Maxratio)

    print(FE)