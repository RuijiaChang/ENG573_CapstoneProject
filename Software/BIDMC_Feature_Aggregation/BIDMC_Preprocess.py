import pandas as pd
import os
from scipy import signal
from scipy.fft import fft, fftfreq
import numpy as np
from vmdpy import VMD
import matplotlib.pyplot as plt


def split_wave(input_csv: str, 
                output_dir: str,                
                time_col='Time [s]',
                value_col=' PLETH',
                length=30.0,
                distance=15.0,
                fs=125) ->None :
    """
    This function is used to split wave into 30s sections and store them.
    Mainly used for single period analysis during function implementation and debugging.

    Input:
      - input_csv: str of input .csv address
      - output_dir: str of directory address
      - time_col: name for time col
      - value_col: name for value col
      - length: segment length(s)
      - distance: segement distance(s)
      - fs: sampling frequency
    """

    df = pd.read_csv(input_csv)
    if time_col not in df.columns or value_col not in df.columns:               #check input validity
        raise ValueError(f"'{time_col}'or'{value_col}' does not exist")
    

    samples_per_seg = int(length * fs)    
    step_samples = int(distance * fs)    

    os.makedirs(output_dir, exist_ok=True)

    total_samples = len(df)
    index = 0
    start = 0

    while start + samples_per_seg <= total_samples:
        
        dir_path = os.path.join(output_dir, f"seg_{index}.csv")
        #skipping exsisting file
        if os.path.exists(dir_path):
            print(f"seg_{index} already exists.")
            index += 1
            start += step_samples
            continue

        end = start + samples_per_seg

        df[[time_col, value_col]].iloc[start:end].to_csv(         #to be modified with preprocessing
            dir_path,
            index=False
        )

        index += 1
        start += step_samples

def butter_filter(PLETH: list[float], freq=125, critial_f=25) -> list[float]:
    """
    This function applies Z-score normalization and low pass filter to PLETH

    Input:
      - PLETH: PPG signal
      - freq: sample frequency
      - critial_f: critical frequency for low pass filter.

    Output: 
      - Return The filtered signal
    """
    # PLETH: PLETH col for each seg

    # Z score normalization
    PLETH = (PLETH - np.mean(PLETH))/np.std(PLETH)

    sos = signal.butter(6, critial_f, 'lp', fs=freq, output='sos')
    filtered = signal.sosfiltfilt(sos, PLETH)

    return filtered

def VMD_deMA(raw_signal: list[float], freq=125) -> list[float]:
    """
    Variational mode decomposition method. Used to remove low frequency motion artifacts. 

    Input:
      - raw_signal: raw PPG signal
      - freq: sample frequency

    Ourpur:
      - return filtered signal
    """
    alpha = 2000        # moderate bandwidth constraint  
    tau = 0.0           # noise-tolerance (no strict fidelity enforcement)
    K = 6               # 6 modes 
    DC = 0              # no DC part imposed
    init = 1            # initialize omegas uniformly
    tol = 1e-7    

    u, u_hat, omega = VMD(raw_signal, alpha, tau, K, DC, init, tol)

    # check all modes and only accept freq>.6
    valid_mod = []
    freq_result = omega[-1] * freq
    for i in range(K):
        if freq_result[i] > 0.6:
            valid_mod.append(i)

    return np.sum(u[valid_mod],axis=0)

def wave_visualize(t: list[float], PLETH: list[float], title: str, duration=5, frequency=125, f0=-1, f1=15):
    """
    This function is used to visualize both time and frequency domain.
    """

    #plot in time domain
    plt.figure()
    plt.plot(t, PLETH)
    plt.xlabel('time')
    plt.ylabel('PLETH')
    plt.title(title)
    plt.xlim(min(t), min(t)+duration)
    plt.show()

    #plot in frequency domain
    N = len(PLETH)
    T = float(1/frequency)
    yf = fft(PLETH)
    xf = fftfreq(N, T)[:N//2]

    plt.figure()
    plt.plot(xf, 2.0/N * np.abs(yf[0:N//2]))
    plt.grid()
    plt.xlabel('frequency')
    plt.ylabel('PLETH')
    plt.title(title)
    plt.xlim(f0, f1)
    plt.show()

def compare_in_t(t: list[float], sig1: list[float], title1: str, sig2: list[float], title2: str, duration=5):
    """
    This function is used to compare two waves in time domian.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    ax1.plot(t, sig1)
    ax1.set_title(title1)
    # ax1.axis([0, 5, 0.35, 0.65])
    ax2.plot(t, sig2)
    ax2.set_title(title2)
    # ax2.axis([0, 5, -0.1, 0.2])
    ax2.set_xlabel('Time [s]')
    plt.xlim(min(t), min(t)+duration)
    plt.tight_layout()
    plt.show()

    plt.plot(t, sig1, label=title1, alpha=0.5)
    plt.plot(t, sig2, label=title2)
    plt.title('Signal Comparison')
    plt.xlim(min(t), min(t)+duration)
    plt.legend()
    plt.tight_layout()
    plt.show()

