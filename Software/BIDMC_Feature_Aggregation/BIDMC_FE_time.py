import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

def extract_key_time_domain_features(ppg_signal: list[list[float]],index_signal: list[list[int]], sampling_rate: int=125):
    """
    Extract key time-domain features from a PPG signal.
    Input:
      - ppg_signal: list of heartbeats magnitude
      - index_signal: list of heartbeats indexes of original signals
    Returns a dictionary with:
      - sys_amp: Systolic peak amplitude (mean)
      - foot_amp: Foot amplitude (mean)
      - x: Amplitude difference (sys_amp - foot_amp)
      - t1: Time of systolic peak (mean, in seconds)
      - tpi: Pulse interval (time between successive foot points, mean, in seconds)
      - tpp: Peak-to-peak interval (mean, in seconds)
      - t1/x: Ratio of t1 to systolic amplitude
      - t1/tpi: Ratio of t1 to pulse interval
      - x/(tpi-t1): Ratio of amplitude difference to (pulse interval minus t1)
      - A1: Pulse area from the foot to the peak (mean)
      - A2: Pulse area from the peak to the next foot (mean)
      - A1/A2: Ratio of pulse areas
      - W_25, W_50, W_75: Waveform widths at 25%, 50%, and 75% of the amplitude difference
    """
    valid_beats = []

    for i in range(len(ppg_signal)):
        Seg = ppg_signal[i]
        # Detect peaks (systolic points) and foots (valleys)
        peaks, _ = find_peaks(Seg, distance=sampling_rate//2)
        foots, _ = find_peaks(-np.array(Seg), distance=sampling_rate//2)
    
        # Must have at least two foots
        # if len(foots) < 2:
            # raise ValueError("Not enough foot points detected.")

    
        # Match each beat: find for each peak the nearest previous foot and the next foot
        # should only consider the peak with max amplitude.
        peak = max(peaks, key=lambda x: Seg[x])
        prev_foots = foots[foots < peak]
        next_foots = foots[foots > peak]

        if len(prev_foots) == 0:
            prev_foots = np.append(prev_foots, 0)
        if len(next_foots) == 0:
            next_foots = np.append(next_foots, len(Seg)-1)

        foot_before = prev_foots[-1]
        foot_after = next_foots[0]
        # PlotSeg(Seg, [peak], [foot_before, foot_after])

        valid_beats.append((i, foot_before, peak, foot_after))
    
    if len(valid_beats) == 0:
        raise ValueError("No valid beats found.")
    
    # Lists to store per-beat features
    sys_amps = []
    foot_amps = []
    xs = []
    t1s = []
    tpis = []
    A1_list = []
    A2_list = []
    W25_list = []
    W50_list = []
    W75_list = []
    t1s_actual = []
    
    # Helper function to compute width at a given threshold via linear search
    def compute_width(signal, start, end, threshold, sr):
        segment = np.array(signal[start:end+1])
        indices = np.where(segment >= threshold)[0]
        if len(indices) == 0:
            return 0
        return (indices[-1] - indices[0]) / sr
    
    for beat in valid_beats:
        Seg_idx, foot_before, peak_idx, foot_after = beat
        Seg = ppg_signal[Seg_idx]
        # Amplitudes
        sys_amp = Seg[peak_idx]
        foot_amp_val = Seg[foot_before]
        diff_amp = sys_amp - foot_amp_val
        
        # Times (in seconds)
        t1_val = peak_idx / sampling_rate
        t1_actual = index_signal[Seg_idx][peak_idx] / sampling_rate
        tpi_val = (foot_after - foot_before) / sampling_rate
        
        # Pulse areas using numerical integration (trapezoidal rule)
        A1 = np.trapz(Seg[foot_before:peak_idx+1])   # From foot to peak
        A2 = np.trapz(Seg[peak_idx:foot_after+1])      # From peak to next foot
        
        # Compute widths: thresholds at 25%, 50%, and 75% of the amplitude difference above the foot level
        thresh25 = foot_amp_val + 0.25 * diff_amp
        thresh50 = foot_amp_val + 0.50 * diff_amp
        thresh75 = foot_amp_val + 0.75 * diff_amp
        
        W25 = compute_width(Seg, foot_before, foot_after, thresh25, sampling_rate)
        W50 = compute_width(Seg, foot_before, foot_after, thresh50, sampling_rate)
        W75 = compute_width(Seg, foot_before, foot_after, thresh75, sampling_rate)
        
        sys_amps.append(sys_amp)
        foot_amps.append(foot_amp_val)
        xs.append(diff_amp)
        t1s.append(t1_val)
        tpis.append(tpi_val)
        A1_list.append(A1)
        A2_list.append(A2)
        W25_list.append(W25)
        W50_list.append(W50)
        W75_list.append(W75)
        t1s_actual.append(t1_actual)
    
    # Compute peak-to-peak interval (tpp) as difference between consecutive systolic peak times (t1s)
    if len(t1s_actual) > 1:
        tpps = np.diff(t1s_actual)
    else:
        tpps = [0]
    
    # Compute per-beat ratios (for each valid beat)
    t1_div_x = [t1s[i] / sys_amps[i] if sys_amps[i] != 0 else 0 for i in range(len(sys_amps))]
    t1_div_tpi = [t1s[i] / tpis[i] if tpis[i] != 0 else 0 for i in range(len(tpis))]
    x_div_diff = [xs[i] / (tpis[i] - t1s[i]) if (tpis[i] - t1s[i]) != 0 else 0 for i in range(len(xs))]
    
    # generate A1/A2 list. if A2[i]==0, set A1/A2[2] = 0
    A1_array = np.array(A1_list)
    A2_array = np.array(A2_list)
    A1_div_A2 = np.divide(A1_array, A2_array, out=np.zeros_like(A1_array), where=(A2_array != 0))

    # Aggregate by computing the mean of each feature over all beats
    features = {
        "sys_amp": sys_amps,
        "foot_amp": foot_amps,
        "x": xs,
        "t1": t1s,
        "tpi": tpis,
        "tpp": tpps if len(tpps) > 0 else 0,
        "t1/x": t1_div_x,
        "t1/tpi": t1_div_tpi,
        "x/(tpi-t1)": x_div_diff,
        "A1": A1_list,
        "A2": A2_list,
        "A1/A2": A1_div_A2,
        "W_25": W25_list,
        "W_50": W50_list,
        "W_75": W75_list
    }
    
    methods = {
        "mean": np.mean,
        "std": np.std,
        "var": np.var,
    }

    features_product = {}

    for featureName, featureData in features.items():
        for methodName, methodFunc in methods.items():
            key = f"{methodName}({featureName})"
            features_product[key] = methodFunc(featureData)

    return features_product


def PlotSeg(PLETH_Seg: list[float], fs: int=125, peaks: list[int]=[], foots: list[int]=[]):
    """
    Helper function, used to plot each heartbeats with foots and peaks attached.
    """
    N = len(PLETH_Seg)
    T = (N - 1) / fs

    t_Seg = np.linspace(0, T, N)

    plt.figure()
    plt.plot(t_Seg, PLETH_Seg)

    for peak in peaks:
        plt.plot(peak/fs, PLETH_Seg[peak], 'ro')

    for foot in foots:
        plt.plot(foot/fs, PLETH_Seg[foot], 'bo')
        
    plt.xlabel('time')
    plt.ylabel('PLETH')
    plt.show()