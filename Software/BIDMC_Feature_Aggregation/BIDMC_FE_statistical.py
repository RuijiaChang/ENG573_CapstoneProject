import numpy as np
import scipy.stats as stats
import scipy.signal as signal



def extract_statistical_features(ppg_signal):
    """
    Extract statistical features from a PPG signal.
    """
    # Basic statistical features
    mean_val = np.mean(ppg_signal)
    median_val = np.median(ppg_signal)
    std_val = np.std(ppg_signal)
    mad_val = np.mean(np.abs(ppg_signal - mean_val))
    percentile_25 = np.percentile(ppg_signal, 25)
    percentile_75 = np.percentile(ppg_signal, 75)
    iqr_val = percentile_75 - percentile_25
    skewness = stats.skew(ppg_signal)
    kurtosis = stats.kurtosis(ppg_signal)

    # Shannon entropy
    prob_distribution = np.histogram(ppg_signal, bins=100, density=True)[0]
    prob_distribution = prob_distribution[prob_distribution > 0]  # Remove zero values
    shannon_entropy = -np.sum(prob_distribution * np.log2(prob_distribution))

    # Spectral entropy
    power_spectrum = np.abs(np.fft.fft(ppg_signal)) ** 2
    power_spectrum /= np.sum(power_spectrum)  # Normalize power spectrum
    power_spectrum = power_spectrum[power_spectrum > 0]  # Remove zero values
    spectral_entropy = -np.sum(power_spectrum * np.log2(power_spectrum))

    # Store results in dictionary
    features = {
        "Mean": mean_val,
        "Median": median_val,
        "StandardDeviation": std_val,
        "MeanAbsoluteDeviation": mad_val,
        "Percentile_25": percentile_25,
        "Percentile_75": percentile_75,
        "InterquartileRange": iqr_val,
        "Skewness": skewness,
        "Kurtosis": kurtosis,
        "ShannonEntropy": shannon_entropy,
        "SpectralEntropy": spectral_entropy
    }

    return features
