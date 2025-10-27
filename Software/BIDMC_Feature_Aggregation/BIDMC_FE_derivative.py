import pandas as pd
import numpy as np

def calc_derivative(values, td=0.008):
    """
    approximating derivative 
    """
    first_derivatives = np.diff(values) / td

    return first_derivatives


def find_first_maximum(values, td=0.008):

    max_idx = np.argmax(values)  # Find the max value
    max_val = values[max_idx]
    max_time = max_idx*td

    return max_time, max_val  # (time, value)

def find_first_minimum(values, td=0.008):

    min_idx = np.argmin(values)  # Find the min value
    min_val = values[min_idx]
    min_time = min_idx*td
    
    return min_time, min_val  # (time, value)

def derivative_features(Segments :list[list[float]]) -> dict[float]:
    """
    Extract derivative domian features.

    Input:
      - Segments: Heartbeats with magnitude

    Output:
      - dict of aggregated derivative domain features

    """
    v1List = []
    tv1List = []
    v2List = []
    tv2List = []
    a1List = []
    ta1List = []
    a2List = []
    ta2List = []
    v2_v1List = []
    a2_a1List = []
    tv1_tv2List = []
    ta1_ta2List = []
    tv1_ta1List = []
    tv1_ta2List = []
    tv2_ta1List = []
    tv2_ta2List = []

    for segment in Segments:

        df1s = calc_derivative(segment)
        df2s = calc_derivative(df1s)
    
        # Extract features from first derivative (df1s)
        tv1, v1 = find_first_maximum(df1s)  # First max peak from 1st derivative
                                            # Time of first max peak

        tv2, v2 = find_first_minimum(df1s)  # First min peak from 1st derivative
                                            # Time of first min peak

        # Extract features from second derivative (df2s)
        ta1, a1 = find_first_maximum(df2s)  # First max peak from 2nd derivative
                                            # Time of first max peak in 2nd derivative

        ta2, a2 = find_first_minimum(df2s)  # First min peak from 2nd derivative
                                            # Time of first min peak in 2nd derivative
        
        v2_v1 = 0 if v1 == 0 else v2/v1
        a2_a1 = 0 if a1 == 0 else a2/a1
        tv1_tv2 = 0 if tv2 == 0 else tv1/tv2
        ta1_ta2 = 0 if ta2 == 0 else ta1/ta2
        tv1_ta1 = 0 if ta1 == 0 else tv1/ta1
        tv1_ta2 = 0 if ta2 == 0 else tv1/ta2
        tv2_ta1 = 0 if ta1 == 0 else tv2/ta1
        tv2_ta2 = 0 if ta2 == 0 else tv2/ta2

        v1List.append(v1)
        tv1List.append(tv1)
        v2List.append(v2)
        tv2List.append(tv2)
        a1List.append(a1)
        ta1List.append(ta1)
        a2List.append(a2)
        ta2List.append(ta2)
        v2_v1List.append(v2_v1)
        a2_a1List.append(a2_a1)
        tv1_tv2List.append(tv1_tv2)
        ta1_ta2List.append(ta1_ta2)
        tv1_ta1List.append(tv1_ta1)
        tv1_ta2List.append(tv1_ta2)
        tv2_ta1List.append(tv2_ta1)
        tv2_ta2List.append(tv2_ta2)


    # Store the extracted features in a dictionary
    features = {
        "mean(v1)": np.mean(v1List),
        "std(v1)": np.std(v1List),
        "var(v1)": np.var(v1List),

        "mean(tv1)": np.mean(tv1List),
        "std(tv1)": np.std(tv1List),
        "var(tv1)": np.var(tv1List),

        "mean(v2)": np.mean(v2List),
        "std(v2)": np.std(v2List),
        "var(v2)": np.var(v2List),

        "mean(tv2)": np.mean(tv2List),
        "std(tv2)": np.std(tv2List),
        "var(tv2)": np.var(tv2List),

        "mean(a1)": np.mean(a1List),
        "std(a1)": np.std(a1List),
        "var(a1)": np.var(a1List),

        "mean(ta1)": np.mean(ta1List),
        "std(ta1)": np.std(ta1List),
        "var(ta1)": np.var(ta1List),


        "mean(a2)": np.mean(a2List),
        "std(a2)": np.std(a2List),
        "var(a2)": np.var(a2List),


        "mean(ta2)": np.mean(ta2List),
        "std(ta2)": np.std(ta2List),
        "var(ta2)": np.var(ta2List),

        "mean(v2/v1)": np.mean(v2_v1List),
        "std(v2/v1)": np.std(v2_v1List),
        "var(v2/v1)": np.var(v2_v1List),

        "mean(a2/a1)": np.mean(a2_a1List),
        "std(a2/a1)": np.std(a2_a1List),
        "var(a2/a1)": np.var(a2_a1List),

        "mean(tv1/tv2)": np.mean(tv1_tv2List),
        "std(tv1/tv2)": np.std(tv1_tv2List),
        "var(tv1/tv2)": np.var(tv1_tv2List),

        "mean(ta1/ta2)": np.mean(ta1_ta2List),
        "std(ta1/ta2)": np.std(ta1_ta2List),
        "var(ta1/ta2)": np.var(ta1_ta2List),

        "mean(tv1/ta1)": np.mean(tv1_ta1List),
        "std(tv1/ta1)": np.std(tv1_ta1List),
        "var(tv1/ta1)": np.var(tv1_ta1List),

        "mean(tv1/ta2)": np.mean(tv1_ta2List),
        "std(tv1/ta2)": np.std(tv1_ta2List),
        "var(tv1/ta2)": np.var(tv1_ta2List),

        "mean(tv2/ta1)": np.mean(tv2_ta1List),
        "std(tv2/ta1)": np.std(tv2_ta1List),
        "var(tv2/ta1)": np.var(tv2_ta1List),

        "mean(tv2/ta2)": np.mean(tv2_ta2List),
        "std(tv2/ta2)": np.std(tv2_ta2List),
        "var(tv2/ta2)": np.var(tv2_ta2List)
    }

    return features