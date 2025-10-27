import pandas as pd
import os
import BIDMC_Preprocess
import BIDMC_FE_freq
import BIDMC_FE_derivative
import BIDMC_FE_statistical
import BIDMC_FE_time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List
import numpy as np
import neurokit2 as nk

def Aggregate_features(input_csv: str, 
                output_dir: str,                
                ) -> None:
    os.makedirs(output_dir, exist_ok=True)
    """
    This function aggreagtes all PPG signals and target SpO2,RR into one .cvs file

    Input:
      - input_csv: address of BIDMC folder
      - output_dir: address of result .csv 
    """

    # version 1 without parallel processing 
    # works, but slow
    # result_csv = []
    # for i in range(1,54):
    #     print(f"Working on file bidmc_{i:02d}_Signals.csv")
    #     file_path = os.path.join(input_csv, f"bidmc_{i:02d}_Signals.csv")
    #     result_csv.append(Aggre_one_wave(file_path))
    # result_df = pd.concat(result_csv, ignore_index=True)

    # version 2 use parallel processing
    wave_nums = [i for i in range(1,54)]
    
    result_df = process_files_parallel(wave_nums, input_csv)

    dir_path = os.path.join(output_dir, "BIDMC_Segmented_features.csv")
    result_df.to_csv(dir_path, index=False)


def process_files_parallel(file_num: List[int], base_add: str) -> pd.DataFrame:
    """
    This function schedules parallel processing

    Input:
      - file_num: a index list of PPG wave files
      - base_add: address of BIDMC folder
      
    Output:
      - Return aggregate dataframe
    """
    with ProcessPoolExecutor(max_workers=12) as executor:  # May change max_workers. 12 is the max my laptop can take
        # record original indexing
        futures = {executor.submit(Aggre_one_wave, file, base_add): idx for idx, file in enumerate(file_num)}
        
        # initialize result list based on original index
        results = [None] * len(file_num)
        
        # fill result based on index
        for future in as_completed(futures):
            idx = futures[future]
            results[idx] = future.result()
    
    return pd.concat(results, ignore_index=True)


def Aggre_one_wave(csv_num: int,
                base_add: str,                 
                time_col='Time [s]',
                value_col=' PLETH',
                length=30.0,
                distance=15.0,
                fs=125) -> pd.DataFrame:
    """
    This function aggregate one wave in to single row with targets and features

    Input:
      - csv_num: the number of wave to aggregate
      - base_add: address of BIDMC folder
      - time_col: name for time col
      - value_col: name for value col
      - length: segment length(s)
      - distance: segement distance(s)
      - fs: sampling frequency

    Output:
      - dataframe of aggregate wave
    """
    
    input_csv_signal = os.path.join(base_add, f"bidmc_{csv_num:02d}_Signals.csv")
    input_csv_numerics = os.path.join(base_add, f"bidmc_{csv_num:02d}_Numerics.csv")

    df = pd.read_csv(input_csv_signal)
    df_num = pd.read_csv(input_csv_numerics)
    
    samples_per_seg = int(length * fs)    
    step_samples = int(distance * fs)    

    total_samples = len(df)
    index = 0
    start = 0

    aggre_R = []                                                    # aggregated result
    while start + samples_per_seg <= total_samples:
        # get desired section
        end = start + samples_per_seg
        curr_wave = df[[time_col, value_col]].iloc[start:end]
        wave_num = os.path.basename(input_csv_signal)[6:8]
        seg_num = index

        # preprocessing
        raw_PLETH = curr_wave[value_col]
        clean_PLETH = BIDMC_Preprocess.VMD_deMA(BIDMC_Preprocess.butter_filter(raw_PLETH))            # preprocessing
   
        segList, idxList = wave_segmentation(clean_PLETH)
        
        #collect features
        FE_stat = BIDMC_FE_statistical.extract_statistical_features(clean_PLETH)
        FE_freq = BIDMC_FE_freq.frequency_features(clean_PLETH)

        try:
            FE_time = BIDMC_FE_time.extract_key_time_domain_features(segList, idxList)
        except ValueError as v:
            print(f"wave {wave_num}, seg {seg_num} no valid beat")
        except IndexError as i:
            print(f"wave {wave_num}, seg {seg_num} index error")

        FE_deriv = BIDMC_FE_derivative.derivative_features(segList)

        # collect target
        FE_targets = get_target(df_num, index)

        # initialize current row and add features 
        curr_row = pd.DataFrame()                                  
        curr_row["wave nunmber"] = [wave_num]
        curr_row["segment nunmber"] = [seg_num]

        # convert dict to DataFrame and concatenate
        features = {**FE_targets, **FE_stat, **FE_freq, **FE_time, **FE_deriv}
        features_df = pd.DataFrame([features])
        curr_row = pd.concat([curr_row, features_df], axis=1)

        aggre_R.append(curr_row)

        index += 1
        start += step_samples

    result_df = pd.concat(aggre_R, ignore_index=True)
    return result_df


def get_target(df,
            idx,
            value_col=[' SpO2', ' RESP'],
            length=30,
            distance=15):
    """
    This function collects average SpO2 and RR value as targets
    """
    
    start = int(idx*distance)
    end = int(start + length)

    SpO2 = df[[value_col[0]]].iloc[start:end]
    RR = df[[value_col[1]]].iloc[start:end]

    target = {"SpO2(mean)" : np.mean(SpO2),
              "RR(mean)" : np.mean(RR)}

    return target


def wave_segmentation(WavePeriod: List[float]) -> tuple[List[List[float]], List[List[float]]]:
    """
    This function is used to segment each 30s period in to heartbeats

    Input:
      - WavePeriod: The period to be segmented

    Output:
      - SegList_Selected: A list of amplitude of each segmented heartbeats 
      - IdxList_Selected: A list of index of each segmented heartbeats
    """

    segments = nk.ppg_segment(WavePeriod, sampling_rate=125, show=False)

    SegList = []
    IdxList = []
    # the last segment is always 0, just ignor it
    for i in range(1,len(segments)):
        # extract signal data
        SegList.append((segments[f"{i}"].Signal.to_list()))
        IdxList.append((segments[f"{i}"].Index.to_list()))

    # calculate average peak magnitude
    PeakList = [np.max(Seg) for Seg in SegList]
    PeaskAve = np.mean(PeakList)

    SegList_Selected = []
    # SegList_Ignor = []
    IdxList_Selected = []
    # IdxList_Ignor = []

    # ignor wave sements whose value are too low
    for i in range(len(PeakList)):

        if PeakList[i] > PeaskAve*0.5:
            SegList_Selected.append(SegList[i])
            IdxList_Selected.append(IdxList[i])

        # else:
        #     SegList_Ignor.append(SegList[i])
        #     IdxList_Ignor.append(IdxList[i])

    return SegList_Selected, IdxList_Selected

def main():
    """
    Run overall aggregation
    """
    wave_path = r"Software/Data/BIDMC/bidmc_csv" # replace with your path
    output_path = r"BIDMC_Regression/features" # replace with your path
    Aggregate_features(wave_path, output_path)

if __name__ == '__main__':
    main()