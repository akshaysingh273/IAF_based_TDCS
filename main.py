from pylsl import StreamInlet, resolve_byprop
from scipy.signal import butter, filtfilt, find_peaks
import numpy as np
from psychopy import parallel
import time
import numpy as np
import threading
import pandas as pd
import mne
import os
import keyboard

trial_flag = False   # flag for trials
total_task = 5   # total number of trials

current_trial_count = 0     # current trial count for ref
current_trial_start_time = None
trial_duration = 4  # duration for stim 

data_coll_flag = True   # flag for data collection
data_coll_start_time = time.time()
data_coll_total_time = 31   # time for data collection in sec
freq_calc_flag = False
freq_input_flag = False
freq_input_start_time = None
freq_input_total_time = 30

sfreq = 1000   # Default sampling frequency
eeg_buffer = []
entire_recording = []

channel_names = None

live_buffer_len = sfreq * 30   # 1000 * 30 = 30,000 samples (30 seconds)

start_time = time.time()

filtered_data = []  # buffer of filtered data
new_sample_count = 0

file_name_time = time.ctime(time.time())

eeg_numpy_file_name = f"recorded_eeg_{file_name_time}.npy".replace(" ","_")
eeg_numpy_file_name = eeg_numpy_file_name.replace(":","-")

xlsx_file_name = f"theta_freq_{file_name_time}.xlsx".replace(" ","_")
xlsx_file_name = xlsx_file_name.replace(":","-")

recording = True
# save_path = "output.eeg"  # .eeg is a binary extension here

def stim(duration:int = 90):
    port = parallel.ParallelPort(address = 0x4EFC)
    print("start trigger sent!")
    # start trig
    port.setData(1) 
    time.sleep(0.01)
    port.setData(0)

    time.sleep(duration+30) # duration in sec

    print("end trigger sent!")
    # end trig
    port.setData(2)
    time.sleep(0.01)
    port.setData(0)

    time.sleep(30)       # duration for tdcs reset (30s for tdcs)
    print("stim_delay reset!")

def trigger_procedure():    
    """send trigger only when the motion start not at the peak
    """
    print("Threshold exceeded! Starting procedure...")
    stim(trial_duration*60)  # Call your stim function (true = 90)
    print("Procedure completed.")

def EEG_setup():
    """
    Set up the EEG stream from LiveAmp using LSL.

    Returns: inlet, sfreq, n_channels, channel_names
    """
    print("Looking for a LiveAmp EEG stream...")
    streams = resolve_byprop('type', 'EEG')
    inlet = StreamInlet(streams[0])

    print("Now pulling samples...")

    stream_info = inlet.info()

    sfreq = int(stream_info.nominal_srate())  # typically 500 Hz
    n_channels = stream_info.channel_count()

    channel_names = []
    ch = stream_info.desc().child("channels").child("channel")
    for _ in range(n_channels):
        label = ch.child_value("label")
        channel_names.append(label)
        ch = ch.next_sibling()

    return inlet, sfreq, n_channels, channel_names

def convert_npy_to_set(set_file, sfreq, channel_names):
    data = np.load(eeg_numpy_file_name)         # shape: (n_samples, n_channels)
    data = data.T                    # shape: (n_channels, n_samples)

    info = mne.create_info(ch_names=channel_names, sfreq=sfreq)

    raw = mne.io.RawArray(data, info)

    mne.export.export_raw(raw=raw, fname=set_file)
    return

from scipy.signal import welch

def freq_calculator(filtered_data) -> None:
    global sfreq, current_trial
    
    _, num_samples = np.shape(filtered_data)
    duration_secs_total = num_samples / sfreq  # total time in seconds

    peak_freq = []

    for segment in range(data_coll_total_time):
        start = 1000 * segment
        stop = 1000 * (segment + 1)
        
        if stop > np.transpose(filtered_data).shape[0]:
            break
        
        filtered_data_segment = np.transpose(filtered_data)[start:stop, :]

        ch_IAF = []
        for ch_idx in range(filtered_data_segment.shape[1]):
            channel_data = filtered_data_segment[:, ch_idx]

            # Welch PSD
            freqs, psd = welch(channel_data, fs=sfreq, nperseg=sfreq*1)  # 1-sec windows

            # Restrict to alpha band (8–13 Hz)
            alpha_mask = (freqs >= 8) & (freqs <= 13)
            freqs_alpha = freqs[alpha_mask]
            psd_alpha = psd[alpha_mask]

            if len(freqs_alpha) > 0:
                # IAF = frequency of max power in alpha band
                iaf = freqs_alpha[np.argmax(psd_alpha)]
                ch_IAF.append(iaf)

        if ch_IAF:
            peak_freq.append(np.mean(ch_IAF))  # take strongest channel

    theta_freq = np.mean(peak_freq) - 5
    print(f"\nTheta Freq (Trial {current_trial_count}): {theta_freq:.2f} Hz")

    # ---------------- Save to Excel in wide format ----------------
    # xlsx_file_name = "theta_freq_results.xlsx"
    stim_col = f"STIM {current_trial_count}"

    if os.path.exists(xlsx_file_name):
        df = pd.read_excel(xlsx_file_name)
    else:
        df = pd.DataFrame({"Theta Freq": ["Theta Freq"]})

    if stim_col not in df.columns:
        df[stim_col] = None

    df.loc[0, stim_col] = theta_freq
    df.to_excel(xlsx_file_name, index=False)
    print(f"Results written to column {stim_col} in {xlsx_file_name}")



def bandpass_filter(data, sfreq, lowcut=8, highcut=13):
    global filtered_data, trial_flag
    # print(f"applying bandpass filter!")
    nyquist = 0.5 * sfreq
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(2, [low, high], btype='band')
    filtered_data = filtfilt(b, a, data, axis=-1)

    # code for finding the freq in alpha range
    freq_calculator(filtered_data)

def EEG_Data_acc(inlet):
    global eeg_buffer, new_sample_count, entire_recording, freq_calc_flag
    while time.time() - data_coll_start_time <= data_coll_total_time:
        sample, _ = inlet.pull_chunk()
        if sample:
            # print(f"shape of sample:{np.shape(sample)}")
            for sam in sample:
                # print(f"shape of sam:{np.shape(sam)}")
                eeg_buffer.append(sam)
                entire_recording.append(sam)
                new_sample_count += 1
            # print(f"shape of eeg_buffer:{np.shape(eeg_buffer)}")
            time.sleep(0.001)
        if len(eeg_buffer) >= live_buffer_len:
            del eeg_buffer[:len(eeg_buffer) - live_buffer_len-1]

if __name__ == '__main__':

    inlet, sfreq, n_channels, channel_names = EEG_setup()
    print(f"channel_names:{channel_names}")

    def exit_handler():
        global recording, entire_recording, eeg_numpy_file_name

        # Save when recording ends
        np.save(eeg_numpy_file_name, np.array(entire_recording))  # shape: (n_samples, n_channels)
        convert_npy_to_set(eeg_numpy_file_name.replace(".npy",".set"), sfreq=sfreq, channel_names=channel_names)
    
    while True:

        if current_trial_count < total_task:
            if data_coll_flag:
                print("data collection started!")
                data_coll_start_time = time.time()
                EEG_Data_acc(inlet)
                data_coll_flag = False
                freq_calc_flag = True
                print("data collection complete!")
            elif freq_calc_flag:
                print("freq calc started!")
                print(f"shape of eeg_buffer: {np.transpose(eeg_buffer).shape}")
                bandpass_filter(np.array(eeg_buffer).T, sfreq, 8, 13)
                freq_calc_flag = False
                freq_input_flag = True
                freq_input_start_time = time.time()
                print("freq calc complete!")
                print("freq input started!")
            elif freq_input_flag:
                while time.time() - freq_input_start_time <= freq_input_total_time:
                    pass
                freq_input_flag = False
                trial_flag = True
                current_trial_start_time = time.time()
                print("freq input complete!")
            elif trial_flag:
                trigger_procedure()
                current_trial_count += 1
                trial_flag = False
                data_coll_flag = True
                data_coll_start_time = time.time()
            
        else:
            exit_handler()
            break
        
        if keyboard.is_pressed('q'):
            exit_handler()
            break
