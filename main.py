from pylsl import StreamInlet, resolve_byprop
from scipy.signal import butter, filtfilt
import pyqtgraph as pg
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets
from psychopy import parallel
import struct
import time
import sys
import threading
import numpy as np
import threading
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QScrollArea, QSizePolicy, QLineEdit, QPushButton, QLabel, QHBoxLayout
from pyqtgraph.Qt import QtCore
from PyQt6.QtCore import QTimer
from scipy.signal import find_peaks
import pandas as pd
import mne
import os
import keyboard


recording = True  # Set to False to stop recording
baseline_flag = True  # flag for baseline
baseline_duration = 3   # duration of baseline in min (True value = 10)

trial_flag = False   # flag for trials
total_task = 5   # total number of trials

current_trial_count = 0     # current trial count for ref
current_trial_start_time = None
trial_duration = 4  # duration for stim 

freq_calc_flag = True
freq_calc_start_time = time.time()
freq_calc_timer = 30        # duration in sec
freq_input_flag = False
freq_input_start_time = None
freq_input_timer = 30

sfreq = 1000   # Default sampling frequency
eeg_buffer = []
entire_recording = []

channel_names = None

live_buffer_len = sfreq * 30   # 1000 * 30 = 30,000 samples (30 seconds)

start_time = time.time()

filtered_data = []  # buffer of filtered data
new_sample_count = 0

eeg_numpy_file_name = f"recorded_eeg_{time.ctime(time.time())}.npy".replace(" ","_")
eeg_numpy_file_name = eeg_numpy_file_name.replace(":","-")

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
    global trial_flag, freq_calc_flag
    print("Threshold exceeded! Starting procedure...")
    stim(4*60)  # Call your stim function (true = 90)
    print("Procedure completed.")

    trial_flag = False
    freq_calc_flag = True

# def threading_function():
#     num_points = 10  # Number of points to read from the board
#     while True:
#         if board_shim.get_board_data_count() >= num_points:
#             data = board_shim.get_board_data(num_points)
#             accel_data = data[24]
#             y = np.mean(accel_data, axis=0)   

#             emg_buffer["Time"].append(str(time.time()))
#             emg_buffer["EMG_1"].append(np.mean(data[0], axis = 0))
#             emg_buffer["EMG_2"].append(np.mean(data[1], axis = 0))
#             emg_buffer["EMG_3"].append(np.mean(data[2], axis = 0))
#             emg_buffer["EMG_4"].append(np.mean(data[3], axis = 0))
#             emg_buffer["EMG_5"].append(np.mean(data[4], axis = 0))
#             emg_buffer["EMG_6"].append(np.mean(data[5], axis = 0))
#             emg_buffer["EMG_7"].append(np.mean(data[6], axis = 0))
#             emg_buffer["EMG_8"].append(np.mean(data[7], axis = 0))
#             emg_buffer["ACCEL_X"].append(np.mean(data[20], axis = 0))
#             emg_buffer["ACCEL_Y"].append(np.mean(data[21], axis = 0))
#             emg_buffer["ACCEL_Z"].append(np.mean(data[22], axis = 0))
#             emg_buffer["GYRO_X"].append(np.mean(data[23], axis = 0))
#             emg_buffer["GYRO_Y"].append(np.mean(data[24], axis = 0))
#             emg_buffer["GYRO_Z"].append(np.mean(data[25], axis = 0))

#             # y = accel_data[-1]              
#             # print(f"gyro_data: {y}")
#             data_buffer.append(y)
#         time.sleep(0.010)  # Adjust sleep time as needed
#         # else:
#         #     board_shim.release_all_sessions()
#         #     break

def EEG_setup():
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


# def saving_emg(file_path):
#     print("Saving EMG data!")
#     df = pd.DataFrame(emg_buffer, columns=emg_buffer.keys())
#     df.to_csv(file_path)

def convert_npy_to_set(set_file, sfreq, channel_names):
    data = np.load(eeg_numpy_file_name)         # shape: (n_samples, n_channels)
    data = data.T                    # shape: (n_channels, n_samples)

    info = mne.create_info(ch_names=channel_names, sfreq=sfreq)

    raw = mne.io.RawArray(data, info)

    mne.export.export_raw(raw=raw, fname=set_file)
    return

def freq_calculator(filtered_data) -> None:
    global sfreq, channel_names, freq_input_flag, freq_calc_flag
    
    _ , num_samples = np.shape(filtered_data)
    duration_secs = num_samples / sfreq  # total time in seconds

    peak_freq = []

    os.system("cls")

    for segment in range(freq_calc_timer):
        filtered_data_segment = [np.transpose(filtered_data)[1000*segment:1000*segment+1]]
        ch_IAF = []
        for _, channel_data in enumerate(filtered_data_segment):
            # Find peaks
            peaks, _ = find_peaks(channel_data)
            
            # Peak frequency in Hz = number of peaks / duration
            ch_IAF.append(len(peaks) / duration_secs)

        peak_freq.append(max(ch_IAF))
        print(f"\nmax Individial Alpha Freq for segment-{segment+1}: {(peak_freq[-1])}")
    print(f"\nTheta Freq: {np.mean(peak_freq)-5}")
    freq_calc_flag = False
    freq_input_flag = False

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
    global eeg_buffer, new_sample_count, entire_recording, freq_calc_timer
    while recording:
        sample, _ = inlet.pull_sample()
        # print(f"sample_shape: {np.shape(sample)}")
        if sample:
            eeg_buffer.append(sample)
            entire_recording.append(sample)
            # print(f"shape of eeg_buffer: {np.shape(eeg_buffer)}")
            # print(f"len of eeg_buffer: {len(eeg_buffer)}")
            new_sample_count += 1
        if len(eeg_buffer) >= live_buffer_len:
            # print(f"condition_checking:{new_sample_count >= sfreq*1}")
            if freq_input_flag:
                threading.Thread(target=bandpass_filter, args = [np.array(eeg_buffer).T, sfreq, 8, 13]).start()  # Update plots in the main thread
                new_sample_count = 0
            del eeg_buffer[:len(eeg_buffer) - live_buffer_len-1]
            # print(f"buffer_len:{len(eeg_buffer)}\nlive_reuffer_len:{live_buffer_len}")

if __name__ == '__main__':

    trial_timer = 0

    inlet, sfreq, n_channels, channel_names = EEG_setup()
    print(f"channel_names:{channel_names}")

    eeg_data_acc_thread = threading.Thread(target=EEG_Data_acc, args=(inlet,), daemon=True)
    eeg_data_acc_thread.start()

    # print(f"baseline:{baseline_flag}")
    # baseline_thread = threading.Thread(target=baseline_check, daemon=True)
    # baseline_thread.start()

    def exit_handler():
        global recording, entire_recording, eeg_numpy_file_name
        recording = False

        # Save when recording ends
        np.save(eeg_numpy_file_name, np.array(entire_recording))  # shape: (n_samples, n_channels)

        eeg_data_acc_thread.join(timeout=1)
        # baseline_thread.join(timeout=1)

        convert_npy_to_set(eeg_numpy_file_name.replace(".npy",".set"), sfreq=sfreq, channel_names=channel_names)
    
    while True:

        if current_trial_count < total_task:
            if not trial_flag:
                if freq_calc_flag:
                    print(f"freq_calc_flag:{freq_calc_flag}")
                    if not (time.time() - freq_calc_start_time <= freq_calc_timer):
                        print("now calculating freq!")
                        freq_input_flag = True
                        freq_input_start_time = time.time()
                if freq_input_flag:
                    print(f"freq_input_flag:{freq_input_flag}")
                    if (time.time() - freq_input_start_time <= freq_input_timer):
                        freq_calc_flag = False
                    else:
                        print("freq_input now over. Starting trial")
                        freq_calc_flag = False
                        freq_input_flag = False
                        current_trial_start_time = time.time()
                        trial_flag = True
                        print(f"trial_flag:{trial_flag}")
                        threading.Thread(target=(trigger_procedure), daemon=True).start()
        
        else:
            exit_handler()
            break
        
        if keyboard.is_pressed('q'):
            exit_handler()
            break
        time.sleep(0.01)  # Reduce CPU usage
