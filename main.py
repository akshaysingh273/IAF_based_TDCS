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
data_coll_total_time = 30   # time for data collection in sec
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
    print("Threshold exceeded! Starting procedure...")
    stim(4*60)  # Call your stim function (true = 90)
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

def freq_calculator(filtered_data) -> None:
    global sfreq, channel_names, freq_input_flag, freq_calc_flag
    
    _ , num_samples = np.shape(filtered_data)
    duration_secs = num_samples / sfreq  # total time in seconds

    peak_freq = []

    os.system("cls")

    for segment in range(data_coll_total_time):  # 30 segments of 1 sec each
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
    sample, _ = inlet.pull_sample()
    if sample:
        eeg_buffer.append(sample)
        entire_recording.append(sample)
        new_sample_count += 1
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
                while time.time() - data_coll_start_time <= data_coll_total_time:
                    EEG_Data_acc(inlet)
                data_coll_flag = False
                freq_calc_flag = True
            elif freq_calc_flag:
                bandpass_filter(np.array(eeg_buffer).T, sfreq, 8, 13)
                freq_calc_flag = False
                freq_input_flag = True
                freq_input_start_time = time.time()
            elif freq_input_flag:
                while not time.time() - freq_input_start_time <= freq_input_total_time:
                    freq_input_flag = False
                    trial_flag = True
                    current_trial_start_time = time.time()
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
