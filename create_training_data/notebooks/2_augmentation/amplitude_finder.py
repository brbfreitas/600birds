from librosa import load
import numpy as np
import os
import time

def window_energy(samples, nperseg = 256, noverlap = 128):
    '''
    Calculate audio energy with a sliding window
    
    Calculate the energy in an array of audio samples
    using a sliding window. Window includes nperseg
    samples per window and each window overlaps by
    noverlap samples. 
    
    Args:
        samples (np.ndarray): array of audio
            samples loaded using librosa.load
        
    
    '''
    def _energy(samples):
        return np.sum(samples**2)/len(samples)
    
    windowed = []
    skip = nperseg - noverlap
    for start in range(0, len(samples), skip):
        window_energy = _energy(samples[start : start + nperseg])
        windowed.append(window_energy)

    return windowed

def identify_silence(
    filename,
    smoothing_factor = 10,
    nperseg = 256,
    noverlap = 128,
    thresh = 0.05
):
    '''
    Identify whether a file is silent
    
    Load samples from an mp3 file and identify
    whether or not it is likely to be silent.
    Silence is determined by finding the energy 
    in windowed regions of these samples, and
    normalizing the detected energy by the average
    energy level in the recording.
    
    If any windowed region has energy above the 
    threshold, returns a 0; else returns 1.
    
    Args:
        filename (str): file to inspect
        smoothing_factor (int): modifier
            to window nperseg
        nperseg: number of samples per window segment
        noverlap: number of samples to overlap
            each window segment
        thresh: threshold value (experimentally
            determined)
        
    Returns:
        1 if file seems silent
        0 if file seems to contain vocalizations 
    '''

    samples, sr = load(filename)
    windowed = window_energy(samples, nperseg*smoothing_factor, noverlap)
    norm_factor = np.mean(windowed)
    ys = (windowed - norm_factor)*100
    return int(np.max(ys) < thresh)

if __name__ == '__main__':
    
    # Create a list of files to analyze
    split_path = '/bgfs/jkitzes/xeno-canto-split/'
    # Create a list of files to analyze
    species_dirs = [os.path.join(split_path, sp) 
                    for sp in os.listdir(split_path)
                    if os.path.isdir(os.path.join(split_path, sp))]
    filenames = []
    for species_dir in species_dirs:
        random.seed(1)
        try:
            filenames.extend(
                [os.path.join(species_dir, mp3_name) 
                for mp3_name in os.listdir(species_dir)]
            )
        except IndexError:
            continue

    # Analyze all files and save to dictionary
    silence_dict = {}
    for filename in filenames:
        silence_dict[filename] = identify_silence(filename)

    # Write results to .csv
    csv = ''
    for filename, value in silence_dict.items():
        line = f"{filename},{value}\n"
        csv += line
    csv_name = '/bgfs/jkitzes/xeno-canto-split/silences.csv'
    with open(csv_name, 'a') as f:
        f.write('filename,silence_detector_v1\n')
        f.write(csv)