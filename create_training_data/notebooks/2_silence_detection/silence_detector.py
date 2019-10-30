from librosa import load
import numpy as np
import os
from glob import glob
import ray
from timeit import default_timer as timer
from audioread.exceptions import NoBackendError

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

@ray.remote
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
    try:
        samples, sr = load(filename)
    except (RuntimeError, NoBackendError):
        return -1.0

    windowed = window_energy(samples, nperseg*smoothing_factor, noverlap)
    norm_factor = np.mean(windowed)
    ys = (windowed - norm_factor)*100
    #return filename, int(np.max(ys) < thresh)
    return np.max(ys)

if __name__ == '__main__':
    # Set up Ray w/ Slurm
    try:
        SLURM_SCRATCH = os.environ["SLURM_SCRATCH"]
        num_cpus = int(os.environ["SLURM_JOB_CPUS_PER_NODE"])
        ray.init(num_cpus=num_cpus)
    except KeyError:
        exit("Error: this should be submitted via Slurm")
    
    # Create a list of files to analyze
    split_path = '/bgfs/jkitzes/xeno-canto-split'

    # Create a list of MP3s
    filenames = sorted(glob(f"{split_path}/*/*.mp3"))
    print(f"Number of files: {len(filenames)}")

    # Run silence detector
    start = timer()
    futs = [identify_silence.remote(fname) for fname in filenames]
    with open(f"{SLURM_SCRATCH}/silences.csv", "w") as f:
        f.write("filename,value\n")
        for idx, fut in enumerate(futs):
            f.write(f"{filenames[idx]},{ray.get(fut)}\n")
    end = timer()
    print(f"Run time: {end - start} seconds")