"""
Some modifications
# Original copyright: https://github.com/microsoft/MS-SNSD (author: chkarada)
"""
import glob
import numpy as np
import soundfile as sf
import os
import argparse
import configparser as CP
from pathlib import Path
from audiolib import audioread, audiowrite, snr_mixer

def main(cfg):
    snr_lower = float(cfg["snr_lower"])
    snr_upper = float(cfg["snr_upper"])
    total_snrlevels = int(cfg["total_snrlevels"])
    
    base_dir = f'../{cfg["phase"]}'
    clean_dir = f'{base_dir}/clean'
    if not os.path.exists(clean_dir):
        assert False, ("Clean speech data is required")
    
    noise_dir = f'{base_dir}/noise'
    if not os.path.exists(noise_dir):
        assert False, ("Noise data is required")
        
    fs = float(cfg["sampling_rate"])
    audioformat = cfg["audioformat"]
    # total_hours = float(cfg["total_hours"])
    # audio_length = float(cfg["audio_length"])
    silence_length = float(cfg["silence_length"])
    noisyspeech_dir = os.path.join(base_dir, 'gen_noisy')
    if not os.path.exists(noisyspeech_dir):
        os.makedirs(noisyspeech_dir)
    clean_proc_dir = os.path.join(base_dir, 'gen_clean')
    if not os.path.exists(clean_proc_dir):
        os.makedirs(clean_proc_dir)
    noise_proc_dir = os.path.join(base_dir, 'gen_noise')
    if not os.path.exists(noise_proc_dir):
        os.makedirs(noise_proc_dir)
        
    # total_secs = total_hours*60*60
    # total_samples = int(total_secs * fs)
    # audio_length = int(audio_length*fs)
    SNR = np.linspace(snr_lower, snr_upper, total_snrlevels)
    cleanfilenames = glob.glob(os.path.join(clean_dir, audioformat))
    if cfg["noise_types_excluded"]=='None':
        noisefilenames = glob.glob(os.path.join(noise_dir, audioformat))
    else:
        filestoexclude = cfg["noise_types_excluded"].split(',')
        noisefilenames = glob.glob(os.path.join(noise_dir, audioformat))
        for i in range(len(filestoexclude)):
            noisefilenames = [fn for fn in noisefilenames if not os.path.basename(fn).startswith(filestoexclude[i])]

    for idx_s in range(len(cleanfilenames)):
        clean, fs = audioread(cleanfilenames[idx_s])
        clean_file_name = Path(cleanfilenames[idx_s]).stem

        for idx_n in range(len(noisefilenames)):
            noise, fs = audioread(noisefilenames[idx_n])
            noise_file_name = Path(noisefilenames[idx_n]).stem
            
            if len(noise)>=len(clean):
                noise = noise[0:len(clean)]            
            else:
                while len(noise)<=len(clean):
                    idx_n = idx_n + 1
                    if idx_n >= np.size(noisefilenames)-1:
                        idx_n = np.random.randint(0, np.size(noisefilenames))
                    newnoise, fs = audioread(noisefilenames[idx_n])
                    noiseconcat = np.append(noise, np.zeros(int(fs*silence_length)))
                    noise = np.append(noiseconcat, newnoise)
            noise = noise[0:len(clean)]
            
            SNR = np.linspace(snr_lower, snr_upper, total_snrlevels)

            for i in range(np.size(SNR)):
                clean_snr, noise_snr, noisy_snr = snr_mixer(clean=clean, noise=noise, snr=SNR[i])
                file_name = f'{clean_file_name}@{noise_file_name}@{int(SNR[i])}.wav'                
                noisypath = os.path.join(noisyspeech_dir, file_name)
                cleanpath = os.path.join(clean_proc_dir, file_name)
                noisepath = os.path.join(noise_proc_dir, file_name)
                audiowrite(noisy_snr, fs, noisypath, norm=False)
                audiowrite(clean_snr, fs, cleanpath, norm=False)
                audiowrite(noise_snr, fs, noisepath, norm=False)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    
    # Configurations: read noisyspeech_synthesizer.cfg
    parser.add_argument("--cfg", default = "noisyspeech_synthesizer.cfg", help = "Read noisyspeech_synthesizer.cfg for all the details")
    parser.add_argument("--cfg_str", type=str, default = "noisy_speech" )
    args = parser.parse_args()

    
    cfgpath = os.path.join(os.path.dirname(__file__), args.cfg)
    assert os.path.exists(cfgpath), f"No configuration file as [{cfgpath}]"
    cfg = CP.ConfigParser()
    cfg._interpolation = CP.ExtendedInterpolation()
    cfg.read(cfgpath)
    
    main(cfg._sections[args.cfg_str])
    