import os
import shutil
import natsort
import torchaudio
from glob import glob
from pathlib import Path
from os.path import join as opj

sample_rate = 16000
tf = torchaudio.transforms.Resample(48000, sample_rate)
path = '../'
os.makedirs(path, exist_ok=True)

## Step1: male - harvard_001.wav -> m_harvard_001.wav
files = glob(f'{path}/make/quiet_harvard/*.wav')

for file in files:
    name = os.path.basename(file)
    new_path = opj(path, 'm_'+name)  # male
    
    speech, sr = torchaudio.load(file)
    speech = tf(speech)   # downsample
    torchaudio.save(new_path, speech, sample_rate)
        
## Step2: female - HARVARD_list{num_folder}_EndPointed/...wavs -> harvard_female_origin/...wavs
files = glob(f'{path}/make/HARVARD_Edited_EP/*/*.wav', recursive=True)
files = natsort.natsorted(files)

for i, file in enumerate(files, 1):
    name = f'f_hvd_{str(i).zfill(3)}.wav'
    new_path = opj(path, name)  # male
    
    speech, sr = torchaudio.load(file)
    speech = tf(speech)    # downsample
    torchaudio.save(new_path, speech, sample_rate)


## Step3: Split all data to train/valid/test
files = glob(path + '*.wav')
files = natsort.natsorted(files)

for file in files:
    number = int(Path(file).stem[-3:])
    if number <= 600:
        os.makedirs(opj(path, 'train', 'clean'), exist_ok=True)
        shutil.move(file, opj(path, 'train', 'clean'))
        os.makedirs(opj(path, 'train', 'noise'), exist_ok=True)
    elif (number > 600) and (number <= 660):
        os.makedirs(opj(path, 'valid', 'clean'), exist_ok=True)
        shutil.move(file, opj(path, 'valid', 'clean'))
        os.makedirs(opj(path, 'valid', 'noise'), exist_ok=True)
    else:
        os.makedirs(opj(path, 'test', 'clean'), exist_ok=True)
        shutil.move(file, opj(path, 'test', 'clean'))
        os.makedirs(opj(path, 'test', 'noise'), exist_ok=True)
        

## Step4: Noise (MS-SNSD) dataset downsampling (48 -> 16kHz)
files = glob(f'{path}/make/MS-SNSD/noise_train/*.wav')

for file in files:
    speech, sr = torchaudio.load(file)
    speech = tf(speech)    # downsample
    torchaudio.save(file, speech, sample_rate)