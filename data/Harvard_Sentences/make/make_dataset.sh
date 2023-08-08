
# 1. unzip zip files (male + female speeches)
unzip '*.zip' -d ./

# 2. git clone noise dataset (MS-SNSD)
git clone https://github.com/microsoft/MS-SNSD.git

# 3. Preprocessing (integrate files and downsample) and Split them to train/valid/test
python make_dataset.py

# 4. (manual) Select noise types from MS-SNSD and move to the folder "data/Harvard_Sentences/train(or valid/test)/noise/"
##

# 5. Mixing clean and noise (set phase and SNR levels in .cfg file)
# python noisyspeech_synthesizer.py