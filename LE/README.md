# MetricGAN-OKD for Listening Enhancement

## Requirements
In your environment (python 3.8), the requirements can be installed with:
```shell
pip install -r requirements.txt
pip install torch==1.7.1+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
```
We verified that this is supported in PyTorch versions 1.7.1 to 1.10.1.


## Setup Datasets
**Harvard Sentences:** Please download the two public speeches (Harvard Sentences) and MS-SNSD noise dataset.
* Male speaker (zip file containing all Harvard sentences produced in quiet condition) from [here](https://datashare.ed.ac.uk/handle/10283/3239).
* Female speaker from [here](https://salford.figshare.com/articles/media/Speech_corpus_-_Harvard_-_edited_and_end-pointed_audio/7862465/1).
* Noise dataset MS-SNSD (git clone) from [here](https://github.com/microsoft/MS-SNSD).
	* We selected five types of noise for training and validation and three for testing (refer to Section 4.3 in the paper).
```shell
cd ../data/Harvard_Sentences/make
sh make_dataset.sh
```
Then, (manually) select noise types from MS-SNSD (noise_train/ or noise_test/) and put them in the folder "data/Harvard_Sentences/train(or valid/test)/noise/".

Finally, execute the command below to mix speech and noise with SNR levels.
```shell
python noisyspeech_synthesizer.py   # set phase and SNR levels in .cfg file
```


The final folder structure should look like this:
```none
MetricGAN-OKD
├── ...
├── data
│   ├── Harvard_Sentences
│   │   ├── make
│   │   ├── train
│   │   │   ├── clean
│   │   │   ├── noise
│   │   │   ├── gen_clean
│   │   │   ├── gen_noise
│   │   │   ├── gen_noisy
│   │   ├── valid
│   │   ├── test
├── LE
```

## Training

## Testing & Inference
