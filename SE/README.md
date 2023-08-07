# MetricGAN-OKD for Speech Enhancement

## Requirements
In your environment (python 3.8), the requirements can be installed with:
```shell
pip install -r requirements.txt
pip install torch==1.7.1+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
```
We verified that this is supported in PyTorch versions 1.7.1 to 1.10.1.


## Setup Datasets
**VoiceBank-DEMAND:** Please download clean_trainset_28spk_wav.zip, noisy_trainset_28spk_wav.zip, clean_testset_wav.zip, and noisy_testset_wav.zip from [here](https://datashare.ed.ac.uk/handle/10283/2791)
and extract them to `../data/VCTK_DEMAND_48k/train(or test)/clean(or noisy)`.

The sample rate of original dataset is 48kHz. We downsample the audio files from 48kHz to 16kHz as follows.
```shell
python downsample.py
```


The final folder structure should look like this:
```none
MetricGAN-OKD
├── ...
├── data
│   ├── VCTK_DEMAND
│   │   ├── train
│   │   │   ├── clean
│   │   │   ├── noisy
│   │   ├── test
│   │   │   ├── clean
│   │   │   ├── noisy
│   │   │   │
├── SE
```

## Training
```shell
python main.py --exp_name=exp1 --target_metric pesq csig
```
You can change the target metrics and hyperparameters (epochs, batch_size, hist_portion, lr, ...).
```shell
python main.py --exp_name=exp2 --target_metric pesq covl --hist_portion=0.1
```


## Testing & Inference
```shell
python inference.py --weight_path results/exp1/model/ --weight_file best_model.pth
```


## Results and Checkpoints
We provide checkpoints on the VoiceBank-DEMAND dataset.

| Target Metric       | PESQ | CSIG | CBAK | COVL | Note |
|---------------------|------|------|------|------|------|
| PESQ, CSIG          | 3.24 | 4.23 | 3.07 | 3.73 |------|
| CSIG, PESQ          | 3.19 | 4.26 | 3.12 | 3.72 |------|
| PESQ, CSIG+CBAK+COVL| 3.15 | 4.26 | 3.25 | 3.71 |------|
| PESQ, CSIG+CBAK     | 3.12 | 4.17 | 3.13 | 3.64 |Causal|

Please download the weights file from [our release](https://github.com/wooseok-shin/MetricGAN-OKD/releases/tag/v1.weights), 
put them in your path, and run inference.
```shell
python inference.py --weight_path your/path/ --weight_file 1_PE_CS_Table2.pth

```
