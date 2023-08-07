import os
import time
import json
import shutil
import random
import numpy as np
import soundfile as sf

import torch
import torch.nn as nn
import torchaudio

from tqdm import tqdm
from glob import glob
from pathlib import Path
from os.path import join as opj
from ptflops import get_model_complexity_info

from model import Generator, Discriminator
from dataloader import create_dataloader
from metric_functions.get_metric_scores import get_pesq_parallel, get_csig_parallel, get_cbak_parallel, get_covl_parallel
from signal_processing import get_spec_and_phase, transform_spec_to_wav

fs = 16000

class Trainer:
    def __init__(self, args, data_paths):
        self.args = args
        self.device = torch.device(args.device)
        self.target_metric = [str(tar_metric) for tar_metric in args.target_metric]
        self.num_samples = args.num_of_sampling
        self.num_models = len(self.target_metric)
        
        self.train_noisy_path = data_paths['train_noisy']
        self.train_clean_path = data_paths['train_clean']
        self.train_enhan_path = data_paths['train_enhan']
        self.test_noisy_path = data_paths['test_noisy']
        self.test_clean_path = data_paths['test_clean']
        
        self.model_output_path = data_paths['model_output']
        self.log_output_path = data_paths['log_output']

        os.makedirs(self.train_enhan_path, exist_ok=True)
        os.makedirs(opj(self.args.output_path, self.args.exp_name, 'test_sample'), exist_ok=True)
        os.makedirs(opj(self.args.output_path, self.args.exp_name, 'tmp'), exist_ok=True)
        
        self.generator_train_paths = glob(self.train_clean_path + '/*.wav')
        self.generator_valid_paths = []
        for sample in self.generator_train_paths:
            for speaker in args.val_speaker:
                if speaker in sample:
                    self.generator_valid_paths.append(sample)
        self.generator_train_paths = list(set(self.generator_train_paths) - set(self.generator_valid_paths))
        self.generator_test_paths = glob(self.test_clean_path + '/*.wav')
        random.shuffle(self.generator_train_paths)
        
        self.init_model_optim()
        self.init_target_metric()
        self.init_noisy_score()
        self.best_scores = [{'pesq':-0.5, 'csig':0, 'cbak':0, 'covl':0, 'avg':0} for _ in range(self.num_models)]

        with open(opj(self.log_output_path, 'log.txt'), 'a') as f:
            f.write(f'Train MetricGAN-{self.target_metric}\n')
            f.write(f'Train set:{len(self.generator_train_paths)}, Valid set:{len(self.generator_valid_paths)}, Test set:{len(self.generator_test_paths)}\n')
            f.write(f'Total parameters:{sum(p.numel() for p in self.generator_list[0].parameters())/10**6:.3f}M\n')

        with open(opj(self.args.output_path, self.args.exp_name, 'args.txt'), 'w') as f:
            json.dump(args.__dict__, f, indent=2)

    def init_model_optim(self):
        self.generator_list = []
        self.optimizer_gen_list = []
        self.D = Discriminator(self.num_models).to(self.device)
        self.optimizer_d = torch.optim.Adam(self.D.parameters(), lr=self.args.lr)

        for i in range(self.num_models):        
            self.generator_list.append(Generator(causal=self.args.causal).to(self.device))
            self.optimizer_gen_list.append(torch.optim.Adam(self.generator_list[i].parameters(), lr=self.args.lr))
        self.MSELoss = nn.MSELoss().to(self.device)

    def init_target_metric(self):
        self.target_metric_list = []

        for metric in self.target_metric:
            if metric == 'pesq':
                self.target_metric_list.append(get_pesq_parallel)
            elif metric == 'csig':
                self.target_metric_list.append(get_csig_parallel)
            elif metric == 'cbak':
                self.target_metric_list.append(get_cbak_parallel)
            elif metric == 'covl':
                self.target_metric_list.append(get_covl_parallel)
        print(self.target_metric_list)
        assert len(self.target_metric_list) == len(self.generator_list), [len(self.target_metric_list), len(self.generator_list)]
    
    def init_noisy_score(self):
        self.noisy_set_scores = [{} for _ in range(self.num_models)]
        Noised_name = glob(self.train_noisy_path + '/*.wav')

        for i in range(self.num_models):
            train_score_C_N = self.target_metric_list[i](self.train_clean_path, Noised_name)
            assert len(Noised_name) == len(train_score_C_N), 'must same length'

            for path, score in tqdm(zip(Noised_name, train_score_C_N)):
                self.noisy_set_scores[i][path] = score

    def load_checkpoint(self, ver='latest'):
        checkpoint = torch.load(opj(self.model_output_path, f'{ver}_model{self.okd_step}.pth'))
        self.epoch = checkpoint['epoch']
        self.generator_list[self.okd_step].load_state_dict(checkpoint['generator'])
        self.optimizer_gen_list[self.okd_step].load_state_dict(checkpoint['g_optimizer'])
        if ver == 'best':
            print(f'---{self.epoch} Epoch loaded: model {self.okd_step} weigths and optimizer---')
        else:
            print(f'---load {self.okd_step} model weigths and optimizer---')

    def train(self):
        start_time = time.time()
        self.epoch = 1
        self.historical_set = []
        for epoch in np.arange(self.epoch, self.args.epochs+1):
            self.epoch = epoch
            print(f'{epoch}Epoch start')
            os.makedirs(opj(self.args.output_path, self.args.exp_name, 'test_sample', f'epoch{self.epoch}'), exist_ok=True)
            
            # random sample some training data  
            random.shuffle(self.generator_train_paths)
            genloader = create_dataloader(self.generator_train_paths[0 : round(1*self.args.num_of_sampling)], self.train_noisy_path,
                                          num_target_metric=1, loader='G', batch_size=self.args.batch_size, num_workers=self.args.num_workers)
            self.train_one_epoch(genloader)

        end_time = time.time()
        with open(opj(self.log_output_path, 'log.txt'), 'a') as f:
            f.write(f'Total training time:{(end_time-start_time)/60:.2f}Minute\n')

        # Best validation scores
        for i in range(self.num_models):
            self.okd_step = i
            self.load_checkpoint(ver='best')
            with open(opj(self.log_output_path, 'log.txt'), 'a') as f:
                f.write(f'--------Model{i} Best score on {self.target_metric[0]}--------')
            self.evaluation(data_list=self.generator_test_paths, phase='best')


    def train_one_epoch(self, genloader):
        if self.epoch >= 2:
            self.train_generator(genloader)

        if self.epoch >= self.args.skip_val_epoch:
            if self.epoch % self.args.eval_per_epoch == 0:
                for i in range(self.num_models):
                    self.okd_step = i
                    self.evaluation(data_list=self.generator_test_paths[0:self.args.num_of_val_sample], phase='test')

        self.train_discriminator()
        
        
    def train_generator(self, data_loader):
        [gen.train() for gen in self.generator_list]
        print('Generator training phase')
        for clean_mag, noise_mag, target, length in tqdm(data_loader):          
            clean_mag = clean_mag.to(self.device)  # [B, T, F]
            noise_mag = noise_mag.to(self.device)  # [B, T, F]
            target = target.to(self.device)
            enh_mag_list = []

            for i in range(self.num_models):
                self.okd_step = i
                mask = self.generator_list[i](noise_mag, length)
                mask = mask.clamp(min=0.05)
                enh_mag = torch.mul(mask, noise_mag).unsqueeze(1)
                enh_mag_list.append(enh_mag.detach())

            for i in range(self.num_models):
                self.okd_step = i
                mask = self.generator_list[i](noise_mag, length)
                mask = mask.clamp(min=0.05)
                enh_mag = torch.mul(mask, noise_mag).unsqueeze(1)

                ref_mag = clean_mag.detach().unsqueeze(1)
                d_inputs = torch.cat([ref_mag, enh_mag], dim=1)
                assert noise_mag.size(2) == 257, 'gen'
                assert clean_mag.size(2) == 257, 'gen'

                score = self.D(d_inputs)[:, i].unsqueeze(1)
                loss = self.MSELoss(score, target)
                assert score.shape == target.shape, [score.shape, target.shape]

                kd_loss = 0
                # Generator train from other discriminators
                if self.args.kd_type == 'dml':
                    for j in range(self.num_models):
                        if i!=j:
                            kd_loss += self.MSELoss(enh_mag, enh_mag_list[j])
                else:
                    raise ValueError("Expected 'dml' for distillation type") 

                self.optimizer_gen_list[i].zero_grad()
                loss = loss + self.args.alpha * kd_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.generator_list[i].parameters(), 5.0)
                self.optimizer_gen_list[i].step()
    
    def evaluation(self, data_list, phase='valid'):
        print(f'Evaluation on {phase} data')
        test_enhanced_name = []
        self.generator_list[self.okd_step].eval()
        if (phase == 'test') or (phase == 'best'):    # Test path
            clean_path = self.test_clean_path
            noise_path = self.test_noisy_path
        else:                                         # phase == valid
            clean_path = self.train_clean_path
            noise_path = self.train_noisy_path

        with torch.no_grad():
            for i, path in enumerate(tqdm(data_list)):
                wave_name = os.path.basename(path)
                name = Path(wave_name).stem
                suffix = Path(wave_name).suffix

                clean_wav, _ = torchaudio.load(path)
                noise_wav, _ = torchaudio.load(opj(noise_path, wave_name))
                noise_mag, noise_phase = get_spec_and_phase(noise_wav.to(self.device))
                assert noise_mag.size(2) == 257, 'eval'
                assert noise_phase.size(2) == 257, 'eval'

                mask = self.generator_list[self.okd_step](noise_mag)
                mask = mask.clamp(min=0.05)

                enh_mag = torch.mul(mask, noise_mag)
                enh_wav = transform_spec_to_wav(torch.expm1(enh_mag), noise_phase, signal_length=clean_wav.size(1)).detach().cpu().numpy().squeeze()

                enhanced_name=opj(self.args.output_path, self.args.exp_name, 'tmp', f'{name}#{self.epoch}{suffix}')
                
                sf.write(enhanced_name, enh_wav, fs)
                test_enhanced_name.append(enhanced_name)

        # Calculate True PESQ
        test_PESQ = get_pesq_parallel(clean_path, test_enhanced_name, norm=False)
        test_CSIG = get_csig_parallel(clean_path, test_enhanced_name, norm=False)
        test_CBAK = get_cbak_parallel(clean_path, test_enhanced_name, norm=False)
        test_COVL = get_covl_parallel(clean_path, test_enhanced_name, norm=False)
        test_PESQ, test_CSIG, test_CBAK, test_COVL = np.mean(test_PESQ), np.mean(test_CSIG), np.mean(test_CBAK), np.mean(test_COVL)

        test_scores = {'pesq':test_PESQ, 'csig':test_CSIG, 'cbak':test_CBAK, 'covl':test_COVL}

        with open(opj(self.log_output_path, 'log.txt'), 'a') as f:
            f.write(f'Epoch:{self.epoch}-M{self.okd_step} | Test PESQ:{test_PESQ:.3f} | Test CSIG:{test_CSIG:.3f} | Test CBAK:{test_CBAK:.3f} | Test COVL:{test_COVL:.3f}\n')

        if (phase == 'valid') or (phase == 'test'):    # Test path
            checkpoint = {  'epoch': self.epoch,
                            'stats': test_scores,
                            'generator': self.generator_list[self.okd_step].state_dict(),
                            'discriminator': self.D.state_dict(),
                            'g_optimizer': self.optimizer_gen_list[self.okd_step].state_dict(),
                            'd_optimizer': self.optimizer_d.state_dict(),
                            }

            if test_scores['pesq'] >= self.best_scores[self.okd_step]['pesq']:
                print('----------------------------------------')
                print('-----------------SAVE-------------------')
                self.best_scores[self.okd_step] = test_scores

                # save the current enhancement model
                torch.save(checkpoint, opj(self.model_output_path, f'best_model{self.okd_step}.pth'))
                print('----------------------------------------')
            torch.save(checkpoint, opj(self.model_output_path, f'latest_model{self.okd_step}.pth'))


    def get_score(self):
        print('Get scores for discriminator training')
        D_paths = self.generator_train_paths[0:self.args.num_of_sampling]

        Enhanced_name = []
        Noised_name = []

        self.generator_list[self.okd_step].eval()
        with torch.no_grad():
            for path in tqdm(D_paths):
                wave_name = os.path.basename(path)
                name = Path(wave_name).stem
                suffix = Path(wave_name).suffix
                
                clean_wav, _ = torchaudio.load(path)
                noise_wav, _ = torchaudio.load(self.train_noisy_path+wave_name)
                noise_mag, noise_phase = get_spec_and_phase(noise_wav.to(self.device))

                mask = self.generator_list[self.okd_step](noise_mag)
                mask = mask.clamp(min=0.05)
                enh_mag = torch.mul(mask, noise_mag)
                enh_wav = transform_spec_to_wav(torch.expm1(enh_mag), noise_phase, signal_length=clean_wav.size(1)).detach().cpu().numpy().squeeze()

                assert noise_mag.size(2) == 257, 'get_score'
                assert noise_phase.size(2) == 257, 'get_score'
                assert enh_mag.size(2) == 257, 'get_score'

                enhanced_name=opj(self.train_enhan_path, name+'#'+str(self.epoch)+f'_model_{self.okd_step}'+suffix)
                sf.write(enhanced_name, enh_wav, fs)
                Enhanced_name.append(enhanced_name)
                Noised_name.append(self.train_noisy_path+wave_name)

        # Calculate true score
        train_scores = []
        for i in range(self.num_models):
            train_score_C_E = self.target_metric_list[i](self.train_clean_path, Enhanced_name)
            train_score_C_N = [self.noisy_set_scores[i][path] for path in Noised_name]
            train_score_C_C = [1.0] * self.num_samples
            train_scores.append(train_score_C_E)
            train_scores.append(train_score_C_N)
            train_scores.append(train_score_C_C)

        train_scores.append(Enhanced_name)
        current_set = np.array(train_scores).T.tolist()   # [num_sampling, num_metrics*3+1]

        random.shuffle(current_set)
        return current_set

    def subtrain_discriminator(self, data_loader, hist=False):
        for i, (enhance_mag, noisy_mag, clean_mag, target) in enumerate(tqdm(data_loader)):
            target = target.to(self.device)    # (B, 3 x num target metrics) | [M1-CE,CN,CC, M2-CE,CN,CC scores]

            if hist:
                inputs = torch.cat([clean_mag, enhance_mag], dim=1).to(self.device)  # (clean, enhanced)
                score = self.D(inputs)
                indices = torch.tensor([3*x for x in range(self.num_models)], device=self.device)
                target_tmp = torch.index_select(target, 1, indices)
                assert score.shape == target_tmp.shape, [score.shape, target_tmp.shape, target.shape]
                loss = self.MSELoss(score, target_tmp)
                self.optimizer_d.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.D.parameters(), 5.0)
                self.optimizer_d.step()
            else:
                for j, target_mag in enumerate([enhance_mag, noisy_mag, clean_mag]):
                    inputs = torch.cat([clean_mag, target_mag], dim=1).to(self.device)  # (clean, enhanced), (clean, noisy), (clean, clean)
                    score = self.D(inputs)
                    indices = j + torch.tensor([3*x for x in range(self.num_models)], device=self.device)
                    target_tmp = torch.index_select(target, 1, indices)
                    assert score.shape == target_tmp.shape, [score.shape, target_tmp.shape, target.shape]
                    loss = self.MSELoss(score, target_tmp)
                    self.optimizer_d.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.D.parameters(), 5.0)
                    self.optimizer_d.step()

    def train_discriminator(self):
        print("Discriminator training phase")
        self.D.train()
        self.okd_step = random.randint(0, len(self.generator_list)-1)

        # Get true score of train data
        current_set = self.get_score()
        # Training current list
        disc_loader = create_dataloader(current_set, self.train_noisy_path, clean_path=self.train_clean_path, loader='D',
                                        batch_size=self.args.batch_size, num_workers=self.args.num_workers)
        self.subtrain_discriminator(disc_loader)

        random.shuffle(self.historical_set)
        
        # Training hist list
        train_hist_length = int(len(self.historical_set) * self.args.hist_portion)
        train_concat_set=self.historical_set[0 : train_hist_length] + current_set
        random.shuffle(train_concat_set)
        disc_loader_hist = create_dataloader(train_concat_set, self.train_noisy_path, clean_path=self.train_clean_path, loader='D',
                                             batch_size=self.args.batch_size, num_workers=self.args.num_workers)
        self.subtrain_discriminator(disc_loader_hist, hist=True)
        
        # Update the history list
        self.historical_set = self.historical_set + current_set

        # Training current list again
        self.subtrain_discriminator(disc_loader)
