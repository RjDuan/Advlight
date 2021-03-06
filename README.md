# AdvLB
Code and model for "Adversarial Laser Beam: Effective Physical-World Attack to DNNs in a Blink" (CVPR 2021)
<p align='center'>
  <img src='imgs/night-test.png' width='700'/>
</p>


## Introduction
Natural phenomena may play the role of adversarial attackers, e.g., a blinding glare results in a fatal crash of a Tesla self-driving car.
What if a beam of light can adversarially attack a DNN? Further, how about using a beam of light, specifically the laser beam, as the weapon to perform attacks.
In this work, we show a simple and cool attack by simply using a laser beam.  
To this end, we propose a novel attack method called Adversarial Laser Beam (AdvLB), which enables manipulation of laser beam's physical parameters to perform adversarial attack.
## Install& Requirements
#### Dependencies
* CUDA VERSION 10.2
#### Create environment
```sh
conda env create -f environment.yaml
conda activate advlb_env
```
#### Code
```sh
git clone https://github.com/RjDuan/AdvLB/
cd AdvLB-main
```
## Basic Usage
#### Attack
```sh
python test.py --model resnet50 --dataset your_dataset
```
#### Defense
Besides revealing the potential threats of AdvLB, in this work, we also analyze the reason of error caused by AdvLB and try to suggest an effective defense for laser beam attack. 
<p align='center'>
  <img src='imgs/err1.png' width='300'/>
   <img src='imgs/err2.png' width='300'/>
</p>
Similar to adversarial training, we progressively improve the robustness by injecting the laser beam as perturbations into the data for training. The details about training can be referred to the paper.


Models | Std. Acc. rate(%) | Attack Succ. rate(%)
------------ | ------------- | -------------
ResNet50(org) | 78.19 | 95.10
ResNet50(adv trained) | 78.40 |77.20


The weights of "adv trained" ResNet50 model can be downloaded [here](https://drive.google.com/file/d/1HtwnsCFqKkoJoSSHo23BP90_ZCAVD_L7/view?usp=sharing).
```sh
python test.py --model df_resnet50 --dataset your_dataset
```
## Dataset
The dataset we used in the paper can be downloaded [here](https://drive.google.com/file/d/1sYNOkks0Ri5zj_xDqNMUuftB9Fp3PdK2/view?usp=sharing)
## Physical setting


## Q&A
Questions are welcome via ranjieduan@gmail.com
## Acknowlegement
* The defense part is completed by Xiaofeng Mao. 
## Cite

```
@inproceedings{duan2021adversarial,
  title={Adversarial Laser Beam: Effective Physical-World Attack to DNNs in a Blink},
  author={Duan, Ranjie and Mao, Xiaofeng and Qin, A Kai and Chen, Yuefeng and Ye, Shaokai and He, Yuan and Yang, Yun},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={16062--16071},
  year={2021}
}
```
<!-- {"mode":"full","isActive":false} -->
