# AdvLB
Code and model for "Adversarial Laser Beam: Effective Physical-World Attack to DNNs in a Blink" (CVPR 2021)

(All will be done before April 18th, hopefully.)
## Introduction
Natural phenomena may play the role of adversarial attackers, e.g., a blinding glare results in a fatal crash of a Tesla self-driving car.
What if a beam of light can adversarially attack a DNN? Further, how about using a beam of light, specifically the laser beam, as the weapon to perform attacks.
In this work, we show a simple and cool attack by simply using a laser beam.  
To this end, we propose a novel attack method called Adversarial Laser Beam (AdvLB), which enables manipulation of laser beam's physical parameters to perform adversarial attack.
## Installation
#### Dependencies
* CUDA VERSION 10.2
#### Create environment
* pip install -e requirements.txt
## Basic Usage
#### Attack
command
#### Defense
Besides revealing the potential threats of AdvLB, in this work, we also try to suggest an effective defense for laser beam attack. Similar to adversarial training, we progressively improve the robustness by injecting the laser beam as perturbations into the data for training. The details about training can be referred to the paper.

Models | Std. Acc. rate(%) | Attack Succ. rate(%)
------------ | ------------- | -------------
ResNet50(org) | 78.19 | 95.10
ResNet50(adv trained) | 78.40 |77.20

The weights of "adv trained" ResNet50 model can be downloaded [here](https://drive.google.com/file/d/1HtwnsCFqKkoJoSSHo23BP90_ZCAVD_L7/view?usp=sharing).
## Physical setting
## Examples
## Limitations
Slow
## Q&A
## Acknowlegement
## Cite
