# SSRL
This repository is an official implementation of the paper Scale-aware Spatio-temporal Relation Learning for Video Anomaly Detection

## Introduction
**Abstract.** Recent progress in video anomaly detection (VAD) has shown that feature discrimination is the key to effectively distinguishing anomalies from normal events.
We observe that many anomalous events occur in limited local regions, and the severe background noise increases the difficulty of feature learning.
In this paper, we propose a scale-aware weakly supervised learning approach to capture local and salient anomalous patterns from the background, using only coarse video-level labels as supervision.
We achieve this by segmenting frames into non-overlapping patches and then capturing inconsistencies among different regions through our patch spatial relation (PSR) module, which consists of self-attention mechanisms and dilated convolutions.
To address the scale variation of anomalies and enhance the robustness of our method, a multi-scale patch aggregation method is further introduced to enable local-to-global spatial perception by merging features of patches with different scales. 
Considering the importance of temporal cues, we extend the relation modeling from the spatial domain to the spatio-temporal domain with the help of the existing video temporal relation network to effectively encode the spatio-temporal dynamics in the video.
Experimental results show that our proposed method achieves new state-of-the-art performance on UCF-Crime and ShanghaiTech benchmarks.

## License

This project is released under the [MIT license](./LICENSE).

## Installation

### Requirements

* Linux, CUDA>=9.2
  
* Python>=3.5

    We recommend you to use Anaconda to create a conda environment:
    ```bash
    conda create -n ssrl python=3.7 
    ```
    Then, activate the environment:
    ```bash
    conda activate ssrl
    ```
  
* PyTorch>=1.5.1, torchvision>=0.6.1 (following instructions [here](https://pytorch.org/))

    For example, if your CUDA version is 9.2, you could install pytorch and torchvision as following:
    ```bash
    conda install pytorch=1.5.1 torchvision=0.6.1 cudatoolkit=9.2 -c pytorch
    ```
  
* Other requirements
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Dataset preparation

Please download extracted i3d features for ShanghaiTech and UCF-Crime dataset from [Baidu Wangpan (extract code: wxxy)](https://pan.baidu.com/s/1gWotx_lzjBbCFx1xcPHGzQ) and put them under the coderoot.

### Training

#### Training on single node

For example, the command for training SSRL on 8 GPUs is as following:

```bash
sh scripts/train_ssrl_stage2.sh
```




