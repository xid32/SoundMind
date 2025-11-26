# SoundMind: RL-Incentivized Logic Reasoning for Audio-Language Models [EMNLP 2025 Main Conference (Oral)]

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/xid32/SoundMind/blob/main/LICENSE) 
[![arXiv](https://img.shields.io/badge/arXiv-2506.12935-b31b1b.svg)](https://arxiv.org/abs/2506.12935) 
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-blue)](https://huggingface.co/datasets/SoundMind-RL/SoundMindDataset) 
[![Dropbox](https://img.shields.io/badge/Dropbox-Dataset-brightgreen.svg)](https://www.dropbox.com/scl/fi/irtbrnmk5e0ecvv8fyrum/audio_dataset.zip?rlkey=p1ebkt9h1bkyjsq3fo2bp667v&st=gxr542e2&dl=0) 

This repository is the official implementation of *SoundMind: RL-Incentivized Logic Reasoning for Audio-Language Models* (EMNLP 2025). We introduce **SoundMind-RL**, a novel rule-based reinforcement learning framework that empowers large-scale audio-language models with advanced logical reasoning capabilities across both audio and textual modalities. To enable such training, we build the **SoundMind dataset**, an Audio Logical Reasoning (ALR) benchmark comprising 6,446 high-quality samples annotated with chain-of-thought reasoning in both audio and text forms.


<p align="center">
<img src="./figs/f1.png" alt="Task Figure">
</p>

## Dataset Download

To download our dataset, please visit this link: [Hugging Face](https://huggingface.co/datasets/SoundMind-RL/SoundMindDataset).

Alternatively, you can also download it from [Dropbox](https://www.dropbox.com/scl/fi/irtbrnmk5e0ecvv8fyrum/audio_dataset.zip?rlkey=p1ebkt9h1bkyjsq3fo2bp667v&st=gxr542e2&dl=0).

Run the following command:

```bash
wget -c "https://www.dropbox.com/scl/fi/irtbrnmk5e0ecvv8fyrum/audio_dataset.zip?rlkey=p1ebkt9h1bkyjsq3fo2bp667v&st=gxr542e2&dl=1" -O audio_dataset.zip
```

The dataset contains train, test, and validation splits with corresponding text descriptions and metadata stored as JSON files. All annotation files are located in the `dataset-annotation-json` folder in this GitHub.

## Requirements

### Recommended Hardware

8× NVIDIA H800 80GB or 8× NVIDIA H100 80GB GPUs.

### Codebase and Compatibility

Our codebase is based on [verl](https://github.com/volcengine/verl). If you are already familiar with [verl](https://github.com/volcengine/verl), you should be able to quickly get started with this repository.

### Environment Setup (Recommended: Anaconda)

- **Python**: Version >= 3.9
- **CUDA**: Version >= 12.1

For training and inference engines to utilize better and faster hardware support, CUDA/cuDNN and other dependencies are required, and some of the dependencies are easy to be overridden when installing other packages.

We need to install the following prerequisites:

- **CUDA**: Version >= 12.4
- **cuDNN**: Version >= 9.8.0


```bash
# change directory to anywhere you like, in verl source code directory is not recommended
wget https://developer.download.nvidia.com/compute/cudnn/9.8.0/local_installers/cudnn-local-repo-ubuntu2204-9.8.0_1.0-1_amd64.deb
dpkg -i cudnn-local-repo-ubuntu2204-9.8.0_1.0-1_amd64.deb
cp /var/cudnn-local-repo-ubuntu2204-9.8.0/cudnn-*-keyring.gpg /usr/share/keyrings/
apt-get update
apt-get -y install cudnn-cuda-12
```


Create and activate a new conda environment:

```bash
conda create -n alr python==3.10
conda activate alr
```

Install verl:

```bash
bash scripts/install_vllm_sglang_mcore.sh
pip install --no-deps -e .
```


Please make sure that the installed packages are not overridden during the installation of other packages.

The packages worth checking are:

- **torch** and torch series
- **vLLM**
- **SGLang**
- **pyarrow**
- **tensordict**


For [Qwen2.5-Omni](https://github.com/QwenLM/Qwen2.5-Omni), we need to update some additional library versions.


```bash
pip install transformers==4.52.3
pip install accelerate
pip install qwen-omni-utils[decord] -U
```


## Preprocessing Data
Our project and code rely on  Audio Logical Reasoning (ALR) dataset.


### Generate Parquet Format Dataset


- **Option 1: Two modal inputs are used**

```bash
cd ./examples/data_preprocess
python alr.py
```


- **Option 2: Only texts are used**

```bash
cd ./examples/data_preprocess
python alr_text.py
```


- **Option 3: Only audio is used**

```bash
cd ./examples/data_preprocess
python alr_audio.py
```

## Checkpoint Download

To download our model checkpoint, please visit this link: [Checkpoint Link](https://www.dropbox.com/scl/fi/f24wyecnycfu6g6ip10ac/qwen2_5_omni_logic.zip?rlkey=xlixctyr8cbfpv85arhka0b8c&st=wd5rlh9b&dl=0)

Run the following command:

```bash
wget -c "https://www.dropbox.com/scl/fi/f24wyecnycfu6g6ip10ac/qwen2_5_omni_logic.zip?rlkey=xlixctyr8cbfpv85arhka0b8c&st=wd5rlh9b&dl=1" -O qwen2_5_omni_logic.zip
```

## RL-Training & Evaluation

If you don't want to use the pre-trained model we provided, you can use the official version. You can change the model path implementation in download_qwen25omni.py and main_grpo.sh.


Run the following command:

```bash
python download_qwen25omni.py
bash main_grpo.sh
```



## ✏️ Citation

If you think this project is helpful, please feel free to leave a star⭐️ and cite our paper:

```
@article{diao2025soundmind,
  title={SoundMind: RL-Incentivized Logic Reasoning for Audio-Language Models},
  author={Diao, Xingjian and Zhang, Chunhui and Kong, Keyi and Wu, Weiyi and Ma, Chiyu and Ouyang, Zhongyu and Qing, Peijun and Vosoughi, Soroush and Gui, Jiang},
  journal={arXiv preprint arXiv:2506.12935},
  year={2025}
}
```
