# Silent Speech

This repo is to reproduce this project. 

# EMG to Text Recognition
> Convert silent speech (EMG signals) directly to text using deep learning

This guide will help you set up and run the pretrained EMG-to-text recognition model, which achieves a Word Error Rate (WER) of approximately 28%.

## Quick Links
- [Pretrained Model](https://doi.org/10.5281/zenodo.7183877)
- [EMG Dataset](https://doi.org/10.5281/zenodo.4064408)
- [Language Model](https://github.com/mozilla/DeepSpeech/releases/download/v0.6.1/lm.binary)

## System Requirements
- Linux/Unix-based system (Windows not supported due to CTC decode library)
- CUDA-capable GPU
- Python 3.8 or higher
- Git

## Installation Steps

1. **Clone the Repository**
```bash
git clone https://github.com/dgaddy/silent_speech.git
cd silent_speech
```

2. **Set up Conda Environment**
```bash
# Create and activate environment
conda env create -f environment.yml
conda activate silent_speech

# Install CTC decode library (Required for text recognition)
pip install git+https://github.com/parlance/ctcdecode.git
```

3. **Download Required Files**

```bash
# Create directories
mkdir -p models/recognition_model
mkdir -p emg_data

# Download pretrained model
wget https://zenodo.org/record/7183877/files/recognition_model.pt -O models/recognition_model/model.pt

# Download EMG dataset
wget https://zenodo.org/record/4064408/files/emg_data.zip
unzip emg_data.zip -d emg_data/

# Download language model
wget https://github.com/mozilla/DeepSpeech/releases/download/v0.6.1/lm.binary
```

4. **Initialize Submodules**
```bash
git submodule init
git submodule update
tar -xvzf text_alignments/text_alignments.tar.gz
```

## Directory Structure
Your directory should look like this after setup:
```
silent_speech/
├── models/
│   └── recognition_model/
│       └── model.pt
├── emg_data/
│   ├── nonparallel_data/
│   ├── silent_parallel_data/
│   └── voiced_parallel_data/
├── lm.binary
└── text_alignments/
```

## Usage

### Running Inference
To evaluate the pretrained model on the test set:
```bash
python recognition_model.py --evaluate_saved "./models/recognition_model/model.pt"
```

### Training a New Model
If you want to train your own model:
```bash
python recognition_model.py --output_directory "./models/recognition_model_new/"
```

### Custom Data Processing
To process your own EMG data, you'll need to format it similarly to the provided dataset. The EMG data should be:
- Sampled at 1000Hz
- Contain the same channel configuration as the original dataset
- Preprocessed to remove artifacts and noise

## Troubleshooting

### Common Issues

1. **CTC Decode Installation Error**
```
Error: If you see compilation errors during ctcdecode installation:
Solution: Ensure you have build-essential and cmake installed:
sudo apt-get install build-essential cmake
```

2. **CUDA Issues**
```
Error: CUDA not found or version mismatch
Solution: Ensure CUDA 11.8 is installed and visible to conda:
conda install cudatoolkit=11.8
```

3. **Dataset Loading Error**
```
Error: Cannot find EMG data directory
Solution: Check if the emg_data directory contains the unzipped dataset files
```

### Additional Tips
- Monitor GPU memory usage during inference - the model requires approximately 4GB VRAM
- For large EMG files, process them in batches to avoid memory issues
- Keep EMG signals properly calibrated and normalized for best results

## Performance Metrics
- Word Error Rate (WER): ~28% on test set
- Processing Speed: ~0.1x realtime on NVIDIA RTX 3080
- Memory Usage: ~4GB VRAM during inference

## Citation
If you use this model in your research, please cite:
```bibtex
@phdthesis{gaddy2022voicing,
  title={Voicing Silent Speech},
  author={Gaddy, David},
  year={2022},
  school={University of California, Berkeley}
}
```

## Support
For issues and questions:
1. Check the [original repository issues](https://github.com/dgaddy/silent_speech/issues)
2. Ensure your environment matches the requirements exactly
3. Check the troubleshooting section above

## License
Check the original repository for license information.



---

# Voicing Silent Speech

This repository contains code for synthesizing speech audio from silently mouthed words captured with electromyography (EMG).
It is the official repository for the papers [Digital Voicing of Silent Speech](https://aclanthology.org/2020.emnlp-main.445.pdf) at EMNLP 2020, [An Improved Model for Voicing Silent Speech](https://aclanthology.org/2021.acl-short.23.pdf) at ACL 2021, and the dissertation [Voicing Silent Speech](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2022/EECS-2022-68.pdf).
The current commit contains only the most recent model, but the versions from prior papers can be found in the commit history.
On an ASR-based open vocabulary evaluation, the latest model achieves a WER of approximately 36%.
Audio samples can be found [here](https://dgaddy.github.io/silent_speech_samples/June2022/).

The repository also includes code for directly converting silent speech to text.  See the section labeled [Silent Speech Recognition](#silent-speech-recognition).

## Data

The EMG and audio data can be downloaded from <https://doi.org/10.5281/zenodo.4064408>.  The scripts expect the data to be located in a `emg_data` subdirectory by default, but the location can be overridden with flags (see the top of `read_emg.py`).

Force-aligned phonemes from the Montreal Forced Aligner have been included as a git submodule, which must be updated using the process described in "Environment Setup" below.
Note that there will not be an exception if the directory is not found, but logged phoneme prediction accuracies reporting 100% is a sign that the directory has not been loaded correctly.

## Environment Setup

We strongly recommend running in Anaconda.
To create a new environment with all required dependencies, run:
```
conda env create -f environment.yml
conda activate silent_speech
```
This will install with CUDA 11.8.

You will also need to pull git submodules for Hifi-GAN and the phoneme alignment data, using the following commands:
```
git submodule init
git submodule update
tar -xvzf text_alignments/text_alignments.tar.gz
```

Use the following commands to download pre-trained DeepSpeech model files for evaluation.  It is important that you use DeepSpeech version 0.7.0 model files for evaluation numbers to be consistent with the original papers.  Note that more recent DeepSpeech packages such as version 0.9.3 can be used as long as they are compatible with version 0.7.x model files.
```
curl -LO https://github.com/mozilla/DeepSpeech/releases/download/v0.7.0/deepspeech-0.7.0-models.pbmm
curl -LO https://github.com/mozilla/DeepSpeech/releases/download/v0.7.0/deepspeech-0.7.0-models.scorer
```

(Optional) Training will be faster if you re-run the audio cleaning, which will save re-sampled audio so it doesn't have to be re-sampled every training run.
```
python data_collection/clean_audio.py emg_data/nonparallel_data emg_data/silent_parallel_data emg_data/voiced_parallel_data
```

## Pre-trained Models

Pre-trained models for the vocoder and transduction model are available at
<https://doi.org/10.5281/zenodo.6747411>.

## Running

To train an EMG to speech feature transduction model, use
```
python transduction_model.py --hifigan_checkpoint hifigan_finetuned/checkpoint --output_directory "./models/transduction_model/"
```
where `hifigan_finetuned/checkpoint` is a trained HiFi-GAN generator model (optional).
At the end of training, an ASR evaluation will be run on the validation set if a HiFi-GAN model is provided.

To evaluate a model on the test set, use
```
python evaluate.py --models ./models/transduction_model/model.pt --hifigan_checkpoint hifigan_finetuned/checkpoint --output_directory evaluation_output
```

By default, the scripts now use a larger validation set than was used in the original EMNLP 2020 paper, since the small size of the original set gave WER evaluations a high variance.  If you want to use the original validation set you can add the flag `--testset_file testset_origdev.json`.

## HiFi-GAN Training

The HiFi-GAN model is fine-tuned from a multi-speaker model to the voice of this dataset.  Spectrograms predicted from the transduction model are used as input for fine-tuning instead of gold spectrograms.  To generate the files needed for HiFi-GAN fine-tuning, run the following with a trained model checkpoint:
```
python make_vocoder_trainset.py --model ./models/transduction_model/model.pt --output_directory hifigan_training_files
```
The resulting files can be used for fine-tuning using the instructions in the hifi-gan repository.
The pre-trained model was fine-tuned for 75,000 steps, starting from the `UNIVERSAL_V1` model provided by the HiFi-GAN repository.
Although the HiFi-GAN is technically fine-tuned for the output of a specific transduction model, we found it to transfer quite well and shared a single HiFi-GAN for most experiments.

# Silent Speech Recognition

This section is about converting silent speech directly to text rather than synthesizing speech audio.
The speech-to-text model uses the same neural architecture but with a CTC decoder, and achieves a WER of approximately 28% (as described in the dissertation [Voicing Silent Speech](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2022/EECS-2022-68.pdf)).

You will need to install the ctcdecode library (1.0.3) in addition to the libraries listed above to use the recognition code.
(This package cannot be built successfully under Windows platform)
```
pip install git+https://github.com/parlance/ctcdecode.git
```

And you will need to download a KenLM language model, such as this one from DeepSpeech:
```
curl https://github.com/mozilla/DeepSpeech/releases/download/v0.6.1/lm.binary
```

Pre-trained model weights can be downloaded from <https://doi.org/10.5281/zenodo.7183877>.

To train a model, run
```
python recognition_model.py --output_directory "./models/recognition_model/"
```

To run a test set evaluation on a saved model, use
```
python recognition_model.py --evaluate_saved "./models/recognition_model/model.pt"
```
