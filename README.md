# 2S-GDA
This repository contains code to reproduce results from the paper:

A Two-Stage Globally-Diverse Adversarial Attack for Vision-Language Pre-training Models

For additional implementation details, experimental results, and visualizations, please refer to the supplementary material provided in Supplementary_Material.pdf.
<p align="left">
    <img src="./images/" width=100%\>
</p>
## Requirements
### Install dependencies
```bash
pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118
```

### Prepare datasets and models
Download the datasets, [Flickr30k](https://shannon.cs.illinois.edu/DenotationGraph/) and [MSCOCO](https://cocodataset.org/#home) (the annotations is provided in ./data_annotation/). Set the root path of the dataset in`./configs/Retrieval_flickr.yaml, image_root`.
The checkpoints of the fine-tuned VLP models is accessible in [ALBEF](https://github.com/salesforce/ALBEF), [TCL](https://github.com/uta-smile/TCL), [CLIP](https://huggingface.co/openai/clip-vit-base-patch16).

### Parameter description
```
--disable_sga_last_step: Disable SGA last step iteration (Formula 3). Default: Disabled.
--enable_synonyms: Enable synonym candidate expansion for text attack. Default: Disabled.
--enable_adaptive_replacement: Use new non-greedy word replacement strategy instead of traditional greedy. Default: Greedy.
--enable_bsr_img_aug: Disable BSR image augmentation. Default: Disabled.
```

## Transferability Evaluation
###Image-Text Retrieval Attack Evaluation
Image-Text Retrieval Attack Evaluation on Flickr30K dataset

1.baseline(SGA):
```bash
python eval.py
```
2.baseline(SGA-BSR):
```bash
python eval.py --enable_bsr_img_aug
```
3.Ours(2S-GDA):
```bash
python eval.py --disable_sga_last_step --enable_synonyms --enable_adaptive_replacement --enable_bsr_img_aug
```
