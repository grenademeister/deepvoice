---
license: mit
datasets:
- awsaf49/sonics
language:
- en
metrics:
- f1
pipeline_tag: audio-classification
tags:
- deepfake
- audio_classification
- fake_song_detection
- music
- song
---

<div style="text-align: center; color: white; padding: 20px;">
    <div align="center"><img src="https://i.postimg.cc/3Jx3yZ5b/real-vs-fake-sonics-w-logo.jpg" width="250" alt="SONICS Logo"></div>
    <h1>SONICS: Synthetic Or Not - Identifying Counterfeit Songs</h1>
    <h3 style="color:red;"><b>ICLR 2025 [Poster]</b></h3>
    <div style="display: flex; justify-content: center; flex-wrap: wrap; gap: 10px; margin: 20px 0;">
        <a href="https://arxiv.org/abs/2408.14080" style="text-decoration: none;">
            <img src="https://img.shields.io/badge/ArXiv-Paper-red" alt="Paper">
        </a>
        <a href="https://huggingface.co/collections/awsaf49/sonics-spectttra-67bb6517b3920fd18e409013" style="text-decoration: none;">
            <img src="https://img.shields.io/badge/HuggingFace-Model-yellow" alt="Hugging Face">
        </a>
        <a href="https://huggingface.co/datasets/awsaf49/sonics" style="text-decoration: none;">
            <img src="https://img.shields.io/badge/HuggingFace-Dataset-orange" alt="Hugging Face Dataset">
        </a>
        <a href="https://www.kaggle.com/datasets/awsaf49/sonics-dataset" style="text-decoration: none;">
            <img src="https://img.shields.io/badge/Kaggle-Dataset-blue?logo=kaggle" alt="Kaggle Dataset">
        </a>
        <a href="https://huggingface.co/spaces/awsaf49/sonics-fake-song-detection" style="text-decoration: none;">
            <img src="https://img.shields.io/badge/HuggingFace-Demo-blue" alt="Hugging Face Demo">
        </a>
    </div>
</div>

---

## 📌 Abstract

The recent surge in AI-generated songs presents exciting possibilities and challenges. These innovations necessitate the ability to distinguish between human-composed and synthetic songs to safeguard artistic integrity and protect human musical artistry. Existing research and datasets in fake song detection only focus on singing voice deepfake detection (SVDD), where the vocals are AI-generated
but the instrumental music is sourced from real songs. However, these approaches are inadequate for detecting contemporary end-to-end artificial songs where all components (vocals, music, lyrics, and style) could be AI-generated. Additionally, existing datasets lack music-lyrics diversity, long-duration songs, and open-access fake songs. To address these gaps, we introduce SONICS, a novel dataset
for end-to-end Synthetic Song Detection (SSD), comprising over 97k songs (4,751 hours) with over 49k synthetic songs from popular platforms like Suno and Udio. Furthermore, we highlight the importance of modeling long-range temporal dependencies in songs for effective authenticity detection, an aspect entirely overlooked in existing methods. To utilize long-range patterns, we introduce SpecTTTra, a novel architecture that significantly improves time and memory efficiency over conventional CNN and Transformer-based models. For long songs, our top-performing variant outperforms ViT by 8% in F1 score, is 38% faster, and uses 26% less memory, while also surpassing ConvNeXt with a 1% F1 score gain, 20% speed boost, and 67% memory reduction.

## 🔗 Links 

- 📄 [**Paper**](https://openreview.net/forum?id=PY7KSh29Z8)  
- 🎵 [**Dataset**](https://huggingface.co/datasets/awsaf49/sonics)  
- 🔬 [**ArXiv**](https://arxiv.org/abs/2408.14080)  
- 💻 [**GitHub**](https://github.com/awsaf49/sonics)  


## 🏆 Model Performance

<style>
    .hf-button {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 6px 12px;
        font-size: 14px;
        font-weight: bold;
        color: white;
        border-radius: 6px;
        text-decoration: none;
    }
    .hf-button img {
        height: 18px;
    }
</style>

| Model Name                     | HF Link | Variant        | Duration | f_clip | t_clip | F1  | Sensitivity | Specificity | Speed (A/S) | FLOPs (G) | Mem. (GB) | # Act. (M) | # Param. (M) |
|--------------------------------|---------|---------------|----------|--------|--------|-----|-------------|-------------|-------------|-----------|-----------|------------|-------------|
| `sonics-spectttra-alpha-5s`   | <a class="hf-button" href="https://huggingface.co/awsaf49/sonics-spectttra-alpha-5s"><img src="https://huggingface.co/front/assets/huggingface_logo-noborder.svg">HF</a>  | SpecTTTra-α   | 5s       | 1      | 3      | 0.78 | 0.69        | 0.94        | 148         | 2.9       | 0.5       | 6          | 17          |
| `sonics-spectttra-beta-5s`    | <a class="hf-button" href="https://huggingface.co/awsaf49/sonics-spectttra-beta-5s"><img src="https://huggingface.co/front/assets/huggingface_logo-noborder.svg">HF</a>  | SpecTTTra-β   | 5s       | 3      | 5      | 0.78 | 0.69        | 0.94        | 152         | 1.1       | 0.2       | 5          | 17          |
| `sonics-spectttra-gamma-5s`   | <a class="hf-button" href="https://huggingface.co/awsaf49/sonics-spectttra-gamma-5s"><img src="https://huggingface.co/front/assets/huggingface_logo-noborder.svg">HF</a>  | SpecTTTra-γ   | 5s       | 5      | 7      | 0.76 | 0.66        | 0.92        | 154         | 0.7       | 0.1       | 2          | 17          |
| `sonics-spectttra-alpha-120s` | <a class="hf-button" href="https://huggingface.co/awsaf49/sonics-spectttra-alpha-120s"><img src="https://huggingface.co/front/assets/huggingface_logo-noborder.svg">HF</a>  | SpecTTTra-α   | 120s     | 1      | 3      | 0.97 | 0.96        | 0.99        | 47          | 23.7      | 3.9       | 50         | 19          |
| `sonics-spectttra-beta-120s`  | <a class="hf-button" href="https://huggingface.co/awsaf49/sonics-spectttra-beta-120s"><img src="https://huggingface.co/front/assets/huggingface_logo-noborder.svg">HF</a>  | SpecTTTra-β   | 120s     | 3      | 5      | 0.92 | 0.86        | 0.99        | 80          | 14.0      | 2.3       | 29         | 21          |
| `sonics-spectttra-gamma-120s` | <a class="hf-button" href="https://huggingface.co/awsaf49/sonics-spectttra-gamma-120s"><img src="https://huggingface.co/front/assets/huggingface_logo-noborder.svg">HF</a>  | SpecTTTra-γ   | 120s     | 5      | 7      | 0.88 | 0.79        | 0.99        | 97          | 10.1      | 1.6       | 20        | 24          |

## 📐 Model Architecture

- **Base Model:** SpectTTTra (Spectro-Temporal Tokens Transformer)  
- **Embedding Dimension:** 384  
- **Number of Heads:** 6  
- **Number of Layers:** 12  
- **MLP Ratio:** 2.67  

## 🎶 Audio Processing

- **Sample Rate:** 16kHz  
- **FFT Size:** 2048  
- **Hop Length:** 512  
- **Mel Bands:** 128  
- **Frequency Range:** 20Hz - 8kHz  
- **Normalization:** Mean-std normalization  

## ♻️ Usage 

```python
# Install from GitHub
!pip install git+https://github.com/awsaf49/sonics.git

# Load model
from sonics import HFAudioClassifier
model = HFAudioClassifier.from_pretrained("awsaf49/sonics-spectttra-alpha-5s")
```

## 📝 Citation

```bibtex
@inproceedings{rahman2024sonics,
        title={SONICS: Synthetic Or Not - Identifying Counterfeit Songs},
        author={Rahman, Md Awsafur and Hakim, Zaber Ibn Abdul and Sarker, Najibul Haque and Paul, Bishmoy and Fattah, Shaikh Anowarul},
        booktitle={International Conference on Learning Representations (ICLR)},
        year={2025},
      }
```
