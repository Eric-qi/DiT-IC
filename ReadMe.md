# ⚡ DiT-IC: Aligned Diffusion Transformer for Efficient Image Compression

<p align="center"> <img src="./assets/logo.png" width="400"> </p> <p align="center">

</p> <p align="center"> 
<a href="https://arxiv.org/abs/2603.13162"> <img src="https://img.shields.io/badge/ArXiv-2603.13162-b31b1b.svg"> 
</a> 
<a href="https://huggingface.co/JunqiShi/DiT-IC"> <img src="https://img.shields.io/badge/🤗%20HuggingFace-Models-yellow">
</a>
<a href="https://njuvision.github.io/DiT-IC/"> <img src="https://img.shields.io/badge/Project-Page-blue"> 
</a> 
<a href="https://pytorch.org/"> <img src="https://img.shields.io/badge/PyTorch-2.8+-ee4c2c?logo=pytorch"> 
</a> 
<img src="https://visitor-badge.laobi.icu/badge?page_id=Eric-qi.DiT-IC"/>
</p>

# 🔥 News

- **[2026/03/15]** 🎉 Code and pre-trained models are officially released!
- **[2026/03/10]** 🚀 Improved perceptual quality by scaling LoRA rank and updated training.
- **[2026/02/21]** 🏆 **DiT-IC is accepted by CVPR 2026!**

# 📖 Introduction

**DiT-IC** is a high-performance neural image compression framework that leverages the power of **Diffusion Transformers (DiT)**. By bridging latent diffusion models with standard entropy coding pipelines, DiT-IC achieves state-of-the-art perceptual quality with high efficiency.

### ✨ Key Features
* **Novel Architecture**: The first Diffusion Transformer tailored for high-fidelity image reconstruction.
* **Aligned LoRA Adaptation**: Efficient fine-tuning via proposed alignment mechanisms, significantly accelerating training process.
* **High Efficiency**: 32x latent space diffusion ensures faster inference and lower memory consumption compared to other models.
* **Deploy-Ready**: Fully compatible with standard entropy coding and easy-to-extend API and your own codecs.

<p align="center">
  <img src="./assets/overall.png" width="900">
</p>





# 📑 Table of Contents

- [Checkpoints & Performance](#-checkpoints-and-performance)
- [Installation](#-installation)
- [Quick Start: Inference](#-quick-start-inference)
- [Quick Start: Training](#-quick-start-training)
- [More Features](#-more-features)
- [BibTeX](#-bibtex)



# 📊 Checkpoints and Performance
We provide scalable model configurations by adjusting the **VAE/DiT LoRA ranks**. Higher ranks generally lead to better reconstruction quality.

👉 **Download Models**: [HuggingFace - DiT-IC Rank64/128 Checkpoints](https://huggingface.co/JunqiShi/DiT-IC)

<table>
<thead>
<tr>
<th rowspan="2" align="center">VAE/Diff.<br>Lora Rank</th>
<th rowspan="2" align="center">λ</th>

<th colspan="3" align="center">📷 Kodak</th>

<th colspan="4" align="center">🏆 CLIC</th>

<th colspan="4" align="center">🌄 DIV2K</th>
</tr>

<tr>
<th>BPP ↓</th>
<th>LPIPS ↓</th>
<th>DISTS ↓</th>

<th>BPP ↓</th>
<th>LPIPS ↓</th>
<th>DISTS ↓</th>
<th>FID ↓</th>

<th>BPP ↓</th>
<th>LPIPS ↓</th>
<th>DISTS ↓</th>
<th>FID ↓</th>
</tr>
</thead>

<tbody>

<tr>
<td align="center">32/64</td>
<td align="center">0.5</td>

<td align="center">0.080</td>
<td align="center">0.094</td>
<td align="center">0.059</td>

<td align="center">0.079</td>
<td align="center">0.067</td>
<td align="center">0.038</td>
<td align="center">3.06</td>

<td align="center">0.089</td>
<td align="center">0.084</td>
<td align="center">0.043</td>
<td align="center">7.22</td>
</tr>

<tr>
<td align="center">64/128</td>
<td align="center">0.5</td>

<td align="center">0.081</td>
<td align="center">0.091</td>
<td align="center">0.055</td>

<td align="center">0.072</td>
<td align="center">0.070</td>
<td align="center">0.034</td>
<td align="center">2.66</td>

<td align="center">0.084 </td>
<td align="center">0.086</td>
<td align="center">0.039</td>
<td align="center">6.52</td>
</tr>


</tbody>
</table>

> *Detailed performance logs for the original manuscript results (rank 32/64) and updated results (rank 64/128) can be found in the `results/` directory.*

# 🔧 Installation

### Requirements

- **Python = 3.12**
- **PyTorch = 2.8**
- **CompressAI** == 1.2.8 (⚠️ *Crucial for consistent bitrate/BPP calculation*)

Other environments may also work, but they have not been tested.

### Install dependencies
```
pip install -r requirements.txt
```

# 💻 Quick Start: Inference


## 1️⃣ Prepare Datasets

The following datasets are used for evaluation:

| Dataset          | Description                 |
| ---------------- | --------------------------- |
| [Kodak](https://r0k.us/graphics/kodak/)            | 24 natural images (768×512) |
| [DIV2K Validation](https://data.vision.ee.ethz.ch/cvl/DIV2K/) | 100 high-resolution images  |
| [CLIC 2020 Test](https://archive.compression.cc/challenge/)   | 428 high-resolution images  |

You may also evaluate the model on your **own datasets**.


## 2️⃣ Run Compression

### Option A: Inference with Merged Weights (Recommended)

The released checkpoints merge LoRA weights into the base model, which simplifies deployment and speeds up inference.
```bash
CUDA_VISIBLE_DEVICES=0 python -u compress.py \
        --config_path="configs/inference_merge.yaml" \
        --codec_path="checkpoints/0.5lrd_merge_ema.pt" \
        --img_path="/data/data/Kodak" \
        --rec_path="results/Kodak/rec/" \
        --bin_path="results/Kodak/bin/" \
        --use_merge \
        2>&1 | tee results/logs/eval_ema_Kodak_0.5lrd_$(date +%Y%m%d_%H%M%S).log
```
### Option B: Inference with Raw LoRA Checkpoints

If you trained the model yourself and the LoRA weights are still not merged, run:

```bash
CUDA_VISIBLE_DEVICES=0 python -u compress.py \
        --config_path="configs/inference.yaml" \
        --codec_path="checkpoints/0.5lrd_lora_0050000.pt" \
        --img_path="/data/data/Kodak" \
        --rec_path="results/Kodak/rec/" \
        --bin_path="results/Kodak/bin/" \
        2>&1 | tee logs/eval_kodak_0.5lrd_$(date +%Y%m%d_%H%M%S).log
```

> Bitstreams will be saved in `--bin_path`, and reconstructed images in `--rec_path`.

> Logs will be stored in the `results/logs/`.

| Argument               | Description                                  |
| ---------------------- | ---------------------------------------------|
| `--use_ema`            | Use EMA weights                              |
| `--save_img`           | Save reconstructed images                    |
| `--entropy_estimation` | Estimate bitrate without real entropy coding |


## 3️⃣ Evaluation (Optional)

To compute image quality metrics using the saved reconstruction folders:

```bash
CUDA_VISIBLE_DEVICES=0 python -m eval.evaluate \
    --recon_dir "results/Kodak/rec/" \
    --gt_dir "/data/data/Kodak" \
    2>&1 | tee logs/eval_kodak_0.5lrd_$(date +%Y%m%d_%H%M%S).log
```

# 🚗 Quick Start: Training

## 1️⃣ Prepare Training Datasets

Download the following datasets and place them in the corresponding directories:

| Dataset    | Size         |
| ---------- | ------------ |
| [LSDIR](https://ofsoundof.github.io/lsdir-data/)      | ~50K images   |
| [MLIC-Train-100K](https://huggingface.co/datasets/Whiteboat/MLIC-Train-100K) | ~100K images |



## 2️⃣ Train from Scratch

Download the required pretrained components: 
- DiT-[SANA](https://huggingface.co/Efficient-Large-Model/Sana_600M_1024px_diffusers)
- Aux Encoder-[ELIC](https://huggingface.co/JunqiShi/DiT-IC).

Place them into `SANA/` and `ELIC/weights`. 

A simplified training pipeline is provided below.  

### Stage 1 — Train w/o GAN loss
```bash
CUDA_VISIBLE_DEVICES=0,1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=. torchrun --standalone --nproc_per_node=2 --nnodes=1 train_nogan_ddp.py --config configs/train_256_nogan.yaml
```

### Stage 2 — Finetune w/ GAN loss
```bash
CUDA_VISIBLE_DEVICES=0,1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=. torchrun --standalone --nproc_per_node=2 --nnodes=1 train_ddp.py --config configs/train_256_gan.yaml
```

> `--nproc_per_node`  is equal to the number of your available GPUs. You can use single GPU training.

💡 The rate–distortion trade-off parameter λ can be either kept the same across stages or adjusted during Stage 2.

Examples:

- Stage1 λ = 2.0 → Stage2 λ = 2.0

- Stage1 λ = 0.5 → Stage2 λ = 2.0

💡 More advanced strategies, such as incorporating image–text embeddings (e.g., CLIP loss), better GAN models, or more refined training pipelines, may further improve performance or accelerate training.  Users can modify the configuration files according to their own requirements and hardware setups.


## 3️⃣ Finetune Pretrained Model

If you start from our pretrained checkpoints:

👉 **Download Models**: [HuggingFace - DiT-IC Rank64/128 Checkpoints](https://huggingface.co/JunqiShi/DiT-IC)

Then finetune at the target bitrate：
```bash
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=. torchrun --standalone --nproc_per_node=1 --nnodes=1 train_ddp.py --config configs/train_merge_256_gan.yaml
```


### 4️⃣ Merge LoRA Weights

Training checkpoints contain:

- trained LoRA weights

- trained codec parameters

For faster inference, you may merge LoRA weights into the base model:
```bash
CUDA_VISIBLE_DEVICES=0 python merge.py \
        --config_path="configs/inference.yaml" \
        --codec_path="checkpoints/0.5lrd_lora_0050000.pt"
```

You can add `--use_ema ` to enable EMA weights.

> The merged checkpoint will be saved in the same `--codec_path` directory.



# 🧩 More Features
You can modify configuration files under `configs/` to adapt the framework to different settings:

- datasets scales

- training images resolutions

- hardware constraints (e.g., memory, FLOPs)

**Planned future updates:**

- FP16 / BF16 inference

- Tiled inference



# 📖 BibTeX
If you find this project useful, please cite:
```
@inproceedings{shi2026ditic,
  title={DiT-IC: Aligned Diffusion Transformer for Efficient Image Compression},
  author={Shi Junqi, Lu Ming, Li Xingchen, Ke Anle, Zhang Ruiqi and Ma Zhan},
  booktitle={CVPR},
  year={2026}
}
```

## 🥰 Acknowledgement

Thanks to the following open-sourced codebase for their wonderful work and codebase!

- [CompressAI](https://github.com/InterDigitalInc/CompressAI)
- [diffusers](https://github.com/huggingface/diffusers)
- [SANA](https://github.com/NVlabs/Sana)
- [StableCodec](https://github.com/LuizScarlet/StableCodec)
- [LightningDiT](https://github.com/hustvl/LightningDiT)

---

<p align="center"> ⭐️ If you find this project helpful, please give it a star! ⭐️ </p>
