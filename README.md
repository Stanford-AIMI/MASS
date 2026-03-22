<div align="center">

# MASS: Learning Generalizable 3D Medical Image Representations from Mask-Guided Self-Supervision

**CVPR 2026**

Yunhe Gao, Yabin Zhang, Chong Wang, Jiaming Liu, Maya Varma, Jean-Benoit Delbrouck, Akshay Chaudhari, Curtis Langlotz

**Stanford University**

[![arXiv](https://img.shields.io/badge/arXiv-2603.13660-b31b1b.svg)](https://arxiv.org/abs/2603.13660)
[![Project Page](https://img.shields.io/badge/Project-Page-green)](https://yhygao.github.io/MASS_page/)
[![MiT License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

</div>

<div align="center">
  <img src="figs/framework.jpg" width="90%">
</div>

## 🔥 News
- **[2026/02]** MASS is accepted to **CVPR 2026**!
- **[2026/03]** Paper is available on [arXiv](https://arxiv.org/abs/2603.13660).
- **Code & pretrained models coming soon!** Stay tuned. ⭐ Star this repo to get notified.

## 💡 Highlights

| | |
|:---:|:---|
| 🚫 **Zero Annotation** | Uses in-context segmentation as the pretext task with auto-generated class-agnostic masks. No expert annotation cost — yet comparable to supervised pretraining and **substantially outperforms** reconstruction/contrastive SSL. |
| 🧠 **In-Context Knowledge** | Like LLMs acquiring language understanding through pretraining, MASS acquires medical knowledge (anatomy, morphology, spatial relationships) directly from pretraining — enabling **few-shot segmentation out of the box**. |
| 📈 **Scalable** | Effective from **20 scans** to **5K multi-modal** CT, MRI, and PET volumes, with consistent gains as data scale and diversity increase. |
| 🎯 **Few-Shot Power** | With only **20–40%** labeled data, MASS matches full supervision and outperforms all prior SSL methods by **>20 Dice points** in low-data regimes. |
| 🔄 **Broad Generalization** | Frozen-encoder classification on **unseen pathologies** matches fully supervised training with thousands of labeled samples — knowledge transfers **beyond segmentation**. |

## Overview

**MASS** (MAsk-guided Self-Supervised learning) is a self-supervised pretraining framework for 3D medical imaging that learns general-purpose representations without expert annotations.

**Core idea:** Automatically generated class-agnostic masks provide sufficient structural supervision for learning semantically rich representations. MASS formulates pretraining as in-context segmentation across thousands of diverse mask proposals — forcing the model to learn what semantically defines medical structures through appearance, shape, spatial context, and anatomical relationships.

**The framework consists of two stages:**
1. **Annotation-free mask generation**: Any class-agnostic mask generator produces 3D region proposals from unlabeled volumes. We default to SAM2, which samples 2D slices, applies automatic segmentation, and propagates masks through volumes.
2. **Mask-guided self-supervised learning**: For each step, sample an image with its auto masks, create two augmented views (reference + query), extract a task embedding from the reference, and predict the corresponding region in the query.

**Downstream deployment modes:**
- **Training-free in-context segmentation** — directly segment novel structures given reference examples, no finetuning needed
- **Task-specific finetuning** — finetune as a standard segmentation model with fixed classes
- **Feature extraction for classification** — use the frozen encoder as a general-purpose feature extractor



## 🚀 Getting Started

> **⏳ Code and pretrained models are coming soon.** We are cleaning the codebase for public release. Star ⭐ this repo to get notified!

The release will include:
- [ ] Pretrained model weights (ResUNet, large-scale multi-modal)
- [ ] SAM2 mask generation pipeline for 3D medical images
- [ ] Self-supervised pretraining code
- [ ] Few-shot segmentation finetuning code
- [ ] In-context inference code
- [ ] Frozen-encoder classification code
- [ ] Data preprocessing scripts



## Citation

If you find MASS useful in your research, please consider citing:

```bibtex
@article{gao2026learning,
  title={Learning Generalizable 3D Medical Image Representations from Mask-Guided Self-Supervision},
  author={Gao, Yunhe and Zhang, Yabin and Wang, Chong and Liu, Jiaming and Varma, Maya and Delbrouck, Jean-Benoit and Chaudhari, Akshay and Langlotz, Curtis},
  journal={arXiv preprint arXiv:2603.13660},
  year={2026}
}
```


## License

This project is released under the [MIT License](LICENSE).