# Person Search on the PRW Dataset
### Machine Learning for Computer Vision — Assignment 2025/2026
### University of Bologna

---

**Student ID:** 0001187294  
**Full Name:** Zunaira Hasnain  
**Institutional Email:** zunaira.zunaira3@studio.unibo.it  

---

## Overview

This project implements a **two-stage Person Search pipeline** on the PRW dataset, jointly solving pedestrian detection and person re-identification. Given a query image with a bounding box around a person, the system detects and retrieves that same person across a gallery of raw, uncropped scene images.

**Pipeline summary:**
- **Detector:** Faster R-CNN (ResNet-50 + FPN-V2), pretrained on COCO and fine-tuned on PRW
- **Re-ID Model:** ConvNeXt-Base backbone + BN-Neck, trained with CE + Batch-Hard Triplet loss
- **Matching:** Cosine similarity on L2-normalized embeddings
- **Evaluation:** Official `eval_search_prw` function with `ignore_cam_id=True`

**Main results (PRW test set):**

| Model | mAP (%) | Top-1 (%) |
|:------|:-------:|:---------:|
| ConvNeXt-Base + CE+Triplet (main) | **59.55** | **88.28** |
| ResNet-50 + CE+Triplet (Ablation B) | 47.73 | 83.03 |
| ConvNeXt + ArcFace (Ablation A) | 48.21 | 87.89 |
| COCO detector only (Ablation C) | 57.24 | 87.07 |
| Oracle (GT boxes) | 61.76 | 90.33 |

---

## Project Structure

```
.
├── ML4CV_zizouff.ipynb       ← Main notebook (all code + explanations)
├── eval_function.py          ← Official evaluator (unchanged from course repo)
├── README.md                 ← This file
├── PRW_weights/              ← Model weights (see download instructions below)
│   ├── detector_final.pt
│   ├── reid_convnext_ce_triplet_40ep_best.pt
│   ├── reid_convnext_arcface_40ep_best.pt
│   ├── reid_convnext_infonce_40eps_best.pt
│   ├── reid_convnext_contrastive_40ep_best.pt
│   └── reid_resnet50_ce_triplet_40ep_best.pt
└── PRW_results/              ← Cached evaluation results (.pkl files)
    ├── baseline_convnext_ce_triplet.pkl
    ├── ablation_A_loss_results.pkl
    ├── ablation_B_backbone_results.pkl
    ├── ablation_C_final_results.pkl
    ├── all_losses_eval.pkl
    ├── oracle_gt_analysis.pkl
    └── oracle_gt_jitter_analysis.pkl
```

---

## How to Run

> **This notebook is designed to run on Google Colab (T4 GPU, 16 GB).**  
> No local setup is required.

### 1. Open the notebook on Google Colab

Upload `ML4CV_zizouff.ipynb` to Colab, or open it directly from your Google Drive.

### 2. Download the PRW dataset

The dataset is available on Kaggle. Run the following in a Colab cell:

```bash
pip install -q kaggle
kaggle datasets download -d edoardomerli/prw-person-re-identification-in-the-wild -p /content/PRW --unzip
```

You will need to configure your Kaggle API key (`kaggle.json`) first.  
Alternatively, download manually from the [Kaggle dataset page](https://www.kaggle.com/datasets/edoardomerli/prw-person-re-identification-in-the-wild) and upload to `/content/PRW`.

### 3. Download model weights and cached results

Weights and results are hosted on OneDrive and Google Drive. The notebook handles this automatically via `gdown` in **Section 1**:

```python
!pip install -q gdown
```

The notebook will download:
- `PRW_weights/` from Google Drive
- `PRW_results/` from Google Drive

**Alternatively**, download manually from the institutional OneDrive links below and upload to Colab:

| Folder | OneDrive Link |
|--------|---------------|
| PRW dataset | [OneDrive – PRW](https://liveunibo-my.sharepoint.com/my?id=%2Fpersonal%2Fzunaira_zunaira3_studio_unibo_it%2FDocuments%2FPRW&viewid=270ba629-e06a-47d5-a821-4ea87c6fefe0) |
| Model weights | [OneDrive – PRW_weights](https://liveunibo-my.sharepoint.com/my?id=%2Fpersonal%2Fzunaira_zunaira3_studio_unibo_it%2FDocuments%2FPRW_weights&viewid=270ba629-e06a-47d5-a821-4ea87c6fefe0) |
| Cached results | [OneDrive – PRW_results](https://liveunibo-my.sharepoint.com/my?id=%2Fpersonal%2Fzunaira_zunaira3_studio_unibo_it%2FDocuments%2FPRW_results&viewid=270ba629-e06a-47d5-a821-4ea87c6fefe0) |


### 4. Run the notebook

With `DO_TRAIN = False` (default), the notebook:
- **Skips training** and loads pretrained weights
- **Loads cached evaluation results** from `.pkl` files
- **Reproduces all experiments and visualizations** in ~10 minutes (or ~2 minutes if the inference cache is already present)

To retrain from scratch, set `DO_TRAIN = True` and enable per-model training flags. Full training takes approximately **20 hours on a T4 GPU**.

> **Note:** Make sure `eval_function.py` is placed inside the `PRW_weights/` folder, as the notebook imports it from there.

---

## Notebook Contents

| Section | Description |
|---------|-------------|
| 1. Setup & Imports | Environment setup, library imports, reproducibility seed |
| 2. EDA | Dataset statistics, protocol validation, distribution analysis |
| 3. PyTorch Datasets | Detection dataset, Re-ID dataset, PK sampler, augmentations |
| 4. Model Architecture | `ReIDNet` (ConvNeXt/ResNet backbone + BN-Neck), loss functions |
| 5. Training & Inference | Training loop, checkpoint selection by val mAP, feature extraction |
| 6. Ablation A — Loss | CE+Triplet vs ArcFace vs InfoNCE vs Contrastive |
| 7. Ablation B — Backbone | ConvNeXt-Base vs ResNet-50 |
| 8. Ablation C — Detector | COCO-pretrained vs PRW fine-tuned detector |
| 9. Oracle Analysis | GT-box upper bound + jitter sensitivity analysis |
| 10. IoU Validation | Detection quality sanity check across all 2057 queries |
| 11. Results Summary | Final comparison table across all experiments |
| 12. Qualitative Analysis | Ranking visualizations, failure case analysis, cross-model comparison |

---

## Evaluation Protocol

- **Function:** `eval_search_prw` (official, unchanged)
- **`ignore_cam_id=True`** as specified in the assignment FAQ
- **Queries:** 2057 bounding boxes from `query_info.txt`
- **Gallery:** 6112 test frames
- **True positive criterion:** IoU > 0.5 with ground-truth box AND identity match
- **Metrics:** mAP and Top-1 accuracy

---

## Dependencies

All dependencies are standard and available by default on Google Colab. No `requirements.txt` is needed for Colab execution. Key packages:

- `torch`, `torchvision`
- `numpy`, `scipy`, `sklearn`
- `matplotlib`, `PIL`
- `pandas`, `tqdm`
- `gdown`, `kaggle`

---

## References

References follow the order of first citation in the text.

**[1]** Zheng, L., Zhang, H., Sun, S., Chandraker, M., Yang, Y., & Tian, Q.
*Person Re-identification in the Wild.* CVPR 2017. [arXiv](https://arxiv.org/abs/1604.02531) *(PRW dataset)*

**[2]** Ren, S., He, K., Girshick, R., & Sun, J.
*Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks.* NeurIPS 2015. [arXiv](https://arxiv.org/abs/1506.01497)

**[3]** Yu, R., Chen, D., Chen, Y., et al.
*Cascade Transformers for End-to-End Person Search (COAT).* CVPR 2022. [arXiv](https://arxiv.org/abs/2203.09642)

**[4]** Xiao, T., Li, S., Wang, B., Lin, L., & Wang, X.
*Joint Detection and Identification Feature Learning for Person Search (OIM).* CVPR 2017. [arXiv](https://arxiv.org/abs/1604.01850)

**[5]** Yan, Y., Zhang, J., Ni, B., et al.
*Learning to Align for Person Search (AlignPS).* CVPR 2021. [arXiv](https://arxiv.org/abs/2103.11617)

**[6]** Liu, Z., Mao, H., Wu, C.-Y., et al.
*A ConvNet for the 2020s (ConvNeXt).* CVPR 2022. [arXiv](https://arxiv.org/abs/2201.03545)

**[7]** Luo, H., Gu, Y., Liao, X., Lai, S., & Jiang, W.
*Bag of Tricks and a Strong Baseline for Deep Person Re-Identification.* CVPR Workshops 2019. [Paper](https://openaccess.thecvf.com/content_CVPRW_2019/papers/TRMTMCT/Luo_Bag_of_Tricks_and_a_Strong_Baseline_for_Deep_Person_CVPRW_2019_paper.pdf) *(BN-Neck, strong baseline)*

**[8]** Deng, J., Guo, J., Xue, N., & Zafeiriou, S.
*ArcFace: Additive Angular Margin Loss for Deep Face Recognition.* CVPR 2019. [arXiv](https://arxiv.org/abs/1801.07698)

**[9]** Hermans, A., Beyer, L., & Leibe, B.
*In Defense of the Triplet Loss for Person Re-Identification.* arXiv 2017. [arXiv](https://arxiv.org/abs/1703.07737) *(batch-hard triplet)*

**[10]** Khosla, P., Teterwak, P., Wang, C., et al.
*Supervised Contrastive Learning.* NeurIPS 2020. [arXiv](https://arxiv.org/abs/2004.11362)

**[11]** Zhong, Z., Zheng, L., Kang, G., Li, S., & Yang, Y.
*Random Erasing Data Augmentation.* AAAI 2020. [arXiv](https://arxiv.org/abs/1708.04896)

**[12]** Hadsell, R., Chopra, S., & LeCun, Y.
*Dimensionality Reduction by Learning an Invariant Mapping.* CVPR 2006. *(contrastive loss)*

**[13]** Zheng, Z., Zheng, L., & Yang, Y.
*A Discriminatively Learned CNN Embedding for Person Re-Identification.* ACM TOMM 2017. *(ID loss + verification baseline)*

**[14]** Szegedy, C., Vanhoucke, V., Ioffe, S., et al.
*Rethinking the Inception Architecture for Computer Vision.* CVPR 2016. *(label smoothing)*
