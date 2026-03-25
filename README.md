# WebClasSeg-25 Model Evaluation Repository

This repository provides a unified framework for training and evaluating multiple deep learning models on the **WebClasSeg-25** dataset for webpage segmentation and classification.

## Overview

The goal of this project is to benchmark different model architectures on the task of webpage segmentation. The models operate either on:

* **Rendered webpage screenshots**, or
* **HTML document representations**

Each model produces **segmentations** of a webpage and assigns **class labels** to those segments, enabling both structural and semantic understanding of web content.

## Supported Models

The repository includes implementations for the following models:

* **ResNet18**
* **SAM2 (Segment Anything Model)**
* **YOLO11**
* **RoBERTa**
* **CodeBERT**

### Special Case: SAM2

Unlike the other models, **SAM2** only generates segmentations but does not classify them. Therefore, it is extended with an additional **ResNet-based classifier** that assigns labels to the generated segments.

* SAM2 handles segmentation
* A dedicated ResNet model performs classification
* This ResNet is **separate** from the standalone ResNet18 model used elsewhere in the repository

## Repository Structure

Each model has its own directory with a consistent structure:

```
model_name/
│
├── preprocess.py   # Dataset preparation (if required)
├── train.py        # Model training
├── predict.py      # Inference on test set
├── requirements.txt
```

### SAM2 Structure

The SAM2 setup differs slightly:

```
sam2/
│
├── preprocess_sam.py
├── preprocess_resnet.py
├── train.sh              # Handles SAM2 training pipeline
│
└── resnet/
    ├── train.py
    ├── predict.py
    └── ...
```

## Workflow

For most models, the workflow follows three steps:

1. **Preprocessing**

   * Converts the WebClasSeg-25 dataset into the required input format

2. **Training**

   * Trains the model on the prepared dataset

3. **Prediction**

   * Generates segmentations and assigns classes on the test set

Each model can be run independently.

## Environment Setup

Each model directory includes its own `requirements.txt`.

We recommend creating **separate virtual environments** per model:

```bash
python -m venv venv_model
source venv_model/bin/activate
pip install -r requirements.txt
```

## Dataset

This repository uses the **WebClasSeg-25** dataset, which provides:

* Functional segmentation labels (e.g., header, footer, navigation)
* Digital maturity classification labels

The dataset supports both:

* Visual (screenshot-based) segmentation
* HTML-based segmentation

## Citation

If you use this repository, please cite:

```
@InProceedings{10.1007/978-3-032-21300-6_14,
author="Saxer, Jasmin
and Gerber, Jonathan
and Weiler, Andreas
and Grossniklaus, Michael",
title="Website Segmentation Beyond Structure: A Benchmark on Functional and Digital Maturity Classes",
booktitle="Advances in Information Retrieval",
year="2026",
publisher="Springer Nature Switzerland",
pages="222--236"
}
```

If you use the dataset, please cite:

```
@inproceedings{gerber2025webclasseg,
  title={Webclasseg-25: A dual-classified webpage segmentation dataset-integrating functional and maturity-based analysis},
  author={Gerber, Jonathan and Saxer, Jasmin and Rabishokr, Kimia and Kreiner, Bruno and Weiler, Andreas},
  booktitle={Proceedings of the 48th International ACM SIGIR Conference on Research and Development in Information Retrieval},
  pages={3792--3801},
  year={2025}
}
```

## Notes

* Models are designed to be modular and comparable
* The pipeline is consistent across models where possible
* SAM2 requires additional handling due to its segmentation-only nature

## Future Work

* Unified evaluation metrics across models
* Additional transformer-based architectures
* Improved HTML-aware segmentation approaches

---

For questions or contributions, feel free to open an issue or pull request.
