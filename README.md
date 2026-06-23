# RF-BoxCAM: Receptive Field Guided High Resolution Visual Explanations

![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)
![YOLO](https://img.shields.io/badge/YOLO-Ultralytics-blue)
![KES 2026](https://img.shields.io/badge/Accepted-KES_2026-success)

Official PyTorch implementation of **RF-BoxCAM**, a white-box Explainable AI (XAI) method tailored for deep object detection architectures. 

This repository contains the code to generate high-resolution, object-centric visual explanations for YOLO models, specifically designed to address the challenges of detecting small, densely packed objects in aerial imagery. RF-BoxCAM operates in a single forward/backward pass, achieving the spatial faithfulness of perturbation-based black-box methods (like D-CLOSE) while operating up to **20x faster**.

## Overview

Existing XAI techniques often fail when applied to modern multi-scale object detectors (like YOLOv8/v11), either producing spatially coarse heatmaps or requiring massive computational overhead. RF-BoxCAM solves this by:
1. **Pre-NMS Aggregation:** Capturing all valid internal evidence by isolating the cluster of bounding boxes targeting a specific object before Non-Maximum Suppression.
2. **Active Grid Cell Isolation:** Using a sparse activation mechanism to only compute gradients for the specific network cells responsible for the prediction.
3. **Receptive Field Reconstruction:** Reconstructing exact, pixel-level input receptive fields to ensure precise spatial alignment without naive upsampling.

## Installation

Clone the repository and install the required dependencies:

```bash
git clone [https://github.com/malekBHS/RF-BoxCAM.git](https://github.com/malekBHS/RF-BoxCAM.git)
cd RF-BoxCAM
pip install -r requirements.txt
