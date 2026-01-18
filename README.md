# Retail Shelf Compliance (GroundingDINO OVD)

This project checks retail shelf compliance using **Open-Vocabulary Detection (OVD)** with **GroundingDINO**.
Given a shelf image and a planogram JSON, it detects products, counts them, and produces a compliance report.

## What this project does
- Detects products on a retail shelf image using GroundingDINO (OVD)
- Matches detections to the expected planogram (JSON)
- Counts items and generates a simple compliance result (OK / missing / extra)

## Model note (SAM vs GroundingDINO)
The original pipeline idea was to use:
- GroundingDINO for detection (finding products)
- SAM for segmentation (precise masks)

In our implementation, **SAM was too heavy and crashed on our machine**, so we used **only GroundingDINO**.
This made the system lighter and more stable, but bounding boxes can be less precise than segmentation.

## Setup
Clone the repo with submodules (GroundingDINO is a submodule):

```bash
git clone --recurse-submodules https://github.com/asalbasu/retail-shelf-compliance.git
cd retail-shelf-compliance
