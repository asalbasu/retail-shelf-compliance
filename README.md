x
# Retail Shelf Compliance (GroundingDINO OVD)

This project performs retail shelf compliance checking using Open-Vocabulary Detection (OVD) with GroundingDINO.
Given a shelf image + planogram JSON, it detects products, counts them, and produces a compliance report.

## Setup

Clone with submodules:
```bash
git clone --recurse-submodules https://github.com/asalbasu/retail-shelf-compliance.git
cd retail-shelf-compliance

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

python3 app.py

