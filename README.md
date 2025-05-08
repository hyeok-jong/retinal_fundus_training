# CAWR

**Code repository for**  
**"Optimizing Retinal Images Based Carotid Atherosclerosis Prediction with Explainable Foundation Models"**  


This work evaluates foundation models (OpenCLIP, DINOv2, MAE, RETFound) with various fine-tuning strategies (classifier-only, partial fine-tuning, LoRA) for predicting carotid atherosclerosis using retinal images. Performance was assessed by classification accuracy, clinical relevance (CVD survival analysis), and explainability (Grad-CAM with vessel segmentation).

## 🔧 Installation

Using Docker:

```bash
docker build -t cawr_image -f Dockerfile .
```

Or install dependencies locally:

```bash
pip install -r requirements.txt
```

## 🚀 Training Example

Run the experiment with:

```bash
bash train_lora_example.sh
```

## 📁 Contents

- `train_lora_example.sh`: reproducible training example
- Survival analysis and explainability (XAI) modules to be released in a future update
