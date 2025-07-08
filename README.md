# Preference Optimization for Vision-Language Models with ORPO

## 🧠 Project Overview

This project investigates **preference optimization techniques** for **Vision-Language Models (VLMs)** using **Odds Ratio Preference Optimization (ORPO)**. It explores how human-like preferences can be instilled in multi-modal AI systems to align their outputs with desirable behaviors, such as accuracy and informativeness.

We use ORPO to train and evaluate VLMs on prompt-response datasets, comparing chosen vs. rejected outputs. Our experiments are tracked using Weights & Biases (W&B), highlighting key trade-offs and performance gains in response quality.

---

## ⚙️ Installation & Usage

### 1. Clone the Repository
```bash
git https://github.com/Zain-Haider-ML/Preference-Optimization-for-Vision-Language-Models-with-ORPO.git
cd Preference-Optimization-for-Vision-Language-Models-with-ORPO
```

### 2. Create Environment
I recommend using Python 3.10+ and a virtual environment.
```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install Requirements
```bash
pip install -r requirements.txt
```

Dependencies include:
- `transformers`
- `datasets`
- `torch`
- `trl`
- `fsspec`
- `wandb`

### 4. Run the Notebook
Open the Jupyter notebook to execute and visualize the preference optimization pipeline:
```bash
jupyter notebook Preference_Optimization_for_Vision_Language_Models_with_ORPO.ipynb
```

---

## 📘 Methodology

### ORPO (Odds Ratio Preference Optimization)

This notebook implements the ORPO training method—a simple, reference-free approach to preference alignment that fine-tunes models using odds ratio-based supervision. ORPO aligns model outputs with human preferences without requiring a separate reward model or reinforcement learning phase.
Key components include:
- **Dataset**: Paired prompts with "chosen" and "rejected" responses, along with corresponding images.
- **Odds Ratio-Based Preference Modeling**: Encodes human preferences by applying a small penalty to disfavored responses during supervised fine-tuning.
- **Loss Function**: A monolithic objective that combines the standard negative log-likelihood (NLL) loss with an odds ratio-based preference loss, allowing the model to distinguish between preferred and non-preferred outputs effectively.
- **Training Loop**: Fine-tunes a base vision-language model (e.g., llava-hf/llava-v1.6-mistral-7b-hf, gemma-3-12b-pt, SmolVLM-256M-Instruct or similar) using preference pairs in a single-stage supervised setup—no separate reward modeling or reinforcement learning required.

> ORPO has been empirically shown to outperform larger models in benchmark evaluations like AlpacaEval, IFEval, and MT-Bench, while remaining computationally efficient and scalable across model sizes.

### 🏗️ Architecture 
- **Model**: SmolVLM-256M-Instruct, a compact Vision-Language Transformer capable of image-grounded generation..
- **Framework**: Hugging Face Transformers for model loading and training. PEFT (Parameter-Efficient Fine-Tuning) for enabling adapter-based training (optional).
- **Training Strategy**: Frozen base model with selectively trainable heads/layers. Also supports full fine-tuning or efficient alternatives like LoRA.

⚠️ Note: In this project, full fine-tuning was used on SmolVLM due to its small size. For larger models, use adapter-based training (e.g., LoRA, QLoRA) for better efficiency.

---

## 📊 Key Results & Findings

Key metrics tracked during ORPO training (based on W&B report):

| Metric                         | Observation Range     |
|-------------------------------|------------------------|
| `eval/rewards/chosen`         | ↑ From -0.28 to -0.20 |
| `eval/rewards/rejected`       | ↑ From -0.28 to -0.20 |
| `eval/rewards/margins`        | ↑ Up to 0.002         |
| `train/loss`                  | ↓ Around 2.2 - 3.7     |
| `train/nll_loss`              | ↓ Around 2.1 - 3.7     |
| `train/rewards/accuracies`    | ↑ ~0.51               |
| `eval/steps_per_second`       | ~1.925 - 1.94          |
| `eval/samples_per_second`     | ~15.3 - 15.54          |

These show that the model effectively learned to distinguish between preferred and dispreferred responses and improved accuracy in aligning with human preferences.

---

## 📈 Visualizations

The following visualizations were generated from the training report (via [Weights & Biases dashboard](https://api.wandb.ai/links/zaynhyder15-brandenburgische-technische-universit-t-cott/lhwdtu0l)):

- **Reward Margin Improvement**: Model's ability to prefer better responses improved steadily.
- **Loss Curves**: NLL and preference loss show gradual convergence.
- **GPU Utilization**: Training was efficient and stable across runs.

---

## 📚 References & Acknowledgments

- ORPO Concept: [Odds Ratio Preference Optimization (ORPO)](https://huggingface.co/docs/trl/main/en/orpo_trainer)
- HuggingFace Transformers: https://github.com/huggingface/transformers
- LoRA & PEFT: https://github.com/huggingface/peft
- Visualization & Tracking: https://wandb.ai/


## 🔁 Reproducibility & Future Work

- ✅ All training parameters, model configuration, and logging are available in the notebook.
- 📊 A PDF file (Preference Optimization report _ huggingface.pdf) containing W&B metric visualizations (training/validation loss, reward margins, sample/sec, GPU stats, etc.) is included in this repository for reference.
- Future work may involve:
  - Scaling ORPO training to larger multimodal models (e.g., LLaVA-1.6-7B, Flamingo, Idefics2)
  - Extending preference learning to multilingual or domain-specific vision-language datasets.
  - Integrating visual grounding into reward modeling or introducing synthetic preference labels

---

## 📬 Contact

For questions or collaborations, feel free to open an issue or reach out via GitHub.
