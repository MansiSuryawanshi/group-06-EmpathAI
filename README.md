# 🧠 EmpathAI: Mental Health Signal Detection Engine

> *"Sometimes the most powerful AI isn't the one that talks the most—it's the one that listens the best."*

Welcome to **EmpathAI**, an intelligent detection system that reads between the lines of online mental health support communities. We decode what people really need—whether it's crisis intervention, emotional validation, or a gentle nudge toward professional help.

## 🎯 What Does This Do?

Imagine thousands of posts flowing through mental health forums every day. Some are just venting. Some are critical cries for help. This system is a **complete end-to-end mental health assistant pipeline** that:

1. **Classifies intent** behind each post (9 categories: Critical Risk, Mental Distress, Seeking Help, etc.)
2. **Assesses concern levels** (Low, Medium, High risk)
3. **Detects crises** using keyword-based systems
4. **Retrieves relevant support knowledge** via FAISS vector search
5. **Generates empathetic responses** using Flan-T5 with RAG
6. **Ensures safety** with disclaimers and helpline resources
7. **Logs interactions** for monitoring and improvement

We've trained and benchmarked **four powerful AI models**:
- **🚀 MiniLM Lightning** - Fast, efficient, production-ready
- **⚡ DistilRoBERTa Turbo** - Lean & mean with LoRA fine-tuning  
- **🔥 RoBERTa Pro** - Heavy-hitter performance
- **💎 DeBERTa-v3 Elite** - Second-best accuracy with efficiency

All orchestrated through a **YAML-driven pipeline** so you never touch code unless you want to.

---

## 📁 The Blueprint

```
configs/                          # 🎛️ Model & data configuration hub
├── baseline_minilm.yaml         # MiniLM settings
├── distilroberta_lora.yaml      # DistilRoBERTa config
├── roberta_lora.yaml            # RoBERTa config
└── data.yaml                    # Universal data paths

data/                            # 📊 Data warehouse
├── raw/                         # Original posts (untouched)
├── processed/                   # Cleaned & tagged datasets
├── llm_tagged/                  # BART zero-shot tagged data
└── splits/                      # Train/Val/Test stratified splits

models/                          # 🧠 Model training scripts & artifacts
├── baseline_minilm_lr.py        # MiniLM intent trainer
├── baseline_minilm_lr_concern.py # MiniLM concern trainer
├── distilroberta_lora.py        # DistilRoBERTa intent trainer
├── distilroberta_lora_concern.py # DistilRoBERTa concern trainer
├── roberta_lora.py              # RoBERTa intent trainer
├── roberta_lora_concern.py      # RoBERTa concern trainer
├── empath_model/                # DeBERTa-v3 adapter artifacts
└── helper.py                    # Shared utilities & magic

scripts/                         # 🔧 Pipeline toolbox
├── create_splits.py             # Generate stratified splits
├── llm_tagging.py               # BART zero-shot classification
├── tag_summary.py               # Analytics & distribution stats
├── tag_rules_assign.py          # Rule-based tagging fallback
├── kb_build.py                  # Knowledge base construction
├── kb_search.py                 # Vector search engine
└── rag_generate.py              # RAG response generation

results/                         # 🏆 Where the magic happens
├── runs/                        # Model checkpoints, logs, predictions
└── final/                       # Best model artifacts

logs/                            # 📝 Interaction history
├── general_*.jsonl              # Regular conversations
└── high_risk_*.jsonl            # Crisis interactions (flagged for review)
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- CUDA 11.8+ (optional, CPU works but slower)
- ~15GB disk space for models

### Installation

**Option 1: Virtual Environment (Recommended)**
```bash
python -m venv .venv
source .venv/bin/activate      # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**Option 2: Conda**
```bash
conda create --name empathAI python=3.10
conda activate empathAI
pip install -r requirements.txt
```

---

## ⚙️ Configuration

Everything lives in `configs/data.yaml`. Change paths once, run everywhere:

```yaml
data:
  raw_path: data/raw/empath_ai_data_w-intent.csv
  processed_path: data/processed/
  splits_dir: data/llm_tagged/splits
  results_dir: results/

training:
  seed: 42
  test_size: 0.15
  val_size: 0.15
```

**One config file. All your paths. Zero headaches.**

---

## 🏷️ Data Pipeline

### Phase 1: Tagging (Already Done ✅)
Your data has been:
- Cleaned & normalized
- Tagged with **9 mental health intents**: Critical Risk, Mental Distress, Seeking Help, Mood Tracking, Positive Coping, Maladaptive Coping, Progress Update, Cause of Distress, Miscellaneous
- Categorized into **3 concern levels**: Low, Medium, High

**Key files:**
- `data/llm_tagged/splits/train.csv` - 3,694 training examples
- `data/llm_tagged/splits/val.csv` - 792 validation examples  
- `data/llm_tagged/splits/test.csv` - 792 test examples

### Phase 2: Re-tagging (If Needed)
```bash
# Zero-shot tagging with BART
python scripts/llm_tagging.py --input data/raw/posts.csv --output data/processed/tagged.csv

# Rule-based fallback
python scripts/tag_rules_assign.py

# Generate statistics
python scripts/tag_summary.py --input data/processed/tagged.csv
```

---

## 🎓 Training Models

Each model is **drop-in ready** with YAML configs:

### MiniLM (Fast & Production-Ready)
```bash
python models/baseline_minilm_lr.py --config configs/baseline_minilm.yaml
python models/baseline_minilm_lr_concern.py --config configs/baseline_minilm.yaml
```
**Results:** ~0.69 macro F1 on intent, ~0.71 on concern (< 1 min training)

### DistilRoBERTa (Balanced Power)
```bash
python models/distilroberta_lora.py --config configs/distilroberta_lora.yaml
python models/distilroberta_lora_concern.py --config configs/distilroberta_lora_concern.yaml
```
**Results:** ~0.75+ F1 (20-30 min training with LoRA)

### RoBERTa (Maximum Accuracy)
```bash
python models/roberta_lora.py --config configs/roberta_lora.yaml
python models/roberta_lora_concern.py --config configs/roberta_lora_concern.yaml
```
**Results:** ~0.77+ F1 (longer training, full power)

### DeBERTa-v3 (Adapter / LoRA)
```bash
# DeBERTa-v3 model artifacts are available in models/empath_model/
# Training and fine-tuning details are stored with adapter configs and saved weights.
```
**Results:** ~0.726 Macro F1, ~0.770 Micro F1 (balanced performance)

---

## 📊 Understanding Your Results

Every training run generates:

```
results/runs/
└── baseline_minilm_lr_20260503_170203/
    ├── pytorch_model.bin          # Saved weights
    ├── training_log.json          # Training metrics over time
    ├── eval_results.json          # Validation metrics
    ├── test_results.json          # Test predictions & scores
    ├── confusion_matrix.png       # Visual performance analysis
    └── hyperparams.yaml           # Exact config used
```

**Key Metrics:**
- **Macro F1:** Unbiased average across all classes
- **Micro F1:** Overall accuracy-weighted metric
- **Per-label F1:** How well each category was detected
- **Confusion Matrix:** See exactly where models struggle

---

## 🔍 Advanced: Knowledge Base & RAG

Build a searchable knowledge base from tagged data:

```bash
# 1. Encode posts into embeddings
python scripts/kb_encode_index.py --input data/processed/kb.csv

# 2. Search similar posts (Retrieval)
python scripts/kb_search.py --query "feeling suicidal" --top_k 5

# 3. Generate empathetic responses (Generation)
python scripts/rag_generate.py --input splits/test_gold.jsonl
```

Outputs contextualized, retrieval-augmented responses for support conversations.

---

## 🧠 Final Model Performance (Updated)

We trained and evaluated multiple models on the BART-augmented dataset.

### 📊 Intent Classification Results

| Rank | Model                        | Macro-F1  | Micro-F1  |
| ---- | ---------------------------- | --------- | --------- |
| 🥇 1 | RoBERTa-Large + LoRA         | **0.774** | **0.820** |
| 🥈 2 | DeBERTa-v3 + LoRA            | **0.726** | **0.770** |
| 🥉 3 | MiniLM + Logistic Regression | 0.696     | 0.733     |
| 4    | DistilRoBERTa + LoRA         | 0.646     | 0.642     |

👉 The DeBERTa-v3 LoRA model is our **second-best model**, offering a strong balance between performance and efficiency.

---

### 📈 Final Test Metrics (DeBERTa-v3 LoRA)

* **Macro F1:** 0.726
* **Micro F1:** 0.770
* **PR-AUC (Macro):** ~0.786
* **Best Threshold:** 0.45

---

## 🚀 From Midterm → Final System

| Component           | Midterm                  | Finals                             |
| ------------------- | ------------------------ | ---------------------------------- |
| Labeling            | Human + weak supervision | BART zero-shot re-labeling         |
| Intent F1           | 0.70                     | **0.774 (+10.6%)**                 |
| Concern F1          | 0.78                     | **0.786 (+0.8%)**                  |
| Knowledge Base      | ❌ Not built              | ✅ 50K+ snippets (FAISS indexed)    |
| Response Generation | ❌ Not built              | ✅ Flan-T5 (grounded RAG)           |
| Crisis Detection    | ❌ Not built              | ✅ 3-tier keyword system            |
| Safety Checks       | ❌ Not built              | ✅ Violation detection + fallback   |
| Logging             | ❌ Not built              | ✅ JSONL logs (general + high-risk) |
| Evaluation          | Basic metrics            | + Relevance, Grounding, Safety     |

---

## 🧩 Full Pipeline (Final System)

The final system is no longer just classification.

It is a **complete end-to-end mental health assistant pipeline**:

1. **Post Classification**

   * Intent prediction
   * Concern level detection

2. **Crisis Detection**

   * Keyword-based 3-tier system
   * Flags high-risk posts

3. **Knowledge Retrieval (RAG)**

   * FAISS vector search over 50K+ entries
   * Retrieves relevant support snippets

4. **Response Generation**

   * Flan-T5 generates grounded responses
   * Uses retrieved context

5. **Safety Layer**

   * Adds disclaimers
   * Provides helpline resources

6. **Logging System**

   * General logs
   * High-risk interaction logs

---

## 💡 Summary

* Midterm: *"We can classify posts."*
* Final: *"We can classify, retrieve, generate, detect crises, and ensure safe responses."*

This transforms the project from a **model** into a **real-world deployable system**.

---

## 🛠️ Development & Debugging

### Check data integrity:
```bash
python -c "import pandas as pd; df = pd.read_csv('data/llm_tagged/splits/train.csv'); print(f'Shape: {df.shape}, Cols: {df.columns.tolist()}')"
```

### Validate all Python files:
```bash
python -m compileall -q .
```

### Run a single model debug:
```python
python -c "
from models.helper import read_split_csv
from pathlib import Path
df = read_split_csv(Path('data/llm_tagged/splits/train.csv'))
print(f'Loaded {len(df)} examples')
print(df.head())
"
```

---

## 📈 Model Comparison (BART-Augmented Dataset)

| Rank | Model                        | Macro-F1  | Micro-F1  | Speed | Memory | Best For |
| ---- | ---------------------------- | --------- | --------- |-------|--------|----------|
| 🥇 1 | RoBERTa-Large + LoRA         | **0.774** | **0.820** | ⚡ | 🧠🧠🧠 | Maximum Accuracy |
| 🥈 2 | DeBERTa-v3 + LoRA            | **0.726** | **0.770** | ⚡⚡ | 🧠🧠 | Balanced Performance |
| 🥉 3 | MiniLM + Logistic Regression | 0.696     | 0.733     | ⚡⚡⚡ | 🧠 | Production, Real-time |
| 4    | DistilRoBERTa + LoRA         | 0.646     | 0.642     | ⚡⚡ | 🧠🧠 | Efficient Fine-tuning |

---

## 🎨 Project Structure Philosophy

**Why YAML?** Configuration as code. Change hyperparameters without touching Python.

**Why modular scripts?** Each stage is independent. Swap data, swap models, keep the pipeline.

**Why multiple models?** Different use cases need different tradeoffs. Fast? Accurate? Balanced? Pick one.

---

## 🤝 Contributing

Found a bug? Want to add a model? Cool! Just make sure:
- ✅ Update `configs/` with new model settings
- ✅ Add model to `models/` with clear logging
- ✅ Test with: `python -m compileall -q .`
- ✅ Update this README

---

## 📞 Support & Issues

- **Data issues?** Check `configs/data.yaml` paths
- **Model errors?** Look at `results/runs/<run_id>/training_log.json`
- **Slow training?** Reduce batch size in config YAML
- **Out of memory?** Switch to MiniLM or reduce max_seq_length

---

## 📜 License & Citation

This project is built with ❤️ for mental health support systems.

**Core Dependencies:**
- Hugging Face Transformers (BERT, RoBERTa, BART)
- Sentence Transformers (embeddings)
- PyTorch (deep learning)
- PEFT (Parameter-Efficient Fine-Tuning)

---

## 🌟 What Makes This Special?

✨ **Not just a model.** A complete pipeline from raw data → predictions → insights.  
✨ **Production-ready.** YAML configs mean zero code changes for new datasets.  
✨ **Transparent.** Every training run logged, every metric saved, every decision visible.  
✨ **Ethical.** Built with mental health support in mind, not profit.

---

**Happy detecting! 🚀🧠💚**
