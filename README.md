# 
# 🌍 M2M100 Fine-Tuned for Technical Domain Translation

![Hugging Face](https://img.shields.io/badge/HuggingFace-Model-orange)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Transformers](https://img.shields.io/badge/Transformers-🤗-yellow)
![License](https://img.shields.io/badge/License-Apache%202.0-green)

---

## 📘 Overview
This repository contains the complete workflow for **fine-tuning Meta’s M2M100 multilingual translation model** on a **technical-domain dataset** (English → multiple languages).  
It includes training and inference notebooks, the dataset used, and a link to the fine-tuned model hosted on Hugging Face.

> **Fine-tuned model:** [Bilal7072/m2m100-finetuned-tech](https://huggingface.co/Bilal7072/m2m100-finetuned-tech)

---

## 📂 Repository Structure

| File | Description |
|------|--------------|
| `M2M100FineTuned.ipynb` | Jupyter notebook for **fine-tuning** the base M2M100 model using the provided dataset. It includes preprocessing, tokenization, model training, and evaluation steps. |
| `inferencem2m100.ipynb` | Notebook demonstrating **inference** using the fine-tuned model. |
| `en_to_multi_dataset.csv` | Custom bilingual dataset used for training (English to multiple target languages). |
| `LICENSE` | Apache-2.0 License for open usage and distribution. |

---

## 🧠 About the Model

The fine-tuned model is based on the **[facebook/m2m100_418M](https://huggingface.co/facebook/m2m100_418M)** checkpoint — a multilingual model supporting 100+ languages.

The model `Bilal7072/m2m100-finetuned-tech` specializes in translating **technical text** (software, engineering, scientific writing, etc.) from **English** to selected target languages.

### 🔗 Access the Model
You can directly use it via Hugging Face:
```python
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

model_name = "Bilal7072/m2m100-finetuned-tech"
tokenizer = M2M100Tokenizer.from_pretrained(model_name)
model = M2M100ForConditionalGeneration.from_pretrained(model_name)

# Example: English → Urdu (replace 'ur' with your desired language code)
tokenizer.src_lang = "en"
encoded = tokenizer("Machine learning enhances data-driven decision making.", return_tensors="pt")
generated_tokens = model.generate(**encoded, forced_bos_token_id=tokenizer.get_lang_id("ur"))
translation = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
print("Translation:", translation)


🧾 Dataset
Column	Description
source	English technical text
target	Translated text in target language
lang (optional)	Target language code

Dataset: en_to_multi_dataset.csv — designed to help the model learn technical term mappings and structure.

⚙️ Fine-Tuning Steps (inside M2M100FineTuned.ipynb)

Load base model and tokenizer

from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
model_name = "facebook/m2m100_418M"
tokenizer = M2M100Tokenizer.from_pretrained(model_name)
model = M2M100ForConditionalGeneration.from_pretrained(model_name)


Preprocess & tokenize dataset
Load CSV, split into train/test, and encode using tokenizer.

Train using Hugging Face Trainer

from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

training_args = Seq2SeqTrainingArguments(
    output_dir="./results",
    evaluation_strategy="steps",
    learning_rate=5e-5,
    per_device_train_batch_size=4,
    num_train_epochs=3,
    save_total_limit=2,
    predict_with_generate=True
)


Evaluate model
Use BLEU or sacreBLEU for translation quality evaluation.

Save fine-tuned checkpoint

model.save_pretrained("m2m100-finetuned-tech")
tokenizer.save_pretrained("m2m100-finetuned-tech")

🚀 Inference Demo (inferencem2m100.ipynb)

This notebook shows:

How to load the fine-tuned model (Bilal7072/m2m100-finetuned-tech)

Generate translations for custom input text

Compare predictions across languages

🧩 Requirements

Install dependencies:

pip install torch transformers sentencepiece sacrebleu pandas jupyter


Optionally:

pip install datasets accelerate

📈 Results (example)
Metric	Score
BLEU (avg)	~35–40 (depends on target language & dataset size)

Results vary with dataset size, domain coverage, and fine-tuning parameters.
The model performs notably better on technical English text compared to general domains.

🔮 Future Enhancements

Expand dataset to cover more technical subdomains (e.g., mechanical, biomedical, or software manuals).

Add reverse translation (target → English).

Deploy the model as an API using FastAPI or Gradio.

Experiment with larger checkpoints (m2m100_1.2B) for improved fluency.

🧑‍💻 Author

Bilal Ahmed
AI/ML Engineer | Deep Learning Enthusiast

🌐 Hugging Face

💼 LinkedIn https://www.linkedin.com/in/bilal-ahmed-392973268/

💻 GitHub https://github.com/BilalAhmed7072/

📜 License

This project is licensed under the Apache-2.0 License.
You’re free to use, modify, and distribute this work with attribution.
