## RUN ON GOOGLE COLAB
import pandas as pd
import torch
import gc
import os
import shutil
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig
from datasets import Dataset
from google.colab import files

# ZMIANA 1: Nowe 5 kategorii, które wybrałeś
CATEGORIES_DESCRIPTION = """
- rec.autos (motoryzacja, samochody, części, transport)
- sci.space (kosmos, astronomia, technologia kosmiczna)
- comp.graphics (grafika komputerowa, formaty obrazów, renderowanie)
- misc.forsale (ogłoszenia, sprzedaż, kupno, oferty)
- sci.med (medycyna, zdrowie, leki, opieka zdrowotna)
""".strip()

MODELS_TO_TRAIN = {
    "phi3":     "unsloth/Phi-3-mini-4k-instruct-bnb-4bit",
    "llama3.1": "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
    "gemma3":   "unsloth/gemma-3-1b-it-bnb-4bit",
}

PROMPT_TEMPLATE = """
### Instruction:
Sklasyfikuj poniższą wiadomość e-mail do jednej z kategorii: {categories}.
Odpowiedz tylko nazwą kategorii.

### Input:
{text}

### Response:
{label}
"""

MAX_SEQ_LENGTH = 2048
LORA_RANK = 16

def build_dataset(csv_path: str) -> Dataset:
    df = pd.read_csv(csv_path)

    def format_examples(examples):
        return {
            "text": [
                PROMPT_TEMPLATE.format(
                    categories=CATEGORIES_DESCRIPTION,
                    text=str(text)[:1500],
                    label=str(label).strip()
                )
                for text, label in zip(examples["clean_text"], examples["label"])
            ]
        }

    ds = Dataset.from_pandas(df)
    return ds.map(format_examples, batched=True)

print("Wczytywanie datasetu treningowego...")
# ZMIANA 2: Ładujemy tylko i wyłącznie zbiór treningowy!
dataset = build_dataset("train_data.csv")
print(f"Załadowano {len(dataset)} przykładów do douczania.")

# ============================================================
# CELL 4 - Trening + eksport GGUF (wszystko w jednej pętli)
# ============================================================
def train_and_export(ollama_name: str, hf_name: str) -> None:
    print(f"\n{'='*55}")
    print(f"  [{1}/2] TRENING: {ollama_name}")
    print(f"{'='*55}\n")

    # --- Ładowanie modelu ---
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=hf_name,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
    )

    # --- LoRA ---
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_RANK,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=LORA_RANK,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    # --- Trening ---
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        dataset_num_proc=2,
        packing=False,
        args=SFTConfig(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=5,
            num_train_epochs=1,
            learning_rate=2e-4,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=10,
            optim="adamw_8bit",
            output_dir=f"outputs_{ollama_name}",
            report_to="none",
            dataset_text_field="text",
        ),
    )
    trainer.train()
    del trainer
    gc.collect()

    # --- Eksport GGUF Q4_K_M ---
    print(f"\n  [{2}/2] EKSPORT → GGUF: {ollama_name}")
    gguf_dir = f"{ollama_name}_gguf"
    model.save_pretrained_gguf(gguf_dir, tokenizer, quantization_method="q4_k_m")

    # Tworzymy Modelfile
    with open(os.path.join(gguf_dir, "Modelfile"), "w") as f:
        f.write("FROM ./model.gguf\n")

    # Pakujemy do ZIP
    zip_name = f"{ollama_name}_gguf"
    shutil.make_archive(zip_name, "zip", gguf_dir)

    # Zwalniamy pamięć GPU
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    # Pobieramy ZIP
    print(f"⬇️  Pobieram {zip_name}.zip ...")
    files.download(f"{zip_name}.zip")
    print(f"✓ {ollama_name} gotowe!\n")
    print(f"  Lokalnie po rozpakowaniu:")
    print(f"  cd {ollama_name}_gguf/")
    print(f"  ollama create {ollama_name}-newsgroups -f Modelfile")
    print(f"  ollama run {ollama_name}-newsgroups\n")


# ============================================================
# CELL 5 - Uruchom wszystko
# ============================================================
for name, hf_path in MODELS_TO_TRAIN.items():
    train_and_export(name, hf_path)

print("\n✅ WSZYSTKIE MODELE WYTRENOWANE I POBRANE!")