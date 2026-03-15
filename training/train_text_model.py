"""
AEGIS AI — Text AI Detection Training
Dataset: Auto-generated human vs AI text samples
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from torch.optim import Adam

# ── Auto Dataset — No download needed ───────────────
HUMAN_TEXTS = [
    "I went to the market today and bought some vegetables.",
    "My dog kept barking at the neighbor's cat all morning.",
    "The movie was okay but the ending felt rushed to me.",
    "I burned my toast again, Monday is already a disaster.",
    "Called my mom after ages, she was really happy to hear from me.",
    "Traffic was insane today, took me 2 hours to reach office.",
    "Finally finished that book I started 3 months ago.",
    "The chai at that small stall near station is unbeatable.",
    "My laptop fan is making weird noises, need to get it checked.",
    "Couldn't sleep last night, kept thinking about random stuff.",
    "The match yesterday was incredible, last over was insane.",
    "I tried cooking biryani for the first time, it was decent.",
    "Why do banks make everything so complicated?",
    "My phone battery barely lasts half a day now.",
    "Just realized I've been mispronouncing that word for years.",
] * 20  # 300 samples

AI_TEXTS = [
    "Artificial intelligence represents a transformative paradigm in modern computing.",
    "The implementation of machine learning algorithms enables sophisticated pattern recognition.",
    "Natural language processing facilitates seamless human-computer interaction.",
    "Deep neural networks have revolutionized the field of computer vision significantly.",
    "The optimization of hyperparameters is crucial for model performance enhancement.",
    "Blockchain technology ensures decentralized and immutable transaction records.",
    "The proliferation of digital technologies has fundamentally altered societal structures.",
    "Sustainable development requires a comprehensive approach to resource management.",
    "The integration of cloud computing solutions enhances operational efficiency.",
    "Advanced analytics provide actionable insights for data-driven decision making.",
    "The convergence of IoT and AI creates unprecedented opportunities for automation.",
    "Robust cybersecurity frameworks are essential for protecting digital infrastructure.",
    "The democratization of AI tools empowers organizations across various sectors.",
    "Leveraging big data analytics can significantly improve business outcomes.",
    "The systematic evaluation of algorithmic bias ensures equitable AI deployment.",
] * 20  # 300 samples

# ── Dataset Class ────────────────────────────────────
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.encodings = tokenizer(
            texts, truncation=True, padding=True,
            max_length=max_len, return_tensors="pt"
        )
        self.labels = torch.tensor(labels)

    def __len__(self): return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids":      self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels":         self.labels[idx]
        }

# ── Setup ─────────────────────────────────────────────
print("Loading tokenizer...")
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

texts  = HUMAN_TEXTS + AI_TEXTS
labels = [0] * len(HUMAN_TEXTS) + [1] * len(AI_TEXTS)  # 0=Human, 1=AI

dataset    = TextDataset(texts, labels, tokenizer)
train_size = int(0.8 * len(dataset))
test_size  = len(dataset) - train_size
train_set, test_set = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
test_loader  = DataLoader(test_set,  batch_size=16)

print(f"Train: {len(train_set)} | Test: {len(test_set)}")

# ── Model ─────────────────────────────────────────────
print("Loading DistilBERT...")
model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=2
)
optimizer = Adam(model.parameters(), lr=2e-5)

# ── Train ─────────────────────────────────────────────
EPOCHS = 3
best_acc = 0.0

for epoch in range(EPOCHS):
    model.train()
    correct, total = 0, 0
    for batch in train_loader:
        optimizer.zero_grad()
        out  = model(**{k: v for k, v in batch.items() if k != "labels"},
                     labels=batch["labels"])
        out.loss.backward()
        optimizer.step()
        preds    = out.logits.argmax(dim=1)
        correct += (preds == batch["labels"]).sum().item()
        total   += batch["labels"].size(0)

    train_acc = correct / total * 100

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch in test_loader:
            out   = model(**{k: v for k, v in batch.items() if k != "labels"})
            preds = out.logits.argmax(dim=1)
            correct += (preds == batch["labels"]).sum().item()
            total   += batch["labels"].size(0)

    test_acc = correct / total * 100
    print(f"Epoch {epoch+1}/{EPOCHS} | Train: {train_acc:.1f}% | Test: {test_acc:.1f}%")

    if test_acc > best_acc:
        best_acc = test_acc
        model.save_pretrained("training/weights/text_model")
        tokenizer.save_pretrained("training/weights/text_model")
        print(f"  ✅ Saved! Best: {best_acc:.1f}%")

print(f"\n🏁 Done! Best: {best_acc:.1f}%")