import os
import re
import logging
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
from torch.optim import AdamW
from training_data_searching import RequestDataset

# Pour les logs
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

# Constantes
MODEL_NAME    = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
# j'ai dû utiliser le cpu de mon PC, cette étape prend le plus de temps
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE    = 32
EPOCHS        = 8
LR_HEAD       = 2e-4
LR_BERT       = 2e-5
WEIGHT_DECAY  = 0.01
UNFREEZE_FROM = 8   # on dé-gèle les 4 dernières couches afin de pouvoir les modifier

# Je labelise
label2id = {"transport":0, "météo":1, "culture":2, "loisirs":3}
id2label = {v:k for k,v in label2id.items()}

# Dataset
train_ds = RequestDataset("train.jsonl", label2id)
val_ds   = RequestDataset("val.jsonl",   label2id)

# Modèle & Head
backbone = SentenceTransformer(MODEL_NAME, device=DEVICE)
embed_dim = backbone.get_sentence_embedding_dimension()

# freeze complet
for p in backbone.parameters():
    p.requires_grad = False
# dégèle des dernières couches BERT
for name, p in backbone.named_parameters():
    if any(f"layer.{i}" in name for i in range(UNFREEZE_FROM, 12)):
        p.requires_grad = True

clf = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(embed_dim, 256),
    nn.GELU(),
    nn.Dropout(0.3),
    nn.Linear(256, len(label2id))
).to(DEVICE)

# Loss pondérée (plus de sampler)
counts   = np.bincount(train_ds.labels)
base_w   = len(train_ds) / counts          # inverse fréquence

boost = {
    label2id["transport"]: 1.8,   # x1.8
    label2id["météo"]:     1.6,   # x1.6
    label2id["loisirs"]:   0.7,   # x0.7
    # pas besoin de changer culture
}

weights = torch.tensor(
    [base_w[i] * boost.get(i, 1.0) for i in range(len(base_w))],
    device=DEVICE,
    dtype=torch.float32
)
weights = weights / weights.mean()

loss_fn = nn.CrossEntropyLoss(weight=weights)
# chargement de données
def collate_fn(batch):
    texts, labs = zip(*batch)
    embs = backbone.encode(
        list(texts),
        convert_to_tensor=True,
        device=DEVICE
    ).float()
    return embs, torch.tensor(labs, device=DEVICE)

train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn
)
val_loader = DataLoader(
    val_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_fn
)

#Optim + Scheduler
no_decay = ["bias", "LayerNorm.weight"]
optim_groups = [
    {
        "params": clf.parameters(),
        "lr": LR_HEAD,
        "weight_decay": WEIGHT_DECAY
    },
    {
        "params": [
            p for n,p in backbone.named_parameters()
            if p.requires_grad and not any(nd in n for nd in no_decay)
        ],
        "lr": LR_BERT,
        "weight_decay": WEIGHT_DECAY
    },
    {
        "params": [
            p for n,p in backbone.named_parameters()
            if p.requires_grad and any(nd in n for nd in no_decay)
        ],
        "lr": LR_BERT,
        "weight_decay": 0.0
    },
]
opt = AdamW(optim_groups)
total_steps = len(train_loader) * EPOCHS
sched = get_linear_schedule_with_warmup(
    opt,
    num_warmup_steps=int(0.06 * total_steps),
    num_training_steps=total_steps
)

# Boucle d'entrainement
history = {"train_loss":[], "val_loss":[], "acc":[], "bal_acc":[], "f1":[]}
best_bal_acc, stale, patience = 0.0, 0, 5

for epoch in range(1, EPOCHS + 1):
    clf.train()
    running_loss = 0.0
    for embs, labs in train_loader:
        logits = clf(embs)
        loss   = loss_fn(logits, labs)
        loss.backward()
        clip_grad_norm_(clf.parameters(), 1.0)
        opt.step(); sched.step(); opt.zero_grad()
        running_loss += loss.item() * labs.size(0)
    train_loss = running_loss / len(train_ds)
    history["train_loss"].append(train_loss)

    clf.eval()
    val_running_loss = 0.0
    all_preds, all_golds = [], []
    with torch.no_grad():
        for embs, labs in val_loader:
            logits = clf(embs)
            loss   = loss_fn(logits, labs)
            val_running_loss += loss.item() * labs.size(0)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_golds.extend(labs.cpu().numpy())

    val_loss = val_running_loss / len(val_ds)
    acc      = accuracy_score(all_golds, all_preds)
    bal_acc  = balanced_accuracy_score(all_golds, all_preds)
    f1_macro = f1_score(all_golds, all_preds, average="macro")
    history["val_loss"].append(val_loss)
    history["acc"].append(acc)
    history["bal_acc"].append(bal_acc)
    history["f1"].append(f1_macro)

    logging.info(
        f"Epoch {epoch}: train_loss={train_loss:.3f} "
        f"val_loss={val_loss:.3f} bal_acc={bal_acc:.3f} f1={f1_macro:.3f}"
    )

    if bal_acc > best_bal_acc:
        best_bal_acc, stale = bal_acc, 0
        os.makedirs("../checkpoints", exist_ok=True)
        torch.save({
            "sbert":    backbone.state_dict(),
            "clf":      clf.state_dict(),
            "label2id": label2id
        }, "../checkpoints/dispatcher_sbert.pt")
    else:
        stale += 1
        if stale >= patience:
            logging.info("Early stopping.")
            break

# ===== Plot =====
ep = list(range(1, len(history["train_loss"]) + 1))
plt.figure()
plt.plot(ep, history["train_loss"], label="train loss")
plt.plot(ep, history["val_loss"],   label="val loss")
plt.legend(); plt.title("Courbes de perte"); plt.show()

plt.figure()
plt.plot(ep, history["bal_acc"], label="Balanced acc")
plt.plot(ep, history["acc"],    "--", label="Acc brute")
plt.legend(); plt.title("Scores de validation"); plt.show()

# ===== Dispatch avec fallback =====
def contains_loisir(text: str) -> bool:
    pat = re.compile(r"\b(que\s+faire|quoi\s+faire|activit[\wé]*|concert|festival)\b", re.I)
    bad = re.compile(r"\b(train|bus|voiture|m[ée]tro|tram|taxi)\b", re.I)
    return bool(pat.search(text)) and not bool(bad.search(text))

def dispatch(text: str) -> str:
    embs   = backbone.encode([text], convert_to_tensor=True, device=DEVICE).float()
    logits = clf(embs)
    probs  = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
    pred   = int(probs.argmax())
    if pred != label2id["loisirs"] and contains_loisir(text):
        if probs[label2id["loisirs"]] >= 0.25:
            pred = label2id["loisirs"]
    return id2label[pred]
