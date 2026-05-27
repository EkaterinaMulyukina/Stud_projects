import numpy as np
import re
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import random

EMB_DIM = 100
WINDOW_SIZE = 3
K_NEG = 10
LR_W2V = 0.05
LR_LSTM = 0.001
EPOCHS_W2V = 1
EPOCHS_LSTM = 3
BATCH_SIZE = 32


def preprocess_simple(text):
    text = re.sub(r"[^\w\s]", "", text.lower())
    return [w for w in text.split() if len(w) > 2]


# 2 lab
def preprocess_to_fragments(text):
    text = text.lower()
    fragments = re.split(r"[\.,\-]", text)
    clean_fragments = []
    for frag in fragments:
        words = [w for w in frag.split() if len(w) > 2]
        if len(words) > 1:
            clean_fragments.append(words)
    return clean_fragments


with open("book.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

w2v_corpus = preprocess_simple(raw_text)
vocab = sorted(list(set(w2v_corpus)))
word2idx = {w: i for i, w in enumerate(vocab)}
idx2word = {i: w for i, w in enumerate(vocab)}
vocab_size = len(vocab)
corpus_fragments = preprocess_to_fragments(raw_text)  # 2 lab

W_target = np.random.randn(vocab_size, EMB_DIM) * 0.1  # 2 lab
W_context = np.random.randn(vocab_size, EMB_DIM) * 0.1


def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -20, 20)))


for i, word in enumerate(w2v_corpus):
    if i % 20000 == 0:
        print(f"  Прогресс: {i}/{len(w2v_corpus)}")
    target_idx = word2idx[word]
    start, end = max(0, i - WINDOW_SIZE), min(
        len(w2v_corpus), i + WINDOW_SIZE + 1
    )
    pos_indices = [
        word2idx[w2v_corpus[j]] for j in range(start, end) if j != i
    ]
    neg_indices = np.random.choice(vocab_size, K_NEG, replace=False)

    for pos_idx in pos_indices:
        score = np.dot(W_target[target_idx], W_context[pos_idx])
        grad = sigmoid(score) - 1
        W_target[target_idx] -= LR_W2V * grad * W_context[pos_idx]
        W_context[pos_idx] -= LR_W2V * grad * W_target[target_idx]

    for neg_idx in neg_indices:
        score = np.dot(W_target[target_idx], W_context[neg_idx])
        grad = sigmoid(score)
        W_target[target_idx] -= LR_W2V * grad * W_context[neg_idx]
        W_context[neg_idx] -= LR_W2V * grad * W_target[target_idx]

pretrained_weights = W_target


class SeqDataset(Dataset):
    def __init__(self, fragments, word2idx):
        self.samples = []
        for frag in fragments:
            indices = [word2idx[w] for w in frag if w in word2idx]
            if len(indices) > 1:
                self.samples.append((indices[:-1], indices[1:]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.tensor(x), torch.tensor(y)


def collate_fn(batch):
    xs, ys = zip(*batch)
    xs_pad = nn.utils.rnn.pad_sequence(xs, batch_first=True, padding_value=0)
    ys_pad = nn.utils.rnn.pad_sequence(ys, batch_first=True, padding_value=-1)
    return xs_pad, ys_pad


random.shuffle(corpus_fragments)
split_idx = int(len(corpus_fragments) * 0.9)
train_data = corpus_fragments[:split_idx]
test_data = corpus_fragments[split_idx:]

train_loader = DataLoader(SeqDataset(train_data, word2idx), batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn,)  # порядок батчей будет меняться
test_loader = DataLoader(SeqDataset(test_data, word2idx), batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn,)

print(f"Обучение: {len(train_data)}, Тест: {len(test_data)}")


class LSTM_Seq2Seq(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, weights):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(
            torch.FloatTensor(weights), freeze=False
        )
        self.encoder = nn.LSTM(emb_dim, hidden_dim, batch_first=True)
        self.decoder = nn.LSTM(emb_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        embedded = self.embedding(x)
        _, (h_n, c_n) = self.encoder(embedded)
        out, _ = self.decoder(embedded, (h_n, c_n))
        return self.fc(out)


device = torch.device("cpu")
model = LSTM_Seq2Seq(vocab_size, EMB_DIM, 256, pretrained_weights).to(device)
criterion = nn.CrossEntropyLoss(ignore_index=-1)
optimizer = optim.Adam(model.parameters(), lr=LR_LSTM)

for epoch in range(EPOCHS_LSTM):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for bx, by in train_loader:
        bx, by = bx.to(device), by.to(device)
        optimizer.zero_grad()
        logits = model(bx)
        loss = criterion(logits.view(-1, vocab_size), by.view(-1))
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = torch.argmax(logits, dim=-1)
        mask = by != -1
        correct += (preds[mask] == by[mask]).sum().item()
        total += mask.sum().item()

    acc = (correct / total) * 100
    perplexity = np.exp(total_loss / len(train_loader))
    print(
        f"Эпоха {epoch + 1}/{EPOCHS_LSTM} | Accuracy: {acc:.2f}% | Perplexity: {perplexity:.2f}"
    )

model.eval()
t_correct, t_total = 0, 0
with torch.no_grad():
    for bx, by in test_loader:
        bx, by = bx.to(device), by.to(device)
        logits = model(bx)
        preds = torch.argmax(logits, dim=-1)
        mask = by != -1
        t_correct += (preds[mask] == by[mask]).sum().item()
        t_total += mask.sum().item()

print(f"\nНа тестовой выборке: Accuracy = {(t_correct / t_total) * 100:.2f}%")


def predict_next_word(phrase):
    model.eval()
    words = phrase.lower().split()
    indices = [word2idx[w] for w in words if w in word2idx]
    if not indices:
        return "Слов нет в словаре"
    in_t = torch.tensor([indices]).to(device)
    with torch.no_grad():
        logits = model(in_t)
        next_word_idx = torch.argmax(logits[0, -1, :]).item()
    return idx2word[next_word_idx]


while True:
    user_input = input("Введите начало фразы: ").strip()
    if user_input.lower() == "exit":
        break
    print(f"Предсказание: {predict_next_word(user_input)}")
