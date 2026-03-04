import ast
import random
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# ============================================================
# 1. Configuración y semilla
# ============================================================

CSV_PATH = "./race_year.csv"
TEST_RATIO = 0.2
SEED = 42

TYRES_INPUT = ["SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"]
TYRES_OUTPUT = TYRES_INPUT + ["<END>"]
tyre_to_idx = {t: i for i, t in enumerate(TYRES_OUTPUT)}
idx_to_tyre = {i: t for t, i in tyre_to_idx.items()}

INPUT_DIM = 17
HIDDEN_DIM = 128
BATCH_SIZE = 32
EPOCHS = 30
LR = 5e-4
ALPHA_LAPS = 0.005

TEACHER_FORCING_START = 1.0
TEACHER_FORCING_END = 0.3

random.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)

# ============================================================
# 2. Funciones de preprocesamiento y codificación
# ============================================================

def encode_current_tyre(tyre):
    vec = [0]*len(TYRES_INPUT)
    if tyre in TYRES_INPUT:
        vec[TYRES_INPUT.index(tyre)] = 1
    return vec

def encode_tyres_used(tyres_used):
    vec = [0]*len(TYRES_INPUT)
    for t in tyres_used:
        if t in TYRES_INPUT:
            vec[TYRES_INPUT.index(t)] = 1
    return vec

def encode_stint(row, scaler=None, numeric_idx=None):
    features = []
    # Variables numéricas
    features.append(row.stint_number)
    features.append(row.stint_length)
    features.append(row.laps_remaining)
    # Booleanas
    features.append(int(row.SC))
    features.append(int(row.VSC))
    features.append(int(row.rain))
    features.append(int(row.is_final_stint))
    # One-hot
    features.extend(encode_current_tyre(row.current_tyre))
    features.extend(encode_tyres_used(row.tyres_used_so_far))
    
    x = torch.tensor(features, dtype=torch.float32)
    
    # Normalización robusta (si se pasan los argumentos)
    if scaler is not None and numeric_idx is not None:
        x[numeric_idx] = torch.tensor(
            scaler.transform(x[numeric_idx].unsqueeze(0))[0],
            dtype=torch.float32
        )
    return x

# ============================================================
# 3. Carga y preparación de datos
# ============================================================

df = pd.read_csv(CSV_PATH)
df["tyres_used_so_far"] = df["tyres_used_so_far"].apply(ast.literal_eval)

for c in ["SC", "VSC", "rain", "is_final_stint"]:
    df[c] = df[c].astype(bool)

df = df.sort_values(["race_id", "driver_id", "stint_number"])

# -------- SPLIT ALEATORIO POR CARRERAS -------- #

race_ids = df.race_id.unique().tolist()
random.shuffle(race_ids)   # <<< CLAVE

split = int(len(race_ids) * (1 - TEST_RATIO))
train_ids = set(race_ids[:split])
test_ids  = set(race_ids[split:])

train_df = df[df.race_id.isin(train_ids)]
test_df  = df[df.race_id.isin(test_ids)]

# -------- NORMALIZACIÓN SOLO CON TRAIN -------- #

NUMERIC_IDX = [0, 1, 2]

scaler = StandardScaler()
numeric_features = [
    [row.stint_number, row.stint_length, row.laps_remaining]
    for _, row in train_df.iterrows()
]

scaler.fit(numeric_features)

print(f"Train races: {len(train_ids)} | Test races: {len(test_ids)}")


# ============================================================
# 4. Dataset y DataLoader
# ============================================================

class TyreStrategyDataset(Dataset):
    def __init__(self, dataframe, scaler=None, numeric_idx=None):
        self.samples = []
        for _, group in dataframe.groupby(["race_id","driver_id"]):
            group = group.sort_values(["stint_number"])
            encoded = [encode_stint(row, scaler, numeric_idx) for row in group.itertuples()]
            tyres = list(group.current_tyre)
            lengths = list(group.stint_length)
            for i in range(len(encoded)):
                x_seq = encoded[:i+1]
                if i < len(encoded)-1:
                    target_tyre = tyre_to_idx[tyres[i+1]]
                    target_laps = lengths[i+1]
                else:
                    target_tyre = tyre_to_idx["<END>"]
                    target_laps = 0.0
                self.samples.append((x_seq, target_tyre, target_laps))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]

def collate_fn(batch):
    sequences, target_tyres, target_laps = zip(*batch)
    sequences = [torch.stack(seq) for seq in sequences]
    lengths = torch.tensor([s.size(0) for s in sequences])
    padded = pad_sequence(sequences, batch_first=True)
    return padded, lengths, torch.tensor(target_tyres, dtype=torch.long), torch.tensor(target_laps, dtype=torch.float32)

# ============================================================
# 5. Modelo Multitask 
# ============================================================

class TyreStrategyLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(INPUT_DIM, HIDDEN_DIM, batch_first=True)
        self.fc_tyre = nn.Linear(HIDDEN_DIM, len(TYRES_OUTPUT))
        self.fc_laps = nn.Linear(HIDDEN_DIM, 1)
    
    def forward(self, x, lengths):
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (h_n, _) = self.lstm(packed)
        hidden = h_n[-1]
        out_tyre = self.fc_tyre(hidden)
        out_laps = self.fc_laps(hidden)
        return out_tyre, out_laps

# ============================================================
# 6. Funciones de Simulación y Métricas (Adaptadas)
# ============================================================

def generate_strategy(model, race_df, scaler, numeric_idx, max_stints=6):
    """
    Simula una estrategia completa paso a paso.
    ADAPTADA: Recibe scaler y numeric_idx explícitamente.
    """
    model.eval()
    race_df = race_df.sort_values("stint_number")
    history = [race_df.iloc[0].copy()] # Primer stint (input inicial)

    idx_inter = tyre_to_idx["INTERMEDIATE"]
    idx_wet = tyre_to_idx["WET"]

    with torch.no_grad():
        for _ in range(max_stints - 1):
            # Codificar historial con argumentos correctos
            seq = torch.stack([
                encode_stint(r, scaler, numeric_idx) for r in history
            ]).unsqueeze(0)

            lengths = torch.tensor([len(history)])
            tyre_logits, laps_pred = model(seq, lengths)

            # Rain masking
            is_raining = history[-1]["rain"]
            if not is_raining:
                tyre_logits[0, idx_inter] = -1e9
                tyre_logits[0, idx_wet] = -1e9

            pred_idx = tyre_logits.argmax(1).item()
            pred_tyre = idx_to_tyre[pred_idx]

            if pred_tyre == "<END>":
                break

            prev = history[-1]
            new_row = prev.copy()
            new_row["stint_number"] = prev["stint_number"] + 1
            new_row["current_tyre"] = pred_tyre
            new_row["tyres_used_so_far"] = prev["tyres_used_so_far"] + [pred_tyre]

            # Actualizar estado dinámico
            pred_laps = max(1, int(round(laps_pred.item())))
            new_row["stint_length"] = pred_laps
            
            # Restamos la duración del stint ANTERIOR a las vueltas restantes
            laps_consumed_prev = prev["stint_length"]
            new_row["laps_remaining"] = max(0, prev["laps_remaining"] - laps_consumed_prev)

            if new_row["laps_remaining"] <= 0:
                history.append(new_row)
                break

            history.append(new_row)

    return [r["current_tyre"] for r in history]

# --- Métricas detalladas portadas de Prueba 2 ---

def exact_sequence_accuracy(model, df, scaler, numeric_idx):
    correct = 0
    total = 0
    for _, group in df.groupby(["race_id", "driver_id"]):
        real = list(group.sort_values("stint_number").current_tyre)
        pred = generate_strategy(model, group, scaler, numeric_idx)
        total += 1
        correct += int(pred == real)
    return correct / total if total else 0

def prefix_accuracy(model, df, scaler, numeric_idx, k=2):
    ok = 0
    total = 0
    for _, group in df.groupby(["race_id", "driver_id"]):
        real = list(group.sort_values("stint_number").current_tyre)
        pred = generate_strategy(model, group, scaler, numeric_idx)
        if len(real) >= k and len(pred) >= k:
            total += 1
            ok += int(real[:k] == pred[:k])
    return ok / total if total else 0

def evaluate_classification_only(model, dataframe):
    """Evalúa Next-Step Accuracy (Teacher Forcing)"""
    model.eval()
    correct, total = 0,0
    idx_inter = tyre_to_idx["INTERMEDIATE"]
    idx_wet = tyre_to_idx["WET"]
    
    for _, group in dataframe.groupby(["race_id","driver_id"]):
        group = group.sort_values("stint_number")
        encoded_rows = [encode_stint(r, scaler, NUMERIC_IDX) for r in group.itertuples()]
        real_tyres = list(group.current_tyre)
        for i in range(len(encoded_rows)-1):
            seq = torch.stack(encoded_rows[:i+1]).unsqueeze(0)
            lengths = torch.tensor([len(seq[0])])
            with torch.no_grad():
                pred_logits, _ = model(seq, lengths)
                current_rain = group.iloc[i]["rain"]
                if not current_rain:
                    pred_logits[0, idx_inter] = -float('inf')
                    pred_logits[0, idx_wet] = -float('inf')
                pred_idx = pred_logits.argmax(1).item()
                pred_tyre = idx_to_tyre[pred_idx]
                target_tyre = real_tyres[i+1]
                if target_tyre != "<END>":
                    correct += int(pred_tyre == target_tyre)
                    total += 1
    return correct/total if total>0 else 0

def demo_prediction_with_laps(model, dataframe):
    model.eval()
    unique_races = dataframe[['race_id','driver_id']].drop_duplicates().values
    r_id, d_id = random.choice(unique_races)
    group = dataframe[(dataframe.race_id==r_id) & (dataframe.driver_id==d_id)].sort_values("stint_number")
    first_stint = group.iloc[0]
    history = [first_stint]
    print(f"\nDemo: Carrera {r_id}, Piloto {d_id} | Inicio con {first_stint.current_tyre} ({first_stint.laps_remaining} vueltas)")
    print(f"{'STINT':<6} | {'TYRE':<20} | {'PRED_LAPS':<12} | {'REMAINING'}")
    
    for i in range(3):
        seq = torch.stack([encode_stint(r, scaler, NUMERIC_IDX) for r in history]).unsqueeze(0)
        lengths = torch.tensor([len(history)])
        with torch.no_grad():
            pred_logits, pred_laps_tensor = model(seq, lengths)
            pred_idx = pred_logits.argmax(1).item()
            pred_tyre = idx_to_tyre[pred_idx]
            if pred_tyre=="<END>": break
            pred_laps = pred_laps_tensor.item()
            
            last_laps_remaining = history[-1]["laps_remaining"]
            current_stint_laps = int(round(pred_laps))
            new_remaining = max(0, last_laps_remaining-current_stint_laps)
            
            print(f"{i+2:<6} | {pred_tyre:<20} | {pred_laps:<12.2f} | {new_remaining}")
            
            new_row = history[-1].copy()
            new_row["current_tyre"] = pred_tyre
            new_row["stint_length"] = current_stint_laps
            new_row["laps_remaining"] = new_remaining
            history.append(new_row)
            if new_remaining<=0: break

# ============================================================
# 7. Pesos de clase para desbalance
# ============================================================

all_tyres = []
for _, group in train_df.groupby(["race_id","driver_id"]):
    all_tyres.extend(list(group.current_tyre))
y_train_indices = [tyre_to_idx[t] for t in all_tyres if t in tyre_to_idx]
# Pesos suavizados basados en proporciones reales (evita colapso a HARD)

weights_tensor = torch.tensor([
    3.5,   # SOFT
    1.0,   # MEDIUM
    1.1,   # HARD
    4.5,   # INTERMEDIATE
    6.0    # WET
], dtype=torch.float32)

weights_tensor = torch.cat([weights_tensor, torch.tensor([1.0])])

print("Pesos finales usados:", weights_tensor)


# ============================================================
# 8. Training Loop (CON registro DE LOSS)
# ============================================================

train_loader = DataLoader(
    TyreStrategyDataset(train_df, scaler, NUMERIC_IDX),
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn
)

model = TyreStrategyLSTM()
criterion_tyre = nn.CrossEntropyLoss(weight=weights_tensor)
criterion_laps = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

print("\nIniciando entrenamiento...")

train_acc_list = []
test_acc_list = []

loss_history = {
    "total": [],
    "tyre": [],
    "laps": []
}

for epoch in range(EPOCHS):
    teacher_forcing_prob = TEACHER_FORCING_START - (
        (TEACHER_FORCING_START - TEACHER_FORCING_END) * (epoch / (EPOCHS - 1))
    )

    loss_sum = 0
    loss_tyre_sum = 0
    loss_laps_sum = 0
    batches = 0

    model.train()

    for X, lengths, y_tyre, y_laps in train_loader:
        optimizer.zero_grad()

        preds_tyre, preds_laps = model(X, lengths)

        # --- CLASIFICACIÓN ---
        loss_cls = criterion_tyre(preds_tyre, y_tyre)

        # --- REGRESIÓN (CORREGIDA) ---
        loss_reg = criterion_laps(
            preds_laps.squeeze(1),   
            y_laps
        )

        # --- LOSS TOTAL ---
        loss = loss_cls + ALPHA_LAPS * loss_reg

        loss.backward()
        optimizer.step()

        loss_sum += loss.item()
        loss_tyre_sum += loss_cls.item()
        loss_laps_sum += loss_reg.item()
        batches += 1

    avg_loss = loss_sum / batches
    avg_tyre = loss_tyre_sum / batches
    avg_laps = loss_laps_sum / batches

    loss_history["total"].append(avg_loss)
    loss_history["tyre"].append(avg_tyre)
    loss_history["laps"].append(avg_laps)

    train_acc = evaluate_classification_only(model, train_df)
    test_acc = evaluate_classification_only(model, test_df)

    train_acc_list.append(train_acc)
    test_acc_list.append(test_acc)

    print(
        f"Epoch {epoch+1:02d} | "
        f"Loss Total: {avg_loss:.4f} "
        f"(Tyre: {avg_tyre:.4f} | Laps: {avg_laps:.1f}) | "
        f"Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}"
    )


# ============================================================
# 10. Resultados Finales y GRÁFICAS COMPLETAS
# ============================================================

# 1. Gráfica de Accuracy
plt.figure(figsize=(10,5))
plt.plot(range(1, EPOCHS+1), train_acc_list, label="Train Accuracy")
plt.plot(range(1, EPOCHS+1), test_acc_list, label="Test Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.show()

# 2. Gráfica de LOSS DESGLOSADA
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Gráfica izquierda: Tyre Loss (Cross Entropy)
ax1.plot(range(1, EPOCHS+1), loss_history["tyre"], color='tab:red', label='Tyre Loss')
ax1.set_title("Pérdida de Clasificación (Neumáticos)")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("CrossEntropy Loss")
ax1.grid(True)
ax1.legend()

# Gráfica derecha: Laps Loss (MSE)
ax2.plot(range(1, EPOCHS+1), loss_history["laps"], color='tab:orange', label='Laps MSE')
ax2.set_title("Error de Regresión (Duración Stint)")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Mean Squared Error (Vueltas^2)")
ax2.grid(True)
ax2.legend()

plt.show()

# ============================================================
# 9. Evaluación final y Reporte Completo
# ============================================================

print("\n--- REPORTE FINAL DE RENDIMIENTO (TEST SET) ---\n")

# 1. Next Step Accuracy (Teacher Forcing)
final_test_acc = evaluate_classification_only(model, test_df)
print(f"Next-step Classification Accuracy: {final_test_acc:.4f} ({final_test_acc*100:.2f}%)")

# 2. Exact Sequence Accuracy (Simulación completa)
acc_exact = exact_sequence_accuracy(model, test_df, scaler, NUMERIC_IDX)
print(f"Exact Sequence Accuracy: {acc_exact:.4f} ({acc_exact*100:.2f}%)")

# 3. Prefix Accuracy (Simulación parcial k=2 y k=3)
acc_prefix_2 = prefix_accuracy(model, test_df, scaler, NUMERIC_IDX, k=2)
print(f"Prefix Accuracy (Primeros 2 stints): {acc_prefix_2:.4f} ({acc_prefix_2*100:.2f}%)")

acc_prefix_3 = prefix_accuracy(model, test_df, scaler, NUMERIC_IDX, k=3)
print(f"Prefix Accuracy (Primeros 3 stints): {acc_prefix_3:.4f} ({acc_prefix_3*100:.2f}%)")

# Demo visual
demo_prediction_with_laps(model, test_df)

# Matriz de confusión
print("\nGenerando matriz de confusión...")
y_true, y_pred = [], []
model.eval()
with torch.no_grad():
    for _, group in test_df.groupby(["race_id","driver_id"]):
        real_targets = list(group.sort_values("stint_number").current_tyre)[1:]
        # Usamos generate_strategy con todos los argumentos requeridos
        pred_targets = generate_strategy(model, group, scaler, NUMERIC_IDX)[1:]
        
        min_len = min(len(real_targets), len(pred_targets))
        y_true.extend(real_targets[:min_len])
        y_pred.extend(pred_targets[:min_len])

labels = TYRES_INPUT
cm = confusion_matrix(y_true, y_pred, labels=labels)
plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicción")
plt.ylabel("Real")
plt.show()

# --- AÑADIDO: REPORTE DETALLADO SIN TOCAR LÓGICA ANTERIOR ---
print("\n--- DETALLE POR CLASE (Classification Report) ---\n")
print(classification_report(y_true, y_pred, labels=labels, zero_division=0))