import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 58008
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# --- Data ---
n = 120_000
X = np.random.uniform(-2 * np.pi, 2 * np.pi, n).reshape(-1, 1)

def f(x):
    return 2 * (2 * np.cos(x) ** 2 - 1) ** 2 - 1

Y = f(X).reshape(-1, 1)

X_train_np, X_test_np, y_train_np, y_test_np = train_test_split(
    X, Y, test_size=60_000, random_state=SEED
)

def to_tensor(arr):
    return torch.tensor(arr, dtype=torch.float32).to(device)

X_train = to_tensor(X_train_np)
y_train = to_tensor(y_train_np)
X_test  = to_tensor(X_test_np)
y_test  = to_tensor(y_test_np)

# --- Model ---
def build_model(n_hidden_layers, n_neurons):
    layers = []
    in_features = 1
    for _ in range(n_hidden_layers):
        layers += [nn.Linear(in_features, n_neurons), nn.ReLU()]
        in_features = n_neurons
    layers.append(nn.Linear(in_features, 1))
    return nn.Sequential(*layers).to(device) # layers is a list, * unpacks the list

# --- Training ---
def train_model(model, X_tr, y_tr, epochs=300, batch_size=256, patience=20):
    # 80/20 train/val split
    val_size = int(0.2 * len(X_tr))
    X_val, y_val = X_tr[:val_size], y_tr[:val_size]
    X_t,   y_t   = X_tr[val_size:], y_tr[val_size:]

    loader = DataLoader(TensorDataset(X_t, y_t), batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None

    for epoch in range(epochs):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            criterion(model(xb), yb).backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val), y_val).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    model.load_state_dict(best_state)
    return model

# --- Grid search ---
depth_list   = [1, 2, 3]
neurons_list = [5, 10, 20, 50, 70, 80]
criterion    = nn.MSELoss()
results      = []

for depth in depth_list:
    for neurons in neurons_list:
        print(f"Building model: depth={depth}, neurons={neurons}")
        model = build_model(depth, neurons)
        train_model(model, X_train, y_train)

        model.eval()
        with torch.no_grad():
            test_mse = criterion(model(X_test), y_test).item()

        n_params = sum(p.numel() for p in model.parameters())
        results.append({"depth": depth, "neurons": neurons, "params": n_params, "test_mse": test_mse})

df = pd.DataFrame(results)

# --- Plot ---
plt.figure(figsize=(8, 6))
for depth in depth_list:
    sub = df[df["depth"] == depth].sort_values("params")
    plt.plot(sub["neurons"], sub["test_mse"], marker="o", label=f"{depth} hidden layer(s)")

plt.xlabel("Number of neurons")
plt.ylabel("Test MSE")
plt.title("Deep vs shallow networks")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("results.png", dpi=150)
plt.show()

print(df.to_string(index=False))