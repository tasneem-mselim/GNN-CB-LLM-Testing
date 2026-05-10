import os, random
import numpy as np
import pandas as pd
import tensorflow as tf

DATA_DIR = "" 
TEST_SUBMISSION_PATH = "" 
EDGE_PATH = os.path.join("edge_index.csv")
X_PATH = os.path.join("x.csv")
YTR_PATH = os.path.join("y_train.csv")
YVA_PATH = os.path.join("y_val.csv")
OUT_PATH ="TEST_SUBMISSION_PATH"

SEED = 25
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# load data
X = pd.read_csv(X_PATH).to_numpy(dtype=np.float32)  # (N,F)
N, F = X.shape

e = pd.read_csv(EDGE_PATH)
src = e["source"].to_numpy(dtype=np.int64)
dst = e["target"].to_numpy(dtype=np.int64)

tr = pd.read_csv(YTR_PATH)
va = pd.read_csv(YVA_PATH)
# node_index and node_label for the training/validation data
tr_idx = tr["index"].to_numpy(dtype=np.int64)
tr_y   = tr["label"].to_numpy(dtype=np.int64)
va_idx = va["index"].to_numpy(dtype=np.int64)
va_y   = va["label"].to_numpy(dtype=np.int64)

# number of classes
C = 7

# build one-hot labels for all nodes + masks (initialization)
Y = np.zeros((N, C), dtype=np.float32)
train_mask = np.zeros((N,), dtype=np.float32)
val_mask   = np.zeros((N,), dtype=np.float32)
# add training label as one hot encoding
Y[tr_idx, tr_y] = 1.0
train_mask[tr_idx] = 1.0

Y[va_idx, va_y] = 1.0
val_mask[va_idx] = 1.0

# build adj matrix
edges = set()

# Add edges (undirected)
for i in range(len(src)):
    edges.add((src[i], dst[i]))
    edges.add((dst[i], src[i]))
# Add self-loops
for i in range(N):
    edges.add((i, i))
# Convert to arrays (no duplicates)
row_u = []
col_u = []
val_u = []
for (r, c) in edges:
    row_u.append(r)
    col_u.append(c)
    val_u.append(1.0) 
    
# Build dense adjacency matrix
A = np.zeros((N, N), dtype=np.float32)
for i in range(len(row_u)):
    A[row_u[i], col_u[i]] = val_u[i]
# Add self-loops
I = np.eye(N, dtype=np.float32)
A_hat = A + I
# Compute degree
deg = A_hat.sum(axis=1)
# Avoid division by zero
deg[deg == 0] = 1.0
# Compute D^(-0.5)
deg_inv_sqrt = 1.0 / np.sqrt(deg)
# Normalize: D^-0.5 A D^-0.5
A_norm = deg_inv_sqrt[:, None] * A_hat * deg_inv_sqrt[None, :]
# Convert to sparse tensor - to remove zeros (no edges)
indices = np.array(np.nonzero(A_norm)).T
values = A_norm[A_norm != 0]
A_hat = tf.sparse.SparseTensor(
    indices=indices, values=values.astype(np.float32), dense_shape=(N, N))
A_hat = tf.sparse.reorder(A_hat)


X_in = tf.keras.Input(shape=(F,))
A_in = tf.keras.Input(shape=(None,), sparse=True)

def GCN(inputs):
    features, adj = inputs
    return tf.sparse.sparse_dense_matmul(adj, features)

h = tf.keras.layers.Dense(16, use_bias=False)(X_in)
h = tf.keras.layers.Lambda(GCN, output_shape=(16,))([h, A_in])
h = tf.keras.layers.ReLU()(h)
h = tf.keras.layers.Dropout(0.5)(h)

h = tf.keras.layers.Dense(C, use_bias=False)(h)
h = tf.keras.layers.Lambda(GCN, output_shape=(C,))([h, A_in])

out = tf.keras.layers.Softmax()(h)
model = tf.keras.Model([X_in, A_in], out)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
     loss="categorical_crossentropy", weighted_metrics=["accuracy"])

model.fit(x=[X, A_hat], y=Y, sample_weight=train_mask,                 
    validation_data=([X, A_hat], Y, val_mask), 
    epochs=5, batch_size=N, verbose=2)

proba = model.predict([X, A_hat], batch_size=N, verbose=0)
pred = proba.argmax(axis=1).astype(np.int64)

test_ids = pd.read_csv(TESTID_PATH)["id"].to_numpy(dtype=np.int64)
sub = pd.DataFrame({"id": test_ids, "target": pred[test_ids]})
sub.to_csv(SUB_PATH, index=False)
print("Saved submission:", SUB_PATH)
