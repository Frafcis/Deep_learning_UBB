import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# --- A. FUNDAMENTO TEÓRICO: IMPLEMENTACIÓN MANUAL DE SCREW THEORY ---

def vec_to_skew(w):
    """Convierte un vector 3x1 en una matriz antisimétrica [w] (so(3))"""
    return np.array([[0, -w[2], w[1]],
                    [w[2], 0, -w[0]],
                    [-w[1], w[0], 0]])

def screw_exp(S, theta):
    w = S[:3]
    v = S[3:]

    
    W_skew = vec_to_skew(w)
    I = np.eye(3)

    R = I + np.sin(theta) * W_skew + (1 - np.cos(theta)) * (W_skew @ W_skew)

    G = (I * theta) + \
        ((1 - np.cos(theta)) * W_skew) + \
        ((theta - np.sin(theta)) * (W_skew @ W_skew))
    
    trans = G @ v

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = trans
    
    return T

L1 = 1.0
L2 = 1.0

M = np.eye(4)
M[0, 3] = L1 + L2

w1 = np.array([0, 0, 1])
q1 = np.array([0, 0, 0])
v1 = -np.cross(w1, q1)
S1 = np.concatenate((w1, v1))

w2 = np.array([0, 0, 1])
q2 = np.array([L1, 0, 0])
v2 = -np.cross(w2, q2)
S2 = np.concatenate((w2, v2))

def forward_kinematics_poe(theta1, theta2):
    T1 = screw_exp(S1, theta1)
    T2 = screw_exp(S2, theta2)
    
    T_final = T1 @ T2 @ M
    
    return T_final[0, 3], T_final[1, 3]

NUM_SAMPLES = 3500 

print("Generando dataset con restricciones físicas...")
data = []
count = 0

margen_colision = np.deg2rad(10)
limite_th2 = np.pi - margen_colision

while count < NUM_SAMPLES:

    th1 = np.random.uniform(-np.pi, np.pi) 
    th2 = np.random.uniform(-np.pi, np.pi)

    if np.abs(th2) > limite_th2:
        continue
    x, y = forward_kinematics_poe(th1, th2)
    
    data.append([x, y, th1, th2])
    count += 1

df = pd.DataFrame(data, columns=['x', 'y', 'theta1', 'theta2'])
print(f"Dataset generado: {len(df)} muestras válidas (sin autocolisiones).")
X = df[['x', 'y']].values
y_true = df[['theta1', 'theta2']].values

X_mean, X_std = X.mean(axis=0), X.std(axis=0)
y_mean, y_std = y_true.mean(axis=0), y_true.std(axis=0)

X_norm = (X - X_mean) / X_std
y_norm = (y_true - y_mean) / y_std

split = int(0.8 * len(df))
X_train, X_test = X_norm[:split], X_norm[split:]
y_train, y_test = y_norm[:split], y_norm[split:]

model = models.Sequential([
    layers.Input(shape=(2,)),
    layers.Dense(64, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(2, activation='linear') 
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

print("Entrenando modelo...")
history = model.fit(X_train, y_train, epochs=40, batch_size=32, validation_split=0.1, verbose=0)

y_pred_norm = model.predict(X_test)

y_pred = (y_pred_norm * y_std) + y_mean
y_real = (y_test * y_std) + y_mean

mse = np.mean((y_real - y_pred)**2)
print(f"MSE Final (en radianes^2): {mse:.4f}")

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Loss')
plt.title('Entrenamiento')
plt.legend()

plt.subplot(1, 2, 2)

plt.plot(y_real[:40, 0], 'o-', label='Real $\\theta_1$')
plt.plot(y_pred[:40, 0], 'x--', label='Pred $\\theta_1$')
plt.title('Predicción vs Realidad')
plt.legend()

plt.tight_layout()
plt.show()