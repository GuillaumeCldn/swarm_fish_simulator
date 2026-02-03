import numpy as np
import matplotlib.pyplot as plt
import os
import shutil

# --- Configuration ---
# Le chemin fourni. Si vous êtes sous Windows, assurez-vous que le chemin est correct.
# Python gère généralement bien les slashs '/' même sous Windows.
filename = r"logs\save-flight-02.03.2026_09.09.13.npy"

# --- Chargement des données ---
print(f"Tentative de chargement de : {filename}")

# Le fichier est techniquement un zip (.npz) même s'il a l'extension .npy
try:
    # Essai de chargement direct (numpy récent gère parfois cela)
    data = np.load(filename)
except Exception as e:
    print(f"Chargement direct échoué ({e}). Tentative de renommage en .npz...")
    # Si échec, on copie le fichier avec la bonne extension temporairement
    temp_filename = filename + ".npz"
    shutil.copy(filename, temp_filename)
    data = np.load(temp_filename)

# --- Extraction des variables ---
# Selon util_logger.py :
# timestamps: (N_drones, N_steps)
# states: (N_drones, 12, N_steps) -> [x, y, z, vx, vy, vz, roll, pitch, yaw, wx, wy, wz]
# controls: (N_drones, 4, N_steps)
timestamps = data['timestamps']
states = data['states']
controls = data['controls']

num_drones = states.shape[0]
num_steps = states.shape[2]
print(f"Données chargées : {num_drones} drones, {num_steps} pas de temps.")

# --- Visualisation 1 : Trajectoires (Vue de dessus X-Y) ---
plt.figure(figsize=(10, 8))
# Positions X et Y (indices 0 et 1 de la dimension 1 de 'states')
pos_x = states[:, 0, :]
pos_y = states[:, 1, :]

for i in range(num_drones):
    plt.plot(pos_x[i], pos_y[i], label=f'Drone {i}')
    # Marquer le début (o) et la fin (x)
    plt.scatter(pos_x[i, 0], pos_y[i, 0], marker='o')
    plt.scatter(pos_x[i, -1], pos_y[i, -1], marker='x')

plt.title(f"Trajectoires - {os.path.basename(filename)}")
plt.xlabel("X [m]")
plt.ylabel("Y [m]")
plt.axis('equal')
plt.grid(True)
plt.legend()
plt.show()

# --- Visualisation 2 : Paramètre d'ordre (Alignement) ---
# Calcul de la cohérence du groupe (0 = désordonné, 1 = tous dans la même direction)
vx = states[:, 3, :]
vy = states[:, 4, :]

speed = np.sqrt(vx**2 + vy**2)

# Normalisation pour obtenir les vecteurs directionnels unitaires
# On évite la division par zéro avec 'where'
u_vx = np.divide(vx, speed, out=np.zeros_like(vx), where=speed!=0)
u_vy = np.divide(vy, speed, out=np.zeros_like(vy), where=speed!=0)

# Moyenne des vecteurs directionnels à chaque pas de temps
avg_u_vx = np.mean(u_vx, axis=0)
avg_u_vy = np.mean(u_vy, axis=0)

# La norme de ce vecteur moyen est le paramètre d'ordre
order_parameter = np.sqrt(avg_u_vx**2 + avg_u_vy**2)

plt.figure(figsize=(10, 4))
plt.plot(timestamps[0, :], order_parameter)
plt.title("Paramètre d'Ordre (Alignement des vitesses)")
plt.xlabel("Temps [s]")
plt.ylabel("Alignement [0-1]")
plt.ylim(0, 1.1)
plt.grid(True)
plt.show()