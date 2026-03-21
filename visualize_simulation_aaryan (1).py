# ==========================================
# VISUAL VALIDATION SCRIPT (Colab Ready)
# ==========================================

import pandas as pd
import matplotlib.pyplot as plt

# ==========================================
# LOAD DATA
# ==========================================
# In Colab, upload the CSV manually when prompted
from google.colab import files

print("📂 Please upload 'simulated_stream.csv'")
uploaded = files.upload()

file_name = list(uploaded.keys())[0]
df = pd.read_csv(file_name)

# ==========================================
# PREPROCESS
# ==========================================
df = df.sort_values(by=["timestamp", "node"])

node_L = df[df["node"] == "Node_L"]
node_M = df[df["node"] == "Node_M"]
node_H = df[df["node"] == "Node_H"]

# ==========================================
# FEATURES TO VISUALIZE
# ==========================================
features = [
    "Air temperature [K]",
    "Torque [Nm]",
    "Rotational speed [rpm]"
]

# ==========================================
# PLOTTING
# ==========================================
for feature in features:
    plt.figure(figsize=(14, 5))

    plt.plot(node_L["timestamp"], node_L[feature], label="Node_L")
    plt.plot(node_M["timestamp"], node_M[feature], label="Node_M")
    plt.plot(node_H["timestamp"], node_H[feature], label="Node_H")

    # Mark failures
    fail_L = node_L[node_L["Machine failure"] == 1]
    fail_M = node_M[node_M["Machine failure"] == 1]
    fail_H = node_H[node_H["Machine failure"] == 1]

    plt.scatter(fail_L["timestamp"], fail_L[feature], marker='x', s=80)
    plt.scatter(fail_M["timestamp"], fail_M[feature], marker='x', s=80)
    plt.scatter(fail_H["timestamp"], fail_H[feature], marker='x', s=80)

    # Mark cascade trigger (first Node_L failure)
    if not fail_L.empty:
        t0 = fail_L["timestamp"].iloc[0]
        plt.axvline(x=t0, linestyle="--")

    plt.title(f"Cascading Failure Visualization: {feature}")
    plt.xlabel("Time")
    plt.ylabel(feature)
    plt.legend()
    plt.grid()

    plt.show()

print("✅ Visualization complete!")
