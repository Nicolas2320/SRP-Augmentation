import pandas as pd
import matplotlib.pyplot as plt

# Load CSV
df = pd.read_csv(
    "results\metrics\cifar100_resnet50_k50_seed0_none_epochs100.csv"
)

# =====================================================
# ACCURACY GRAPH
# =====================================================

plt.figure(figsize=(10, 6))

plt.plot(
    df["epoch"],
    df["train_acc"],
    label="Train Accuracy"
)

plt.plot(
    df["epoch"],
    df["val_acc"],
    label="Validation Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.title(
    "ResNet50 - No Augmentation on CIFAR-100 (k=50)"
)

plt.legend()

plt.grid(True)

plt.show()

# Save plot in `results/figures/`