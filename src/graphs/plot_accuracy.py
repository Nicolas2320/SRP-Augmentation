import pandas as pd
import matplotlib.pyplot as plt

# Load CSV
df = pd.read_csv(
    "results/metrics/cifar10_resnet18_kN_seed0_cutmix_epochs30.csv"
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
    "ResNet18 - CutMix on CIFAR-10 (k=N)"
)

plt.legend()

plt.grid(True)

plt.show()

# Save plot in `results/figures/`