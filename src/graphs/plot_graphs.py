import pandas as pd
import matplotlib.pyplot as plt
import os
import glob

# 1. Set up the folders
metrics_folder = 'results/metrics'
figures_folder = 'results/figures'

# Make sure the figures folder exists so we don't get an error
os.makedirs(figures_folder, exist_ok=True)

# 2. Find all the CSV files
csv_files = glob.glob(os.path.join(metrics_folder, '*.csv'))
print(f"Found {len(csv_files)} CSV files. Drawing graphs now...")

# 3. Loop through every single file automatically
for file_path in csv_files:
    # Get just the file name to use as a title
    base_name = os.path.basename(file_path).replace('.csv', '')
    
    # Read the numbers from the CSV
    df = pd.read_csv(file_path)
    
    # --- GRAPH A: ACCURACY ---
    plt.figure(figsize=(10, 5))
    plt.plot(df['epoch'], df['train_acc'], label='Train Accuracy', color='blue', linewidth=2)
    plt.plot(df['epoch'], df['val_acc'], label='Validation Accuracy', color='orange', linewidth=2)
    
    plt.title(f'Accuracy: {base_name}')
    plt.ylim(0, 1.0)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    # Save it to the figures folder
    plt.savefig(os.path.join(figures_folder, f"{base_name}_accuracy.png"))
    plt.close() # Close the graph so your Mac's memory doesn't get full
    
    # --- GRAPH B: LOSS ---
    plt.figure(figsize=(10, 5))
    plt.plot(df['epoch'], df['train_loss'], label='Train Loss', color='blue', linewidth=2)
    plt.plot(df['epoch'], df['val_loss'], label='Validation Loss', color='orange', linewidth=2)
    
    plt.title(f'Loss: {base_name}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Save it to the figures folder
    plt.savefig(os.path.join(figures_folder, f"{base_name}_loss.png"))
    plt.close()

print("Success! All graphs have been saved to the results/figures folder.")
