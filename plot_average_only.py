
import json
import matplotlib.pyplot as plt
import os
import numpy as np

def load_data(json_path):
    with open(json_path, 'r') as f:
        return json.load(f)

def compute_average_curve(history, metric_name):
    if not history:
        return [], []
    min_epochs = min(len(fold) for fold in history)
    avg_values = []
    epochs = []
    for epoch_idx in range(min_epochs):
        val_sum = 0
        count = 0
        epoch_num = history[0][epoch_idx]['epoch']
        for fold_data in history:
            val_sum += fold_data[epoch_idx][metric_name]
            count += 1
        avg_values.append(val_sum / count)
        epochs.append(epoch_num)
    return epochs, avg_values

def plot_single_model_train_vs_val_avg(data, augmentator_name, model_name, output_path, title_override=None):
    try:
        model_data = data[augmentator_name][model_name]
        history = model_data['training_history']
    except KeyError:
        print(f"Data not found for {augmentator_name} - {model_name}")
        if augmentator_name in data:
            print(f"Available models in {augmentator_name}: {list(data[augmentator_name].keys())}")
        else:
             print(f"Augmentator {augmentator_name} not found in data keys: {list(data.keys())}")
        return

    epochs, avg_train = compute_average_curve(history, 'train_loss')
    _, avg_val = compute_average_curve(history, 'val_loss')

    plt.figure(figsize=(8, 6))
    plt.plot(epochs, avg_train, label='Avg Train Loss', marker='o', linestyle='-', color='blue')
    plt.plot(epochs, avg_val, label='Avg Val Loss', marker='x', linestyle='--', color='orange')
    
    title = title_override if title_override else f"{augmentator_name}\n{model_name} - Average Loss"
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.ylim(bottom=0)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")

def plot_comparison_val_loss_avg(data, augmentator_name, output_path, title_override=None):
    try:
        aug_data = data[augmentator_name]
    except KeyError:
        print(f"Data not found for {augmentator_name}")
        return

    plt.figure(figsize=(10, 6))
    
    # Pre-defined colors/styles for consistency if desired, or let matplotlib handle it
    styles = ['-', '--', '-.', ':']
    markers = ['o', 's', '^', 'D']
    
    idx = 0
    for model_name, model_data in aug_data.items():
        if 'training_history' not in model_data or not model_data['training_history']:
            continue
            
        history = model_data['training_history']
        epochs, avg_val = compute_average_curve(history, 'val_loss')
        
        plt.plot(epochs, avg_val, label=f"{model_name}", 
                 linestyle=styles[idx % len(styles)], 
                 marker=markers[idx % len(markers)],
                 markevery=5) # don't clutter with markers
        idx += 1

    title = title_override if title_override else f"Validation Loss Comparison - {augmentator_name}"
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.ylim(bottom=0)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")

if __name__ == "__main__":
    JSON_FILE = "results/results_cache.json"
    REPORT_DIR = "docs/report"
    
    data = load_data(JSON_FILE)
    
    # 1. Comparison: Identity (Val Loss)
    plot_comparison_val_loss_avg(
        data, 
        augmentator_name="Aug_Identity_K5", 
        output_path=os.path.join(REPORT_DIR, "val_loss_identity.png"),
        title_override="Validation Loss - Identity Augmentation (Average of 5 Folds)"
    )
    
    # 2. Comparison: Combined (Val Loss)
    plot_comparison_val_loss_avg(
        data, 
        augmentator_name="Combined_2Geo_2Photo_1SEM_x2", 
        output_path=os.path.join(REPORT_DIR, "val_loss_combined_2geo.png"),
        title_override="Validation Loss - Combined Augmentation (Average of 5 Folds)"
    )

    # 3. Swin Standard Train vs Val
    plot_single_model_train_vs_val_avg(
        data,
        augmentator_name="Combined_2Geo_2Photo_1SEM_x2",
        model_name="Swin_Model_Node",
        output_path=os.path.join(REPORT_DIR, "train_val_swin_std.png"),
        title_override="Swin Standard: Train vs Val (Average)"
    )

    # 4. Swin Large Train vs Val
    plot_single_model_train_vs_val_avg(
        data,
        augmentator_name="Combined_2Geo_2Photo_1SEM_x2",
        model_name="Swin_Big_Model_Node",
        output_path=os.path.join(REPORT_DIR, "train_val_swin_large.png"),
        title_override="Swin Large: Train vs Val (Average)"
    )
