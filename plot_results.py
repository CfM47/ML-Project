
import json
import matplotlib.pyplot as plt
import os
import math
import numpy as np

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")

def load_data(json_path):
    try:
        with open(json_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found at {json_path}")
        return None
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON file at {json_path}")
        return None

def get_plot_layout(num_plots):
    cols = 2 if num_plots > 1 else 1
    rows = math.ceil(num_plots / cols)
    return rows, cols

def compute_average_curve(history, metric_name):
    """
    Computes average curve across folds.
    Returns: epochs (list), avg_values (list)
    """
    if not history:
        return [], []
    
    # Find min number of epochs to ensure we can average
    min_epochs = min(len(fold) for fold in history)
    
    avg_values = []
    epochs = []
    
    for epoch_idx in range(min_epochs):
        # Collect values for this epoch from all folds
        val_sum = 0
        count = 0
        epoch_num = history[0][epoch_idx]['epoch'] # Assume all have same epoch numbering
        
        for fold_data in history:
            val_sum += fold_data[epoch_idx][metric_name]
            count += 1
            
        avg_values.append(val_sum / count)
        epochs.append(epoch_num)
        
    return epochs, avg_values

def plot_train_vs_val(data, base_output_dir):
    """
    1. Train Loss vs Val Loss for each Augmentator + Model.
    Includes Average plot.
    """
    output_dir = os.path.join(base_output_dir, "train_vs_val")
    ensure_dir(output_dir)
    
    print("Generating: Train vs Val plots...")

    for augmentator_name, augmentator_data in data.items():
        for model_name, model_data in augmentator_data.items():
            
            if 'training_history' not in model_data:
                continue
            
            training_history = model_data['training_history']
            num_folds = len(training_history)
            
            if num_folds == 0:
                 continue

            # Add one for the Average plot
            total_plots = num_folds + 1
            rows, cols = get_plot_layout(total_plots)
            fig, axes = plt.subplots(rows, cols, figsize=(12, 6 * rows))
            fig.suptitle(f"{augmentator_name} - {model_name}\nTrain vs Val Loss", fontsize=16)
            
            if total_plots > 1:
                axes_flat = axes.flatten()
            else:
                axes_flat = [axes]

            # Plot Individual Folds
            for i, fold_epochs in enumerate(training_history):
                ax = axes_flat[i]
                
                epochs = [e['epoch'] for e in fold_epochs]
                train_loss = [e['train_loss'] for e in fold_epochs]
                val_loss = [e['val_loss'] for e in fold_epochs]
                
                ax.plot(epochs, train_loss, label='Train Loss', marker='o', linestyle='-')
                ax.plot(epochs, val_loss, label='Val Loss', marker='x', linestyle='--')
                
                ax.set_title(f"Fold {i+1}")
                ax.set_xlabel("Epoch")
                ax.set_ylabel("Loss")
                ax.legend()
                ax.grid(True)
                ax.set_ylim(bottom=0)
            
            # Plot Average
            ax_avg = axes_flat[num_folds]
            avg_epochs, avg_train = compute_average_curve(training_history, 'train_loss')
            _, avg_val = compute_average_curve(training_history, 'val_loss')
            
            ax_avg.plot(avg_epochs, avg_train, label='Avg Train Loss', marker='o', linestyle='-', color='purple')
            ax_avg.plot(avg_epochs, avg_val, label='Avg Val Loss', marker='x', linestyle='--', color='orange')
            
            ax_avg.set_title(f"Average ({num_folds} folds)")
            ax_avg.set_xlabel("Epoch")
            ax_avg.set_ylabel("Loss")
            ax_avg.legend()
            ax_avg.grid(True)
            ax_avg.set_ylim(bottom=0)

            # Hide unused subplots
            for j in range(total_plots, len(axes_flat)):
                axes_flat[j].axis('off')

            plt.tight_layout(rect=[0, 0.03, 1, 0.97])
            
            filename = f"{augmentator_name}_{model_name}_loss.png".replace(" ", "_").replace("/", "-")
            save_path = os.path.join(output_dir, filename)
            plt.savefig(save_path)
            plt.close(fig)

def plot_comparisons(data, group_by, metric, base_output_dir):
    """
    Generates comparison plots.
    Includes Average plot in the last subplot.
    """
    
    if group_by == 'augmentator':
        folder_name = f"compare_augmentations_{metric.split('_')[0]}" 
        primary_key_type = "Model"
        secondary_key_type = "Augmentator"
        
        items = {}
        for aug_name, aug_data in data.items():
            for mod_name, mod_data in aug_data.items():
                if mod_name not in items:
                    items[mod_name] = {}
                items[mod_name][aug_name] = mod_data
                
    elif group_by == 'model':
        folder_name = f"compare_models_{metric.split('_')[0]}"
        primary_key_type = "Augmentator"
        secondary_key_type = "Model"
        items = data 
    else:
        return

    output_dir = os.path.join(base_output_dir, folder_name)
    ensure_dir(output_dir)
    print(f"Generating: {folder_name} plots...")

    for primary_name, secondary_dict in items.items():
        
        max_folds = 0
        valid_entries = []
        
        for sec_name, sec_data in secondary_dict.items():
            if 'training_history' in sec_data and len(sec_data['training_history']) > 0:
                max_folds = max(max_folds, len(sec_data['training_history']))
                valid_entries.append(sec_name)
        
        if max_folds == 0:
            continue

        # Add one for Average plot
        total_plots = max_folds + 1
        rows, cols = get_plot_layout(total_plots)
        fig, axes = plt.subplots(rows, cols, figsize=(12, 6 * rows))
        metric_title = "Validation Loss" if metric == 'val_loss' else "Training Loss"
        fig.suptitle(f"{primary_key_type}: {primary_name}\nComparing {secondary_key_type}s ({metric_title})", fontsize=16)
        
        if total_plots > 1:
            axes_flat = axes.flatten()
        else:
            axes_flat = [axes]
            
        # Draw plots for each fold
        for fold_idx in range(max_folds):
            ax = axes_flat[fold_idx]
            ax.set_title(f"Fold {fold_idx+1}")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.grid(True)
            ax.set_ylim(bottom=0)
            
            has_data = False
            for sec_name in valid_entries:
                sec_data = secondary_dict[sec_name]
                history = sec_data['training_history']
                
                if fold_idx < len(history):
                    fold_data = history[fold_idx]
                    epochs = [e['epoch'] for e in fold_data]
                    values = [e[metric] for e in fold_data]
                    ax.plot(epochs, values, label=sec_name)
                    has_data = True
            
            if has_data:
                ax.legend(fontsize='small')

        # Draw Average Plot
        ax_avg = axes_flat[max_folds]
        ax_avg.set_title(f"Average (All Folds)")
        ax_avg.set_xlabel("Epoch")
        ax_avg.set_ylabel("Loss")
        ax_avg.grid(True)
        ax_avg.set_ylim(bottom=0)
        
        has_avg_data = False
        for sec_name in valid_entries:
            sec_data = secondary_dict[sec_name]
            history = sec_data['training_history']
            
            if history:
                 avg_epochs, avg_values = compute_average_curve(history, metric)
                 if avg_epochs:
                     ax_avg.plot(avg_epochs, avg_values, label=sec_name)
                     has_avg_data = True
        
        if has_avg_data:
            ax_avg.legend(fontsize='small')

        # Hide unused subplots
        for j in range(total_plots, len(axes_flat)):
            axes_flat[j].axis('off')

        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        
        filename = f"{primary_name}_{metric}.png".replace(" ", "_").replace("/", "-")
        save_path = os.path.join(output_dir, filename)
        plt.savefig(save_path)
        plt.close(fig)

if __name__ == "__main__":
    JSON_FILE = "results/results_cache.json"
    PLOTS_DIR = "plots"
    
    data = load_data(JSON_FILE)
    if data:
        # 1. Train vs Val (Original)
        plot_train_vs_val(data, PLOTS_DIR)
        
        # 2. Compare Augmentations (Val Loss)
        plot_comparisons(data, group_by='augmentator', metric='val_loss', base_output_dir=PLOTS_DIR)
        
        # 3. Compare Models (Val Loss)
        plot_comparisons(data, group_by='model', metric='val_loss', base_output_dir=PLOTS_DIR)
        
        # 4. Compare Augmentations (Train Loss)
        plot_comparisons(data, group_by='augmentator', metric='train_loss', base_output_dir=PLOTS_DIR)
        
        # 5. Compare Models (Train Loss)
        plot_comparisons(data, group_by='model', metric='train_loss', base_output_dir=PLOTS_DIR)
        
        print("All plots generated successfully.")
