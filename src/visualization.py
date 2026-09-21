# src/visulization.py

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix

sns.set_theme(style="whitegrid")

def plot_data_distribution(df: pd.DataFrame, title: str = "Sortutako Datuen Banaketa"):
    plt.figure(figsize=(10, 6))
    plt.scatter(df['x'], df['y'], color='blue', alpha=0.6, label="Zaratadun datuak", s=15)
    
    plt.title(title, fontsize=14)
    plt.xlabel('X (Irudia)', fontsize=12)
    plt.ylabel('Y (Helburua)', fontsize=12)
    plt.legend()

    plt.tight_layout()
    plt.show()

def plot_comparision(df: pd.DataFrame, y_true_col: str = ""):
    pass

def plot_shadow_distribution(df: pd.DataFrame, is_shadow: np.ndarray, title: str = "Informazio Itzala"):
    plt.figure(figsize=(10, 6))
    visible = df[~is_shadow]
    shadow = df[is_shadow]
    
    plt.scatter(visible['x'], visible['y'], c='blue', alpha=0.6, label='Ikusgai', s=15)
    
    plt.scatter(shadow['x'], shadow['y'], c='red', alpha=0.6, label='Informazio Itzala', s=15)
    
    plt.title(title, fontsize=14)
    plt.xlabel('X (Irudia)', fontsize=12)
    plt.ylabel('Y (Helburua)', fontsize=12)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_split_distribution(split_data, title: str = "Datuen Banaketa"):
    plt.figure(figsize=(10, 6))
    
    plt.scatter(split_data.X_train, split_data.y_train, 
                color='blue', alpha=0.5, label='Training Set', s=15)
                
    plt.scatter(split_data.X_test_visible, split_data.y_test_visible, 
                color='cyan', edgecolor='blue', alpha=0.6, 
                label='Test Set (Ikusgai)', s=20, marker='s')
                
    plt.scatter(split_data.X_test_shadow, split_data.y_test_shadow, 
                color='red', alpha=0.7, label='Test Set (Itzala)', s=25, marker='x')

    plt.title(title, fontsize=14)
    plt.xlabel('X (Irudia)', fontsize=12)
    plt.ylabel('Y (Helburua)', fontsize=12)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_experiment_results(split_data, predictions: dict, uncertainties=None, title: str = "Esperimentuaren Emaitzak"):
    plt.figure(figsize=(12, 8))
    plt.scatter(split_data.X_train, split_data.y_train, 
                color='blue', alpha=0.3, label='Train Data (Erreala)', s=15)
    plt.scatter(split_data.X_test_shadow, split_data.y_test_shadow, 
                color='red', alpha=0.6, label='Shadow Real Data (Ezkutua)', s=25, marker='x')
    
    x_all = np.concatenate([
        split_data.X_train.values.flatten(),
        split_data.X_test_visible.values.flatten(),
        split_data.X_test_shadow.values.flatten()
    ])
    y_pred_all = np.concatenate([
        predictions['train'],
        predictions['visible'],
        predictions['shadow']
    ])
    
    sort_idx = np.argsort(x_all)
    x_all_sorted = x_all[sort_idx]
    y_pred_all_sorted = y_pred_all[sort_idx]

    plt.plot(x_all_sorted, y_pred_all_sorted, 
             color='green', label='Model Prediction (Inferentzia)', linewidth=2)
    
    if uncertainties is not None:
        y_std_all = np.concatenate([
            uncertainties['train'],
            uncertainties['visible'],
            uncertainties['shadow']
        ])
        y_std_all_sorted = y_std_all[sort_idx]

        plt.fill_between(
            x_all_sorted,
            y_pred_all_sorted - 2 * y_std_all_sorted,
            y_pred_all_sorted + 2 * y_std_all_sorted,
            color='red', alpha=0.2, label='Ezjakintasuna (±2σ)'
        )

    plt.title(title, fontsize=14)
    plt.xlabel("X", fontsize=12)
    plt.ylabel("Y", fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_mnist_classification_results(split_data, predictions: dict, uncertainties: dict, title: str = "MNIST Sailkapena: 3 digitua ezkutatuta"):
    import seaborn as sns
    from sklearn.metrics import confusion_matrix
    
    y_test_visible = split_data.y_test_visible
    pred_vis = predictions['visible']
    
    y_test_shadow = split_data.y_test_shadow
    pred_shadow = predictions['shadow']
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    cm_vis = confusion_matrix(y_test_visible, pred_vis, labels=np.arange(10))
    sns.heatmap(cm_vis, annot=True, fmt='d', cmap='Blues', ax=axes[0])
    axes[0].set_title(f"{title}\nKonfusio Matrizea (Ikusgai)")
    axes[0].set_xlabel('Iragarpena')
    axes[0].set_ylabel('Erreala')
    
    ent_vis = uncertainties['visible']
    ent_shadow = uncertainties['shadow']
    
    sns.kdeplot(ent_vis, label='Entropia (Ikusgai)', ax=axes[1], fill=True, color='blue', alpha=0.4)
    sns.kdeplot(ent_shadow, label='Entropia (Itzala)', ax=axes[1], fill=True, color='red', alpha=0.4)
    axes[1].set_title("Ziurgabetasuna (Entropia) Banaketa")
    axes[1].set_xlabel("Entropia")
    axes[1].set_ylabel("Dentsitatea")
    axes[1].legend()
    
    plt.tight_layout()
    plt.show()

def plot_classification_results(split_data, predictions, uncertainties, title="Klasifikazio Emaitzak"):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    cm = confusion_matrix(split_data.y_test_visible, predictions['visible'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0], cbar=False)
    axes[0].set_title(f"{title}\nKonfusio Matrizea (Ikusgai)")
    axes[0].set_xlabel('Iragarpena')
    axes[0].set_ylabel('Erreala')
    
    ent_vis = uncertainties['visible']
    ent_shadow = uncertainties['shadow']
    
    sns.kdeplot(ent_vis, label='Entropia (Ikusgai)', ax=axes[1], fill=True, color='blue', alpha=0.4)
    sns.kdeplot(ent_shadow, label='Entropia (Itzala)', ax=axes[1], fill=True, color='red', alpha=0.4)
    axes[1].set_title("Ziurgabetasuna (Entropia) Banaketa")
    axes[1].set_xlabel("Entropia")
    axes[1].set_ylabel("Dentsitatea")
    axes[1].legend()
    
    plt.tight_layout()
    plt.show()

def plot_image_sample_uncertainties(split_data, predictions, uncertainties, num_samples=5):
    """
    Formatu anitzeko irudiak bistaratzeko funtzioa (MNIST edo CIFAR-10).
    X-ren dimentsioen arabera irudiaren itxura moldatzen du.
    """
    X_val = split_data.X_test_shadow.values
    input_dim = X_val.shape[1]
    
    if input_dim == 784:
        X_shadow = X_val.reshape(-1, 28, 28)
        cmap = 'gray'
    elif input_dim == 3072:
        X_shadow = X_val.reshape(-1, 32, 32, 3)
        cmap = None
    else:
        raise ValueError(f"Dimentsio ezezaguna irudientzat: {input_dim}. Ez da ez MNIST (784) ez CIFAR-10 (3072).")

    y_shadow = split_data.y_test_shadow.values if hasattr(split_data.y_test_shadow, 'values') else split_data.y_test_shadow
    pred_shadow = predictions['shadow']
    ent_shadow = uncertainties['shadow']
    
    indices = np.random.choice(len(X_shadow), min(num_samples, len(X_shadow)), replace=False)
    
    fig, axes = plt.subplots(1, len(indices), figsize=(15, 3))
    if len(indices) == 1:
        axes = [axes]
        
    for i, idx in enumerate(indices):
        img = X_shadow[idx]
        
        if cmap is None:
             img_arr = np.array(img, dtype=float)
             img_min = img_arr.min()
             img_max = img_arr.max()
             if img_max > img_min:
                 img_arr = (img_arr - img_min) / (img_max - img_min)
                 
             img_arr = np.clip(img_arr, 0.0, 1.0)
             axes[i].imshow(img_arr)
        else:
             axes[i].imshow(img, cmap=cmap)
             
        axes[i].set_title(f"Err: {y_shadow[idx]} | Irag: {pred_shadow[idx]}\nEntr: {ent_shadow[idx]:.2f}", 
                          fontsize=10, 
                          color='red' if y_shadow[idx] != pred_shadow[idx] else 'green')
        axes[i].axis('off')
        
    plt.tight_layout()
    plt.show()

plot_mnist_classification_results = plot_classification_results
plot_mnist_sample_uncertainties = plot_image_sample_uncertainties

def plot_weka_pca(split_data, title="Weka Datuen PCA Banaketa"):
    from sklearn.decomposition import PCA
    
    X_tr = split_data.X_train.values if hasattr(split_data.X_train, 'values') else split_data.X_train
    X_vis = split_data.X_test_visible.values if hasattr(split_data.X_test_visible, 'values') else split_data.X_test_visible
    X_sh = split_data.X_test_shadow.values if hasattr(split_data.X_test_shadow, 'values') else split_data.X_test_shadow
    
    if X_tr.shape[1] < 2:
        print("Datuak 2 dimentsio baino gutxiago ditu, ezin da PCA 2D-n marraztu.")
        return
        
    pca = PCA(n_components=2)
    pca.fit(X_tr)
    
    tr_pca = pca.transform(X_tr)
    vis_pca = pca.transform(X_vis)
    sh_pca = pca.transform(X_sh)
    
    plt.figure(figsize=(8, 6))
    plt.scatter(tr_pca[:,0], tr_pca[:,1], c='blue', alpha=0.3, label='Train', s=10)
    plt.scatter(vis_pca[:,0], vis_pca[:,1], c='cyan', alpha=0.5, label='Test Visible', s=15)
    plt.scatter(sh_pca[:,0], sh_pca[:,1], c='red', alpha=0.7, label='Test Shadow', s=20)
    
    plt.title(title, fontsize=14)
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    plt.legend()
    plt.tight_layout()
    plt.show()