import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

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

def plot_experiment_results(split_data, predictions: dict, title: str = "Esperimentuaren Emaitzak"):
    plt.figure(figsize=(12, 8))
    
    plt.scatter(split_data.X_train, split_data.y_train, 
                color='blue', alpha=0.3, label='Train Data (Erreala)', s=15)
    plt.scatter(split_data.X_test_shadow, split_data.y_test_shadow, 
                color='red', alpha=0.6, label='Shadow Real Data (Ezkutua)', s=25, marker='x')
    
    if 'shadow' in predictions:
        plt.scatter(split_data.X_test_shadow, predictions['shadow'], 
                    color='green', s=15, label='Model Prediction (Inferentzia)')
    
    plt.title(title, fontsize=14)
    plt.xlabel("X", fontsize=12)
    plt.ylabel("Y", fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
