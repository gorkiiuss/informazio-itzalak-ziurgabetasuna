import os
import sys
import numpy as np
import joblib

# Go up twice to reach the root 'gral' folder so modules can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.generators import MNISTGenerator
from src.classifiers import RandomForestClassifierModel, NeuralNetworkMCDropoutClassifier
from src.ensemble import EnsembleClassifier
from src.interpreter import UncertaintyInterpreter

def main():
    print("Prestatzen: Ereduak eta Interpretatzailea Web Aplikaziorako...")
    
    # 1. Kargatu datuak
    print("-> MNIST dataset argia kargatzen...")
    gen_train = MNISTGenerator(train=True)
    gen_test = MNISTGenerator(train=False)
    
    df_train = gen_train.generate(n_samples=-1)
    df_test = gen_test.generate(n_samples=-1)
    
    y_train = df_train['y']
    x_train_flat = df_train.drop(columns=['y']).values
    
    x_test_flat = df_test.drop(columns=['y']).values
    
    # 2. Ereduak definitu eta trebatu (Datu OSOTAN, itzal barik)
    rfc = RandomForestClassifierModel(n_estimators=50, max_depth=15, random_state=42)
    # Parametro argiagoekin aplikazio azkarra lortzeko
    nn = NeuralNetworkMCDropoutClassifier(hidden_dim=128, num_classes=10, dropout_rate=0.3, epochs=10, n_iter=20)
    
    ensemble = EnsembleClassifier([rfc, nn])
    
    print("-> RFC eta Sare Neuronala trebatzen (Gerta daiteke minutu batzuk luzatzea)...")
    ensemble.fit(x_train_flat, y_train)
    
    # 3. Interpretatzailea Doitu
    print("-> Entropiak kalkulatzen interpretatzailea (UncertaintyInterpreter) doitzeko...")
    # 'Ezagunak' (Test set)
    _, entropies_known = ensemble.predict_proba_with_uncertainty(x_test_flat)
    
    # 'Ezezagunak' (Zarata uniformea sortu MNIST balioen tartean)
    # Zaratak entropia maximoa bultzatu beharko luke ensemblean (RFCren erabakietan bereziki)
    random_noise = np.random.rand(2000, 28*28).astype(np.float32)
    _, entropies_unknown = ensemble.predict_proba_with_uncertainty(random_noise)
    
    interpreter = UncertaintyInterpreter()
    best_threshold, best_acc = interpreter.fit(entropies_known, entropies_unknown)
    print(f"-> Interpretatzailearen Muga (Threshold) optimoa: {best_threshold:.4f} (Zehaztasuna OOD bereizketan: {best_acc:.4f})")
    
    # 4. Gorde dena 'models' karpetan
    models_dir = os.path.join(os.path.dirname(__file__), 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    ensemble_path = os.path.join(models_dir, 'ensemble.pkl')
    interpreter_path = os.path.join(models_dir, 'interpreter.pkl')
    
    print(f"-> Ereduak gordetzen honeko bidean: {models_dir} ...")
    joblib.dump(ensemble, ensemble_path)
    joblib.dump(interpreter, interpreter_path)
    print("Dena prest! Ereduak esportatu dira.")

if __name__ == '__main__':
    main()
