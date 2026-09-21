import numpy as np

class UncertaintyInterpreter:
    def __init__(self):
        self.threshold = None
        self.best_acc = 0.0

    def fit(self, entropies_known: np.ndarray, entropies_unknown: np.ndarray):
        """
        Doitu eredu hau muga optimo bat aurkitzeko, entropiaren arabera
        datu ezagunak (0) eta ezezagunak (1) bereizteko.
        """
        all_entropies = np.concatenate([entropies_known, entropies_unknown])
        y_true = np.concatenate([np.zeros(len(entropies_known)), np.ones(len(entropies_unknown))])
        min_e, max_e = np.min(all_entropies), np.max(all_entropies)
        thresholds = np.linspace(min_e, max_e, 100)
        
        best_threshold = min_e
        best_accuracy = 0.0
        
        for t in thresholds:
            y_pred = (all_entropies > t).astype(int)
            accuracy = np.mean(y_pred == y_true)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_threshold = t
                
        self.threshold = best_threshold
        self.best_acc = best_accuracy
        return best_threshold, best_accuracy

    def predict(self, entropies: np.ndarray):
        """
        Iragarri ea laginak ezezagunak (1) edo ezagunak (0) diren, mugaren arabera.
        """
        if self.threshold is None:
            raise ValueError("Eredua aurretik doitu ('fit') egin behar da muga finkatzeko.")
        return (entropies > self.threshold).astype(int)

    def evaluate(self, entropies_known: np.ndarray, entropies_unknown: np.ndarray):
        """
        Ikusgai eta itzalean dauden datuak hartu, muga optimoa aurkitu
        eta metrikak itzuliko ditu (Zehaztasuna).
        """
        if self.threshold is None:
            self.fit(entropies_known, entropies_unknown)
            
        y_pred_known = self.predict(entropies_known)
        y_pred_unknown = self.predict(entropies_unknown)
        
        acc_known = np.mean(y_pred_known == 0)
        acc_unknown = np.mean(y_pred_unknown == 1) if len(entropies_unknown) > 0 else 0
        
        total_acc = (np.sum(y_pred_known == 0) + np.sum(y_pred_unknown == 1)) / (len(entropies_known) + len(entropies_unknown))
        
        return {
            'threshold': self.threshold,
            'accuracy': total_acc,
            'acc_known': acc_known,
            'acc_unknown': acc_unknown
        }
