import os
import sys
import base64
import io
import argparse
import time

import numpy as np
import joblib
from PIL import Image, ImageOps
from flask import Flask, request, jsonify, render_template

# Add root folder to sys.path so src imports work
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

app = Flask(__name__)
DEBUG_MODE = False

import json
import uuid
import datetime

# Ereduen ibilbideak kargatzeko
MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')
INTERPRETER_PATH = os.path.join(MODELS_DIR, 'interpreter.pkl')
METADATA_PATH = os.path.join(MODELS_DIR, 'metadata.json')

ACTIVE_MODEL_ID = 'v1'
tree_metadata = {}
inference_history = {}

ensemble = None
interpreter = None

# Active Continual Learning states
label_to_index = {str(i): i for i in range(10)}
index_to_label = {i: str(i) for i in range(10)}

subset_x = None
subset_y = None

def get_subset():
    global subset_x, subset_y
    if subset_x is None:
        from src.generators import MNISTGenerator
        print("Lokalizatzen: MNIST subset entrenamendu azkarretarako (2000 lagin)")
        gen = MNISTGenerator(train=True)
        df = gen.generate(n_samples=2000)
        subset_y = df['y'].values
        subset_x = df.drop(columns=['y']).values.astype(np.float32)
    return subset_x, subset_y

def load_metadata():
    global tree_metadata, ACTIVE_MODEL_ID, inference_history
    if os.path.exists(METADATA_PATH):
        with open(METADATA_PATH, 'r') as f:
            data = json.load(f)
            tree_metadata = data.get('tree', {})
            ACTIVE_MODEL_ID = data.get('active_id', 'v1')
            inference_history = data.get('history', {})
    else:
        # Initialize root
        tree_metadata = {
            'v1': {
                'id': 'v1',
                'parent_id': None,
                'name': 'Base MNIST Ensemble',
                'capabilities': [],
                'metrics': {'accuracy': 0.99, 'entropy': 0.1},
                'timestamp': datetime.datetime.now().isoformat()
            }
        }
        ACTIVE_MODEL_ID = 'v1'
        inference_history = {}
        save_metadata()

def save_metadata():
    with open(METADATA_PATH, 'w') as f:
        json.dump({
            'tree': tree_metadata,
            'active_id': ACTIVE_MODEL_ID,
            'history': inference_history
        }, f, indent=2)

def load_models():
    global ensemble, interpreter
    load_metadata()
    
    if interpreter is None:
        try:
            interpreter = joblib.load(INTERPRETER_PATH)
        except Exception as e:
            print(f"Errorea interpretatzailea kargatzean: {e}")
            
    # Load specific ensemble version based on ACTIVE root
    ensemble_filename = 'ensemble.pkl' if ACTIVE_MODEL_ID == 'v1' else f'ensemble_{ACTIVE_MODEL_ID}.pkl'
    try:
        ensemble = joblib.load(os.path.join(MODELS_DIR, ensemble_filename))
        print(f"Eredua ({ACTIVE_MODEL_ID}) eta interpretatzailea arrakastaz kargatu dira.")
    except Exception as e:
        print(f"Errorea {ensemble_filename} eredua kargatzean: {e}")
        ensemble = None

def parse_image(image_b64):
    image_data = image_b64.split(",")[1]
    image_bytes = base64.b64decode(image_data)
    image = Image.open(io.BytesIO(image_bytes))
    
    # Griseskala eta inbertsioa (mnist bezala)
    image = image.convert('L')
    image = ImageOps.invert(image)
    
    img_array_pre = np.array(image)
    non_empty_columns = np.where(img_array_pre.max(axis=0) > 0)[0]
    non_empty_rows = np.where(img_array_pre.max(axis=1) > 0)[0]
    
    if len(non_empty_columns) > 0 and len(non_empty_rows) > 0:
        bbox = (non_empty_columns.min(), non_empty_rows.min(), non_empty_columns.max(), non_empty_rows.max())
        cropped = image.crop(bbox)
        
        w, h = cropped.size
        new_w, new_h = (20, int(20 * (h / w))) if w > h else (int(20 * (w / h)), 20)
        new_w, new_h = max(1, new_w), max(1, new_h)
        resized = cropped.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        resized_arr = np.array(resized)
        total_mass = np.sum(resized_arr)
        if total_mass > 0:
            y_coords, x_coords = np.indices(resized_arr.shape)
            cy = np.sum(y_coords * resized_arr) / total_mass
            cx = np.sum(x_coords * resized_arr) / total_mass
        else:
            cy, cx = new_h / 2.0, new_w / 2.0
            
        paste_x, paste_y = int(round(14.0 - cx)), int(round(14.0 - cy))
        new_img = Image.new('L', (28, 28), 0)
        new_img.paste(resized, (paste_x, paste_y))
        image = new_img
    else:
        image = image.resize((28, 28), Image.Resampling.LANCZOS)
    
    img_array = np.array(image, dtype=np.float32) / 255.0
    return img_array.reshape(1, 784)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    load_models()
    
    if ensemble is None or interpreter is None:
        return jsonify({'error': 'Ereduak ez daude prest. Lehenik prepare_models.py exekutatu.'}), 500
        
    data = request.json
    if not data or 'image' not in data:
        return jsonify({'error': 'Ez da irudirik jaso'}), 400
    
    try:
        img_flat = parse_image(data['image'])
        
        if DEBUG_MODE:
            print(f"\\n--- DEBUG: Inference Request {timestamp} ---")
            print(f"Max Pixel Value: {np.max(img_array):.3f}")
            print(f"Min Pixel Value: {np.min(img_array):.3f}")
            print(f"Total Mass: {np.sum(img_array):.3f}")
        
        # Iragarpena eta Entropia kalkulatu Ensemble bidez
        mean_probs, entropy = ensemble.predict_proba_with_uncertainty(img_flat)
        predicted_class = int(np.argmax(mean_probs[0]))
        confidence = float(np.max(mean_probs[0]))
        ent_val = float(entropy[0])
        
        if DEBUG_MODE:
            print(f"Mean Probs: \\n{np.round(mean_probs[0], 3)}")
            print(f"Entropy: {ent_val:.4f}")
            print(f"Predicted Class: {predicted_class} (Conf: {confidence:.3f})")
        
        is_unknown = bool(interpreter.predict(entropy)[0] == 1)
        threshold_val = float(getattr(interpreter, 'threshold', 1.25))
        probs_list = mean_probs[0].tolist()
        class_labels = [index_to_label.get(i, str(i)) for i in range(len(probs_list))]
        
        if DEBUG_MODE:
            print(f"Is Unknown (OOD): {is_unknown}")
            print("--------------------------------------\n")
            
        # Gordeko dugu inferentziaren historia
        if ACTIVE_MODEL_ID not in inference_history:
            inference_history[ACTIVE_MODEL_ID] = []
            
        inference_record = {
            'timestamp': datetime.datetime.now().isoformat(),
            'prediction': predicted_class,
            'prediction_label': index_to_label.get(predicted_class, str(predicted_class)),
            'confidence': confidence,
            'entropy': ent_val,
            'is_unknown': is_unknown,
            'threshold': threshold_val,
            'probabilities': probs_list,
            'labels': class_labels
        }
        inference_history[ACTIVE_MODEL_ID].append(inference_record)
        # Keep only last 50 queries to prevent memory bloat
        inference_history[ACTIVE_MODEL_ID] = inference_history[ACTIVE_MODEL_ID][-50:]
        save_metadata()
        
        return jsonify(inference_record)
        
    except Exception as e:
        if DEBUG_MODE:
            print(f"Error handling request: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/train_new_class', methods=['POST'])
def train_new_class():
    global ensemble, interpreter, subset_x, subset_y, ACTIVE_MODEL_ID
    load_models()
    
    data = request.json
    label = data.get('label', '').strip()
    images_b64 = data.get('images', [])
    
    if not label or not images_b64:
        return jsonify({'error': 'Label edo irudi faltan'}), 400
        
    try:
        if label not in label_to_index:
            new_idx = len(label_to_index)
            label_to_index[label] = new_idx
            index_to_label[new_idx] = label
        
        idx = label_to_index[label]
        
        new_x_list = []
        for b64 in images_b64:
            new_x_list.append(parse_image(b64))
        X_new = np.vstack(new_x_list)
        y_new = np.array([idx] * len(images_b64))
        
        # Append to global subset
        base_x, base_y = get_subset()
        subset_x = np.vstack([base_x, X_new])
        subset_y = np.concatenate([base_y, y_new])
        
        # Retrain newly sized models
        from src.classifiers import RandomForestClassifierModel, NeuralNetworkMCDropoutClassifier
        from src.ensemble import EnsembleClassifier
        
        num_classes = len(label_to_index)
        rfc = RandomForestClassifierModel(n_estimators=50, max_depth=15, random_state=42, num_classes=num_classes)
        nn = NeuralNetworkMCDropoutClassifier(hidden_dim=128, num_classes=num_classes, dropout_rate=0.3, epochs=15, n_iter=20)
        
        new_ensemble = EnsembleClassifier([rfc, nn])
        
        if DEBUG_MODE: print(f"Retraining ensemble with new size: {num_classes} classes, {len(subset_x)} samples")
        new_ensemble.fit(subset_x, subset_y)
        
        # Evaluate stability on JUST the newly provided images
        mean_probs, entropy = new_ensemble.predict_proba_with_uncertainty(X_new)
        mean_entropy = float(np.mean(entropy))
        
        if DEBUG_MODE: print(f"New class entropy: {mean_entropy:.4f}")
        
        total_given = len(np.where(subset_y == idx)[0])
        
        # Stability threshold maximoa 50 marrazkialditan utziko dugu
        if mean_entropy > 0.4 and total_given < 50:
            return jsonify({
                "status": "needs_more_data",
                "entropy": mean_entropy,
                "current_total": total_given,
                "required_total": total_given + 5
            })
            
        # Success, keep the new ensemble in the Tree
        new_version_id = f"v{uuid.uuid4().hex[:6]}"
        joblib.dump(new_ensemble, os.path.join(MODELS_DIR, f"ensemble_{new_version_id}.pkl"))
        
        # Update Tree metadata
        parent_capabilities = tree_metadata[ACTIVE_MODEL_ID].get('capabilities', [])
        new_capabilities = parent_capabilities.copy()
        if label not in new_capabilities:
            new_capabilities.append(label)
            
        tree_metadata[new_version_id] = {
            'id': new_version_id,
            'parent_id': ACTIVE_MODEL_ID,
            'name': f"{tree_metadata[ACTIVE_MODEL_ID]['name']} + {label}",
            'capabilities': new_capabilities,
            'metrics': {'entropy': mean_entropy, 'samples_added': total_given},
            'timestamp': datetime.datetime.now().isoformat()
        }
        
        # Change active to new
        ACTIVE_MODEL_ID = new_version_id
        save_metadata()
        
        ensemble = new_ensemble
        return jsonify({
            "status": "success",
            "metrics": {
                "entropy": mean_entropy,
                "total_samples": total_given,
                "new_version_id": new_version_id
            }
        })
        
    except Exception as e:
        if DEBUG_MODE: print(f"Error in train_new_class: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/get_tree', methods=['GET'])
def get_tree():
    load_metadata()
    return jsonify({
        'tree': list(tree_metadata.values()),
        'active_id': ACTIVE_MODEL_ID,
        'history': inference_history.get(ACTIVE_MODEL_ID, [])
    })

@app.route('/set_active', methods=['POST'])
def set_active():
    global ACTIVE_MODEL_ID
    data = request.json
    model_id = data.get('model_id')
    if model_id in tree_metadata:
        ACTIVE_MODEL_ID = model_id
        save_metadata()
        load_models() # reload globally
        return jsonify({'status': 'success', 'active_id': ACTIVE_MODEL_ID})
    return jsonify({'error': 'Model not found'}), 404

@app.route('/merge_model', methods=['POST'])
def merge_model():
    global ACTIVE_MODEL_ID
    data = request.json
    model_id = data.get('model_id')
    
    if model_id not in tree_metadata:
        return jsonify({'error': 'Model not found'}), 404
        
    node = tree_metadata[model_id]
    parent_id = node.get('parent_id')
    
    if parent_id is None:
        return jsonify({'error': 'Cannot merge root node'}), 400
        
    # Check if this node is the ONLY child of its parent
    siblings = [k for k, v in tree_metadata.items() if v.get('parent_id') == parent_id and k != model_id]
    if len(siblings) > 0:
        return jsonify({'error': 'Cannot merge. Parent has branch conflicts (multiple children).'}), 400
        
    # Valid merge scenario: Delete parent
    parent_file = 'ensemble.pkl' if parent_id == 'v1' else f'ensemble_{parent_id}.pkl'
    parent_path = os.path.join(MODELS_DIR, parent_file)
    if os.path.exists(parent_path):
        os.remove(parent_path)
        
    # Point the lineage of the merged node to the grand-parent
    grandparent_id = tree_metadata[parent_id].get('parent_id')
    node['parent_id'] = grandparent_id
    
    # Remove old parent from metadata
    del tree_metadata[parent_id]
    
    # Clean up history reference
    if parent_id in inference_history:
        del inference_history[parent_id]
        
    save_metadata()
    return jsonify({'status': 'merged_successfully', 'new_tree': list(tree_metadata.values())})

@app.route('/reset_tree', methods=['POST'])
def reset_tree():
    global ACTIVE_MODEL_ID, tree_metadata, inference_history, ensemble
    load_metadata()
    
    # Root node bilatu
    root_id = None
    for k, v in tree_metadata.items():
        if v.get('parent_id') is None:
            root_id = k
            break
            
    if root_id is None:
        root_id = 'v1'
        
    # Ezabatu entrenatutako bertsioen fitxategiak (.pkl)
    for model_id in list(tree_metadata.keys()):
        if model_id != root_id:
            filename = f"ensemble_{model_id}.pkl"
            filepath = os.path.join(MODELS_DIR, filename)
            if os.path.exists(filepath):
                try:
                    os.remove(filepath)
                except Exception as e:
                    print(f"Errorea {filename} ezabatzean: {e}")
                    
    # Arbola oinarrizko ereduera berrezarri
    root_node = tree_metadata.get(root_id, {
        'id': root_id,
        'parent_id': None,
        'name': 'Oinarrizko MNIST Ensemble',
        'capabilities': [],
        'metrics': {'entropy': 0.1},
        'timestamp': datetime.datetime.now().isoformat()
    })
    
    tree_metadata = {root_id: root_node}
    ACTIVE_MODEL_ID = root_id
    inference_history = {root_id: []}
    
    save_metadata()
    load_models()
    return jsonify({'status': 'reset_successful', 'active_id': ACTIVE_MODEL_ID})

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Ensemble MNIST Web Server")
    parser.add_argument('--debug', action='store_true', help="Enable debug logs and save intermediate images")
    args = parser.parse_args()
    
    if args.debug:
        DEBUG_MODE = True
        print("[DEBUG MODE ENABLED] Intermediate images will be saved and verbose logs printed.")

    app.run(host='0.0.0.0', port=5000, debug=True)
