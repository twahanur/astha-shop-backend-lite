from flask import Flask, request, jsonify
import pickle
import numpy as np
from numpy.linalg import norm
import urllib.request
import os
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from flask_cors import CORS

# Import tflite runtime for a smaller memory footprint
import tensorflow as tf

# Lazy-load Keras modules only when needed
image_processing = None
preprocess_input_func = None

def load_image_dependencies():
    global image_processing, preprocess_input_func
    if image_processing is None:
        from tensorflow.keras.preprocessing import image
        from tensorflow.keras.applications.resnet50 import preprocess_input
        image_processing = image
        preprocess_input_func = preprocess_input

# --- Global Initialization (Optimized for low memory and fast startup) ---

# Ensure 'uploads' directory exists
UPLOAD_FOLDER = "./uploads/"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# --- Optimized Model Loading ---

# 1. Load TensorFlow Lite Model (Much smaller and faster than full Keras)
print("Loading TFLite feature extractor model...")
try:
    interpreter = tf.lite.Interpreter(model_path="feature_extractor.tflite")
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print("TFLite model loaded.")
except Exception as e:
    print(f"Error: Failed to load TFLite model 'feature_extractor.tflite'. {e}")
    interpreter = None

# 2. Load pre-computed features and filenames
print("Loading embeddings and filenames...")
try:
    with open('embeddings.pkl', 'rb') as f:
        feature_list = pickle.load(f)
    with open('filenames.pkl', 'rb') as f:
        filenames = pickle.load(f)
    print("Embeddings and filenames loaded.")
except Exception as e:
    print(f"Error loading pickle files: {e}")
    feature_list, filenames = [], []

# 3. Initialize Nearest Neighbors
if feature_list:
    neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)
    print("NearestNeighbors model fitted.")
else:
    print("Warning: No feature list, NearestNeighbors not fitted.")
    neighbors = None

# 4. Load CSV for search function
print("Loading grandfinaleX.csv...")
try:
    df_products = pd.read_csv('./grandfinaleX.csv', on_bad_lines='skip')
    SEARCH_COLUMNS = ['gender', 'masterCategory', 'subCategory', 'articleType', 'productDisplayName']
    for col in SEARCH_COLUMNS:
        if col in df_products.columns:
            df_products[col] = df_products[col].astype(str).str.lower()
        else:
            SEARCH_COLUMNS.remove(col)
    print("CSV loaded and pre-processed for search.")
except Exception as e:
    print(f"Error loading grandfinaleX.csv: {e}")
    df_products = pd.DataFrame()

# --- Flask App Configuration ---
app = Flask(__name__)
CORS(app)

# --- Utility Functions (Updated for TFLite) ---
def extract_features_tflite(img_path, interpreter):
    # Load image processing libraries on demand
    load_image_dependencies()

    img = image_processing.load_img(img_path, target_size=(224, 224))
    img_array = image_processing.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input_func(expanded_img_array)

    # Set the tensor to the TFLite model's input
    interpreter.set_tensor(input_details[0]['index'], preprocessed_img)
    interpreter.invoke() # Run inference

    # Get the result from the TFLite model's output
    result = interpreter.get_tensor(output_details[0]['index']).flatten()
    normalized_result = result / norm(result)
    return normalized_result

# --- Routes ---

@app.route('/', methods=['GET'])
def helloworld():
    return jsonify({'response': "Hello World! Your API is running."})

@app.route('/recommend', methods=['GET'])
def recommend():
    img_url = request.args.get('name')
    img_id = request.args.get('id')

    if not img_url or not img_id:
        return jsonify({'error': 'Missing "name" (image URL) or "id" parameters'}), 400

    if not interpreter or not neighbors:
        return jsonify({'error': 'Image recommendation services are not initialized.'}), 503

    temp_img_path = os.path.join(UPLOAD_FOLDER, f"{img_id}.jpg")

    try:
        urllib.request.urlretrieve(img_url, temp_img_path)
        # Use the TFLite feature extractor
        result_features = extract_features_tflite(temp_img_path, interpreter)
        distances, indices = neighbors.kneighbors([result_features])
        # A more robust way to extract and convert IDs
        final_ids = [int(os.path.splitext(os.path.basename(filenames[i]))[0]) for i in indices[0]]
        return jsonify({'result': final_ids})
    except Exception as e:
        print(f"Error in /recommend: {e}")
        return jsonify({'error': 'An error occurred during recommendation.'}), 500
    finally:
        if os.path.exists(temp_img_path):
            os.remove(temp_img_path)

@app.route('/image_search', methods=['GET'])
def imagesearch():
    img_url = request.args.get('url')
    if not img_url:
        return jsonify({'error': 'Missing "url" parameter'}), 400

    if not interpreter or not neighbors:
        return jsonify({'error': 'Image search services are not initialized.'}), 503

    temp_img_path = os.path.join(UPLOAD_FOLDER, "test_search.jpg")
    try:
        urllib.request.urlretrieve(img_url, temp_img_path)
        result_features = extract_features_tflite(temp_img_path, interpreter)
        distances, indices = neighbors.kneighbors([result_features])
        final_ids = [int(os.path.splitext(os.path.basename(filenames[i]))[0]) for i in indices[0]]
        return jsonify({'result': final_ids})
    except Exception as e:
        print(f"Error in /image_search: {e}")
        return jsonify({'error': 'An error occurred during image search.'}), 500
    finally:
        if os.path.exists(temp_img_path):
            os.remove(temp_img_path)

@app.route('/search', methods=['GET'])
def search():
    query = request.args.get('query')
    if not query:
        return jsonify({'error': 'Missing "query" parameter'}), 400
    
    if df_products.empty:
        return jsonify({'error': 'Product data for search is not loaded.'}), 503

    # More efficient, vectorized search
    query_keywords = query.lower().split()
    search_pattern = '|'.join(query_keywords)
    
    mask = df_products[SEARCH_COLUMNS].apply(
        lambda col: col.str.contains(search_pattern, na=False)
    ).any(axis=1)

    result_df = df_products[mask]

    if 'id' in result_df.columns:
        top_ids = result_df['id'].dropna().astype(int).drop_duplicates().head(20).tolist()
    else:
        top_ids = []

    return jsonify({'searchResult': top_ids})

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))