import os
import cv2
import numpy as np
import joblib
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from werkzeug.utils import secure_filename
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
model_filename_fs3 = 'random_forest_model_fs3.pkl'
model = joblib.load(model_filename_fs3)


def preprocess_image_and_pca(image_path, n_components=100):
    image = cv2.imread(image_path)
    if image is None:
        return None, None
    
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    edges = cv2.Canny(gray, 100, 200)
    gray_flat = gray.flatten()
    edge_flat = edges.flatten()
    combined_features = np.concatenate((hist.flatten(), gray_flat, edge_flat))
    
    
    combined_features = combined_features.reshape(1, -1)  
    
    
    n_samples, n_features = combined_features.shape
    if n_samples < n_components or n_features < n_components:
        return None, None  
  
    pca = PCA(n_components=n_components)
    pca.fit(combined_features)
    reduced_features = pca.transform(combined_features)
    
    return reduced_features.flatten(), pca


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file and allowed_file(file.filename):
     
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
       
        reduced_image, pca = preprocess_image_and_pca(file_path)
        
        if reduced_image is None:
            return jsonify({'error': 'Invalid or insufficient features in processed image'})
        
    
        prediction = model.predict([reduced_image])[0]
        class_name = str(prediction)  
        os.remove(file_path)

        return jsonify({'class': class_name})

    else:
        return jsonify({'error': 'File type not allowed'})

if __name__ == '__main__':
    app.run(debug=True)
