# Image Classification Web Application using Flask

## Overview

This project implements a web application for image classification using Flask. It allows users to upload images and get predictions from a pre-trained machine learning model.

## Features

- **Upload Functionality**: Users can upload images for classification.
- **Machine Learning Model**: Uses a RandomForestClassifier model trained on image features.
- **Responsive UI**: Simple web interface for uploading images and displaying results.
- **Scalable**: Easily extendable for different models or additional features.

## Project Structure

The repository structure is as follows:
Move your project directory to this directory - flask_image_classifier
- **app.py**: Flask application code for handling routes and image classification.
- **random_forest_model_fs3.pkl**: Pre-trained RandomForestClassifier model for image classification.
- **uploads/**: Directory to store temporarily uploaded images.
- **templates/**: HTML templates for rendering web pages (`upload.html` for file upload form).

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yash0203patil/Assignment-for-Machine-Learning-Engineer.git
   cd <repository-directory>
   ```

2. **Install Dependencies:**:
    ```bash
   pip install -r requirements.txt
   ```

3. **Run the Application**:
    ```bash
   python app.py
   ```
