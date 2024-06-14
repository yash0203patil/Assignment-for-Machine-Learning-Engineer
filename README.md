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

- **app.py**: Flask application code for handling routes and image classification.
- **model.pkl**: Pre-trained RandomForestClassifier model for image classification.
- **uploads/**: Directory to store temporarily uploaded images.
- **templates/**: HTML templates for rendering web pages (`upload.html` for file upload form).

## Installation

1. **Clone the Repository**:
   ```bash
   git clone <repository-url>
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
