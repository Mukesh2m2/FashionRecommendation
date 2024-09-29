import streamlit as st
import os
from PIL import Image
import pickle
import numpy as np
from numpy.linalg import norm
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.neighbors import NearestNeighbors


# Add custom CSS for decorations and zoom functionality
def add_custom_css():
    st.markdown(
        """
        <style>
        /* Add zoom on hover for images */
        .image-container img {
            transition: transform 0.2s; /* Animation */
        }
        .image-container img:hover {
            transform: scale(1.5); /* Zoom in */
        }

        /* Style upload box and background */
        .css-1q8dd3e {
            border-radius: 15px;
            background-color: #f0f0f5;
            border: 2px solid #6c757d;
        }

        /* Add some padding around content */
        .stApp {
            background-color: #e0f7fa;
            padding: 20px;
        }

        /* Add borders and shadow to image columns */
        .css-1cpxqw2 {
            box-shadow: 2px 2px 8px rgba(0, 0, 0, 0.2);
            border-radius: 10px;
            margin-bottom: 20px;
        }

        /* Title styling */
        h1 {
            font-family: 'Arial', sans-serif;
            color: #00796b;
            text-align: center;
            padding-bottom: 10px;
        }

        /* Uploaded image caption styling */
        .uploaded-image-caption {
            color: #004d40; 
            text-align: center;
            margin-top: 5px;
            font-weight: bold;
        }

        /* Recommendation text styling */
        .recommendation-caption {
            color: #004d40; 
            text-align: center;
            margin-top: 5px;
            font-weight: bold;
        }
        
        .recommendation-caption-1 {
            color: #004d40; 
            text-align: center;
            margin-top: 5px;
            font-weight: bold;
            text-width: 150%
        }

        </style>
    """,
        unsafe_allow_html=True,
    )


class FashionRecommender:
    def __init__(self):
        # Initialize model
        self.model = ResNet50(
            weights="imagenet", include_top=False, input_shape=(224, 224, 3)
        )
        self.model.trainable = False
        self.model = tf.keras.Sequential([self.model, GlobalMaxPooling2D()])

        # Load precomputed features and filenames
        self.feature_list = np.array(pickle.load(open("embeddings.pkl", "rb")))
        self.filenames = pickle.load(open("filenames.pkl", "rb"))


    def save_file(self, uploaded_file):
        try:
            # Create directory if it doesn't exist
            if not os.path.exists("uploads"):
                os.makedirs("uploads")

            file_path = os.path.join("uploads", uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            return file_path

        except Exception as e:
            st.error(f"An error occurred while saving the file: {e}")
            return None


    def feature_extraction(self, img_path):
        try:
            img = image.load_img(img_path, target_size=(224, 224))
            img_array = image.img_to_array(img)
            expanded_img_array = np.expand_dims(img_array, axis=0)
            preprocessed_img = preprocess_input(expanded_img_array)
            result = self.model.predict(preprocessed_img).flatten()
            norm_result = result / norm(result)
            return norm_result

        except Exception as e:
            st.error(f"An error occurred while extracting features: {e}")
            return None


    def recommend(self, features):
        try:
            neighbors = NearestNeighbors(
                n_neighbors=10, algorithm="brute", metric="euclidean"
            )
            neighbors.fit(self.feature_list)
            distances, indices = neighbors.kneighbors([features])
            return indices

        except Exception as e:
            st.error(f"An error occurred during recommendation: {e}")
            return None


    def display_recommendations(self, indices):
        st.write( f"<div class='recommendation-caption-1'>Recommended Items </div>", unsafe_allow_html=True)

        cols = st.columns(5)
        for i in range(5):
            with cols[i]:
                img_path = self.filenames[indices[0][i]]
                # Display image with caption
                st.image(img_path, use_column_width=True)
                st.markdown(
                    f"<div class='recommendation-caption'>Recommendation {i+1}</div>",
                    unsafe_allow_html=True,
                )

        cols = st.columns(5)
        for i in range(5, 10):
            with cols[i - 5]:
                img_path = self.filenames[indices[0][i]]
                # Display image with caption
                st.image(img_path, use_column_width=True)
                st.markdown(
                    f"<div class='recommendation-caption'>Recommendation {i+6}</div>",
                    unsafe_allow_html=True,
                )


# Streamlit App
st.title("Fashion Recommendations")

# Add custom CSS
add_custom_css()

# Initialize Fashion Recommender
recommender = FashionRecommender()

# File upload and processing
uploaded_file = st.file_uploader("Upload an image file")

if uploaded_file is not None:
    file_path = recommender.save_file(uploaded_file)
    if file_path:
        # Display the uploaded image
        display_image = Image.open(uploaded_file)
        st.image(
            display_image, caption="", use_column_width=True
        )  
        st.markdown(
            f"<div class='uploaded-image-caption'>Uploaded Image: {uploaded_file.name}</div>",
            unsafe_allow_html=True,
        )

        # Extract features
        features = recommender.feature_extraction(file_path)
        if features is not None:
            # Recommend similar items
            indices = recommender.recommend(features)
            if indices is not None:
                # Display recommendations
                recommender.display_recommendations(indices)
                
        else:
            st.error("Feature extraction failed.")
            
    else:
        st.error("File could not be saved. Please upload again.")
