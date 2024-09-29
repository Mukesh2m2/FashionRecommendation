import pickle
import numpy as np
from numpy.linalg import norm
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.neighbors import NearestNeighbors
import cv2


class ImageRetriever:
    def __init__(
        self,
        model_weights="imagenet",
        embeddings_file="embeddings.pkl",
        filenames_file="filenames.pkl",
    ):
        # Load the ResNet50 model
        self.model = ResNet50(
            weights=model_weights, include_top=False, input_shape=(224, 224, 3)
        )
        self.model.trainable = False
        self.model = tf.keras.Sequential([self.model, GlobalMaxPooling2D()])

        # Load embeddings and filenames
        self.feature_list = np.array(pickle.load(open(embeddings_file, "rb")))
        self.filenames = pickle.load(open(filenames_file, "rb"))

        # Initialize Nearest Neighbors
        self.neighbors = NearestNeighbors(
            n_neighbors=10, algorithm="brute", metric="euclidean"
        )
        self.neighbors.fit(self.feature_list)


    def extract_features(self, img_path):
        """Extracts features of an input image using the pretrained ResNet50 model."""
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        expanded_img_array = np.expand_dims(img_array, axis=0)
        preprocessed_img = preprocess_input(expanded_img_array)

        result = self.model.predict(preprocessed_img).flatten()
        norm_result = result / norm(result)  # Normalize the result

        return norm_result


    def find_similar_images(self, img_path):
        """Finds and displays the most similar images."""
        norm_result = self.extract_features(img_path)

        # Find nearest neighbors
        distances, indices = self.neighbors.kneighbors([norm_result])

        # Display the results
        for idx in indices[0]:
            temp_img = cv2.imread(self.filenames[idx])
            cv2.imshow("output", cv2.resize(temp_img, (512, 512)))
            cv2.waitKey(0)



if __name__ == "__main__":
    retriever = ImageRetriever()
    sample_image_path = "sample_images/img_3.jpg"
    retriever.find_similar_images(sample_image_path)
