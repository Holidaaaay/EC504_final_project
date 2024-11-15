import cv2
import numpy as np
from sklearn.cluster import KMeans
from PIL import Image

def load_and_preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (100, 100))
    return image

def cluster_pixels(image, n_clusters=2):
    pixels = image.reshape(-1, 3)
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(pixels)
    labels = kmeans.labels_.reshape(image.shape[:2])
    preprocessed_image = np.zeros_like(image)
    preprocessed_image[labels == 0] = [0, 255, 0]  # Foreground in green
    preprocessed_image[labels == 1] = [255, 0, 0]  # Background in blue

    return Image.fromarray(preprocessed_image)

def process_image(image_path):
    image = load_and_preprocess_image(image_path)
    processed_image = cluster_pixels(image)
    return processed_image
