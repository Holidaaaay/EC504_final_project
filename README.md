# Project Running Instructions

## Project Overview

This project is an interactive image segmentation application. 

Users can upload images and click on specific areas to remove segments of the image, providing a basic foreground/background segmentation feature.

The project's repository is hosted on GitHub: [EC504 Final Project](https://github.com/Holidaaaay/EC504_final_project).

------

## Requirements

Ensure your system meets the following requirements:

1. **Python** version >= 3.8
2. Required Python libraries:
   - Flask
   - OpenCV (cv2)
   - NumPy
   - Matplotlib
   - scikit-learn
   - PyMaxflow(for graph construction)

------

## Steps to Run the Project

### 1. Clone the Project

Clone the repository from GitHub:

```
git clone https://github.com/Holidaaaay/EC504_final_project.git
cd EC504_final_project
```

### 2. Install Dependencies

Navigate to the project directory and install the required dependencies:

```
pip install -r requirements.txt
```

### 3. Start the Flask Server

Run the following command to start the Flask server:

```
python app.py
```

Once the server starts successfully, you’ll see a message similar to:

```
 * Running on http://127.0.0.1:5000/ (Press CTRL+C to quit)
```

### 4. Access the Web Application

Open your browser and navigate to http://127.0.0.1:5000/. You will see a file upload interface.

------

## How to Use

### Upload an Image

1. Click the `Choose File` button to select a PNG or JPG image.
2. Click the `Upload` button. Once the upload is complete, the page will display the uploaded image.

### Remove a Segment

1. Click on the uploaded image to select the region you want to segment. The application will automatically record the coordinates of your click.
2. Click the `Remove Segment` button. The system will compute the foreground/background segmentation and display the processed image.

------

## Directory Descriptions

- **uploads/**: Contains the images uploaded by users.
- **results/**: Contains the processed images after segmentation.
- **segment.html**: The frontend HTML file for user interaction.

------

## Limitations and Improvements

- The segmentation logic is based on KMeans clustering and B-K graph cuts, which may produce suboptimal results for complex images.
- When the foreground and background have similar colors or textures, the algorithm struggles to achieve the desired segmentation.

