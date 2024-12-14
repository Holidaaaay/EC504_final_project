# EC504_final_project
Project Running Instructions
Project Overview
This project is an interactive image segmentation application built with Flask. Users can upload images and click on specific areas to remove segments of the image. The segmentation logic is based on KMeans clustering and graph cuts, providing a basic foreground/background segmentation feature.

Requirements
Ensure your system meets the following requirements:

Python version >= 3.8
Required Python libraries:
Flask
OpenCV (cv2)
NumPy
Matplotlib
scikit-learn
maxflow (for graph cuts)
Steps to Run the Project
1. Clone or Download the Project
Download the project files and ensure the structure is as follows:

bash
Copy code
project/
├── app.py                # Flask main application
├── segmentation.py       # Image segmentation logic
├── segment.html          # Frontend HTML file
├── uploads/              # Directory for uploaded images
├── results/              # Directory for processed images
└── requirements.txt      # Dependency list
2. Install Dependencies
Navigate to the project directory and install the required dependencies:

bash
Copy code
pip install -r requirements.txt
3. Start the Flask Server
Run the following command to start the Flask server:

bash
Copy code
python app.py
Once the server starts successfully, you’ll see a message similar to:

csharp
Copy code
 * Running on http://127.0.0.1:5000/ (Press CTRL+C to quit)
4. Access the Web Application
Open your browser and navigate to http://127.0.0.1:5000/. You will see a file upload interface.

How to Use
Upload an Image
Click the Choose File button to select a PNG or JPG image.
Click the Upload button. Once the upload is complete, the page will display the uploaded image.
Remove a Segment
Click on the uploaded image to select the region you want to segment. The application will automatically record the coordinates of your click.
Click the Remove Segment button. The system will compute the foreground/background segmentation and display the processed image.
Directory Descriptions
uploads/: Contains the images uploaded by users.
results/: Contains the processed images after segmentation.
segment.html: The frontend HTML file for user interaction.
Limitations and Improvements
Algorithm Performance:

The segmentation logic is based on KMeans clustering and graph cuts, which may produce suboptimal results for complex images.
When the foreground and background have similar colors or textures, the algorithm struggles to achieve the desired segmentation.
Static Output:

Currently, the segmentation endpoint (/segment/<filename>) returns a predefined processed image (demo_processed_image.png) as a placeholder. Integration with the actual segmentation logic from segmentation.py is needed.
Enhancements for Better Results:

The segmentation results could be significantly improved by integrating advanced models such as U-Net or other deep learning-based segmentation methods.
Interactivity:

While the application captures user clicks, these inputs are not yet dynamically incorporated into the segmentation process.
