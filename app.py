import os

import cv2
from flask import Flask, request, render_template, send_from_directory
from segmentation import segment_image

app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
RESULT_FOLDER = os.path.join(BASE_DIR, 'results')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file and file.filename != '':
            filename = file.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            return render_template('segment.html', filename=filename, message="Click on the image to select a region. To remove the selected region, click the 'Remove' button.")
    return render_template('segment.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/results/<filename>')
def processed_file(filename):
    return send_from_directory(app.config['RESULT_FOLDER'], filename)

@app.route('/click/<filename>', methods=['POST'])
def click(filename):
    x = int(request.form.get('x'))
    y = int(request.form.get('y'))
    message = f"You clicked at coordinates: ({x}, {y}). To remove the selected region, click the 'Remove' button."

    # Save the clicked coordinates as a background seed
    bg_seed_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{filename}_bg_seed.txt")
    with open(bg_seed_path, 'w') as f:
        f.write(f"{x},{y}")

    return render_template('segment.html', filename=filename, message=message)

@app.route('/segment/<filename>', methods=['POST'])
def segment(filename):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    result_filepath = os.path.join(app.config['RESULT_FOLDER'], 'segmented_' + filename)

    # Load the background seed from file
    bg_seed_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{filename}_bg_seed.txt")
    if os.path.exists(bg_seed_path):
        with open(bg_seed_path, 'r') as f:
            bg_seed = tuple(map(int, f.read().strip().split(',')))
    else:
        bg_seed = (0, 0)  # Default background seed if none provided

    # Call segment_image with the background seed
    segmented_image = segment_image(filepath, bg_seed)
    cv2.imwrite(result_filepath, segmented_image)

    return render_template('segment.html', filename=filename, result_filename='segmented_' + filename, message="Segmentation complete. View the result below.")

if __name__ == '__main__':
    app.run(debug=True)
