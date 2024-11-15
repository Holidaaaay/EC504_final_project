import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from segmentation import process_image, load_and_preprocess_image

def open_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        original_image = load_and_preprocess_image(file_path)
        processed_image = process_image(file_path)
        display_images(original_image, processed_image)

def display_images(original, processed):
    original_img_tk = ImageTk.PhotoImage(Image.fromarray(original))
    processed_img_tk = ImageTk.PhotoImage(processed)

    original_image_label.config(image=original_img_tk)
    original_image_label.image = original_img_tk

    processed_image_label.config(image=processed_img_tk)
    processed_image_label.image = processed_img_tk

root = tk.Tk()
root.title("Image Preprocessing Prototype")

open_button = tk.Button(root, text="Open Image", command=open_image)
open_button.pack()

original_image_label = tk.Label(root, text="Original Image")
original_image_label.pack(side="left", padx=10, pady=10)

processed_image_label = tk.Label(root, text="Processed Image")
processed_image_label.pack(side="right", padx=10, pady=10)

root.mainloop()
