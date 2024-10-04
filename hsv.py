import tkinter as tk
from PIL import Image, ImageTk
import cv2
import numpy as np

# Create the main window
window = tk.Tk()
window.geometry("500x600")
window.title("HSV Threshold")

# Load the image
image = cv2.imread("4.png")

# Function to update the HSV threshold values
def update_hsv_threshold(*args):
    hue_min = hue_min_scale.get()
    hue_max = hue_max_scale.get()
    sat_min = sat_min_scale.get()
    sat_max = sat_max_scale.get()
    val_min = val_min_scale.get()
    val_max = val_max_scale.get()

    # Convert the image to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define the lower and upper threshold values for HSV
    lower_threshold = np.array([hue_min, sat_min, val_min])
    upper_threshold = np.array([hue_max, sat_max, val_max])

    # Create a mask using the threshold values
    mask = cv2.inRange(hsv_image, lower_threshold, upper_threshold)

    # Apply the mask to the original image
    masked_image = cv2.bitwise_and(image, image, mask=mask)

    # Update the displayed image
    photo = ImageTk.PhotoImage(Image.fromarray(masked_image))
    image_label.configure(image=photo)
    image_label.image = photo

# Create a label to display the image
image_label = tk.Label(window)
image_label.pack()

# Create scale widgets to adjust the HSV threshold values
hue_min_scale = tk.Scale(window, from_=0, to=179, orient=tk.HORIZONTAL, label="Hue Min", command=update_hsv_threshold)
hue_min_scale.pack(side="left")

hue_max_scale = tk.Scale(window, from_=0, to=179, orient=tk.HORIZONTAL, label="Hue Max", command=update_hsv_threshold)
hue_max_scale.pack(side="left")

sat_min_scale = tk.Scale(window, from_=0, to=255, orient=tk.HORIZONTAL, label="Saturation Min", command=update_hsv_threshold)
sat_min_scale.pack(side="left")

sat_max_scale = tk.Scale(window, from_=0, to=255, orient=tk.HORIZONTAL, label="Saturation Max", command=update_hsv_threshold)
sat_max_scale.pack(side="left")

val_min_scale = tk.Scale(window, from_=0, to=255, orient=tk.HORIZONTAL, label="Value Min", command=update_hsv_threshold)
val_min_scale.pack(side="left")

val_max_scale = tk.Scale(window, from_=0, to=255, orient=tk.HORIZONTAL, label="Value Max", command=update_hsv_threshold)
val_max_scale.pack(side="left")

# Start the main event loop
window.mainloop()
