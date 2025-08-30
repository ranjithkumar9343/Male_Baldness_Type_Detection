!nvidia-smi

from google.colab import drive
drive.mount('/content/drive')

!pip install ultralytics

%cd /content/drive/MyDrive/dataset

from ultralytics import YOLO

# Load a YOLO model for classification (e.g., YOLOv8s)
model = YOLO('yolov8n-cls.pt')  # You can also use 'yolov8m-cls.pt', 'yolov8l-cls.pt', etc.

# Train the model
model.train(data='/content/drive/MyDrive/dataset', epochs=20, batch=16, imgsz=224)

results = model.val(data='/content/drive/MyDrive/dataset')
print(results)

# Re-import necessary libraries
import os
import matplotlib.pyplot as plt
from ultralytics import YOLO
from PIL import Image

# Ensure the YOLO model is loaded
model = YOLO("/content/drive/MyDrive/dataset/runs/classify/train/weights/best.pt")

# Function to predict and visualize results
def predict_and_visualize(model, image_path):
    # Perform prediction
    results = model.predict(source=image_path)

    # Display the image with predictions
    results_img = results[0].plot(show=False)
    plt.figure(figsize=(10, 10))
    plt.imshow(results_img)
    plt.axis('off')
    plt.show()

# Path to your image
image_path = "/content/drive/MyDrive/predict1.jpg"

# Predict and visualize
predict_and_visualize(model, image_path)




# Re-import necessary libraries
import os
import matplotlib.pyplot as plt
from ultralytics import YOLO
from PIL import Image

# Ensure the YOLO model is loaded
model = YOLO("/content/drive/MyDrive/best.pt")

# Function to predict and visualize results
def predict_and_visualize(model, image_path):
    # Perform prediction
    results = model.predict(source=image_path)

    # Display the image with predictions
    results_img = results[0].plot(show=False)
    plt.figure(figsize=(10, 10))
    plt.imshow(results_img)
    plt.axis('off')
    plt.show()

# Path to your image
image_path = "/content/drive/MyDrive/prajwaltop.jpg"

# Predict and visualize
predict_and_visualize(model, image_path)



import os
import matplotlib.pyplot as plt
from ultralytics import YOLO
from PIL import Image

# Ensure the YOLO model is loaded
model = YOLO("/content/drive/MyDrive/best.pt")

# Function to predict and visualize results
def predict_and_visualize(model, image_path):
    # Perform prediction
    results = model.predict(source=image_path)

    # Display the image with predictions
    results_img = results[0].plot(show=False)
    plt.figure(figsize=(10, 10))
    plt.imshow(results_img)
    plt.axis('off')
    plt.show()

# Path to your image
image_path = "/content/drive/MyDrive/17.jpg"

# Predict and visualize
predict_and_visualize(model, image_path)


