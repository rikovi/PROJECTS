# import os
# import cv2
# import shutil
# import math
# import random
# import pickle
# import tensorflow as tf
# import argparse
# import keras
# import numpy as np

# import warnings
# warnings.filterwarnings("ignore")
# from PIL import Image
# # from scipy.misc import imresize
# from keras.models import model_from_json

# # import our own modules
# # import data
# import face
# import cnn_model

# model=cnn_model.initialize_model()

# # process images and split them into training, validation and test sets
# # total_count = data.process_data()
# # data.split_data(total_count)

# def imresize(image, size):
#     pil_image = Image.fromarray(image)
#     return np.array(pil_image.resize((size[1], size[0]), Image.LANCZOS))

# # create the training and validation sets for race recognition
# print("Begin creating training and validation sets")
# file_path = "colour_files/model_sets.txt"
# if os.path.exists(file_path): # if file exists, simply load file containing the sets 
# 	print("sets file already exists")
# 	# open file and load sets dictionary
# 	new_f = open(file_path, "rb")
# 	sets = pickle.load(new_f) 
# 	new_f.close()

# 	# initialize set variables
# 	train_x = sets["train_x"]
# 	train_y = sets["train_y"]
# 	valid_x = sets["valid_x"]
# 	valid_y = sets["valid_y"]
# # else: # create the sets and store them in a pickle file for faster access 
# # 	# initialize training set and validaiton set and also normalize their values 
# # 	print("Sets file doesn't exist, now creating them")
# # 	(train_x, train_y) = cnn_model.build_data("training", "all")
# # 	(valid_x, valid_y) = cnn_model.build_data("validation", "all")
# # 	train_x = train_x.reshape(train_x.shape[0], 50, 50, 3).astype('float32') / 255.0
# # 	valid_x = valid_x.reshape(valid_x.shape[0], 50, 50, 3).astype('float32') / 255.0

# # 	# store the sets in a pickle file for faster access 
# # 	model_sets = {"train_x": train_x, "train_y": train_y, "valid_x": valid_x,
# # 	           "valid_y": valid_y}
# # 	new_f = open("{}".format(file_path), "wb") 
# # 	pickle.dump(model_sets, new_f) 
# # 	new_f.close()

# from tensorflow.keras.layers import Layer
# import tensorflow.keras.backend as K
# from tensorflow.keras.models import model_from_json

# class AttentionLayer(Layer):
#     def __init__(self, **kwargs):
#         super(AttentionLayer, self).__init__(**kwargs)

#     def build(self, input_shape):
#         # Define trainable weights for the attention mechanism here
#         super(AttentionLayer, self).build(input_shape)

#     def call(self, inputs):
#         # Implement the attention logic
#         return K.sum(inputs, axis=1)

#     def compute_output_shape(self, input_shape):
#         # Define the shape of the output
#         return input_shape[0], input_shape[2]
    

# # Load the JSON string (already created and saved)
# with open("colour_files/model.json", "r") as json_file:
#     loaded_model_json = json_file.read()

# # Load model architecture with custom layer
# # model = model_from_json(loaded_model_json, custom_objects={'AttentionLayer': AttentionLayer})


# # check if model exists
# model_path = "colour_files/model.json"
# if os.path.exists(model_path): 
# 	# if file exists, simply load file containing the model
# 	print("Race Recognition model already created")
# 	# load json and create model
# 	# json_file = open(model_path, 'r')
# 	# loaded_model_json = json_file.read()
# 	# json_file.close()
# 	with open(model_path,'r') as f:
# 		loaded_model_json = f.read()
# 	# model = model_from_json(loaded_model_json)
# 	model = model_from_json(loaded_model_json, custom_objects={'AttentionLayer': AttentionLayer})
# 	# load weights into new model
# 	model.load_weights("color_files/model.h5")
# 	print("Loaded race model from disk")

# 	# evaluate loaded model on test data
# 	model.compile(optimizer=tf.train.AdamOptimizer(learning_rate=0.001), 
# 	                    loss='sparse_categorical_crossentropy',
# 	                    metrics=['accuracy'])

# # else:
# # 	# initialize the model and train the network for weights and make predictions 
# # 	# on the detected faces
# # 	print("Initializing network")
# # 	model = cnn_model.initialize_model()
# # 	model = cnn_model.train_network(model, train_x, train_y, valid_x, valid_y, 30)

# # 	# serialize model to JSON and weights to HDF5 for faster access
# # 	model_json = model.to_json()
# # 	with open(model_path, "w") as json_file:
# # 	    json_file.write(model_json) 
# # 	model.save_weights("colour_files/model.h5")
# # 	print("Saved race model to disk")

# # take in commandline input and parse the arguments

# args = {
#     "path": "dataset/UTKFace/10_0_0_20170103200329407.jpg.chip.jpg",  # Input image path
#     "path2": "dataset/UTKFace/10_0_0_20170103233459275.jpg.chip.jpg" # Output image path
# }

# parser = argparse.ArgumentParser()
# parser.add_argument("-p", "--path", required=True,
# 	help="path to the input image")
# parser.add_argument("-p2", "--path2", required=True,
# 	help="path to the output image")
# args = vars(parser.parse_args())

# # initialize variables and detect the faces in the given image 
# # input_img_path = "../test_photos/group_photo7.jpg"

# input_img_path = args["path"]
# # input_img_path = args[r'C:\Users\komal\Downloads\AI_project\dataset\UTKFace\12_1_3_20170117174916351.jpg.chip.jpg']
# detections = face.detect_faces(input_img_path)
# test_x = np.zeros((len(detections), 50, 50, 3))
# img = cv2.imread(input_img_path)
# race_list = ["white", "black", "asian", "indian", "others"]

# # loop over each detected face, perform face alignment and resize
# for i, box in enumerate(detections):
# 	# find facial landmarks for detected face, and process the image
# 	landmarks = face.find_landmarks(input_img_path, box) 
# 	aligned_face = face.align_face(input_img_path, landmarks) # align face
# 	result = imresize(aligned_face, (50, 50, 3))

# 	# use the trained model to make a prediciton on the race of the detected face
# 	temp_face = np.array([list(result)])
# 	races = model.predict(temp_face)
# 	race_idx = np.argmax(races[0])

# 	# draw the bounding box around each face 
# 	(left, top, right, bottom) = box.astype("int")
# 	cv2.rectangle(img, (left, top), (right, bottom), (0, 0, 255), 2)

# 	# show the race above every detected face
# 	cv2.putText(img, "{}".format(race_list[race_idx]), (left - 10, top - 10), 
# 				cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
  
# # cv2.imwrite('result.jpg', img)
# cv2.imwrite(args["path2"], img)
# print("image saved")


import os
import numpy as np
import pickle
import tensorflow as tf
import argparse
from tensorflow.keras.models import model_from_json
from PIL import Image, ImageDraw, ImageFont
import sys
import cv2  # Add OpenCV for face detection
# sys.path.append('/content')  # Ensure that the current directory is in the Python path

# Face detection function using OpenCV's Haar Cascade
def detect_faces(image_path):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Load the image
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30),
                                          flags=cv2.CASCADE_SCALE_IMAGE)

    return faces  # Returns a list of bounding boxes of detected faces

# Process images and split them into training, validation, and test sets
print("Begin creating training and validation sets")
file_path = "colour_files/model_sets.txt"
if os.path.exists(file_path):  # If file exists, load file containing the sets
    print("sets file already exists")
    # Open file and load sets dictionary
    with open(file_path, "rb") as new_f:
        sets = pickle.load(new_f)
    # Initialize set variables
    train_x = sets["train_x"]
    train_y = sets["train_y"]
    valid_x = sets["valid_x"]
    valid_y = sets["valid_y"]

# Check if model exists
model_path = r"C:\Users\komal\OneDrive\Documents\NOTES\AI\Project\color_files\model.json"
if os.path.exists(model_path):
    # If file exists, load file containing the model
    print("Race Recognition model already created")
    # Load JSON and create model
    with open(model_path, 'r') as json_file:
        loaded_model_json = json_file.read()
    model = model_from_json(loaded_model_json)
    # Load weights into new model
    model.load_weights("colour_files/model.h5")
    print("Loaded race model from disk")

    # Compile the model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

# Parse command-line input
# Manually define the image paths since argparse won't work in Colab/Notebooks
args = {
    "path": "/content/drive/MyDrive/input_image.jpg",  # Input image path
    "path2": "/content/drive/MyDrive/output_image.jpg"  # Output image path
}

# Initialize variables and detect faces in the given image using the detect_faces function
input_img_path = args["path"]
detections = detect_faces(input_img_path)  # Detect faces using OpenCV
img = Image.open(input_img_path)  # Open the image using PIL
race_list = ["white", "black", "asian", "indian", "others"]



# Create an ImageDraw object to draw on the image
draw = ImageDraw.Draw(img)

# Loop over each detected face, perform face alignment (if needed), and resize
for i, (x, y, w, h) in enumerate(detections):
    # Crop the detected face region
    face_img = img.crop((x, y, x + w, y + h))

    # Resize using PIL instead of scipy
    result = face_img.resize((50, 50))  # Resize using PIL (no need for scipy)

    # Convert to a numpy array for model prediction
    result_array = np.array(result)

    # Ensure that the image has 3 channels
    if result_array.ndim == 2:  # Convert grayscale to 3 channels
        result_array = np.stack([result_array] * 3, axis=-1)

    # Normalize the image (if required by your model)
    result_array = result_array.astype('float32') / 255.0

    # Use the trained model to make a prediction on the race of the detected face
    temp_face = np.expand_dims(result_array, axis=0)  # Add batch dimension
    races = model.predict(temp_face)
    race_idx = np.argmax(races[0])

    # Draw the bounding box around each face
    draw.rectangle([x, y, x + w, y + h], outline="red", width=2)

    # Show the race above every detected face
    font = ImageFont.load_default()  # Load a default font
    draw.text((x - 10, y - 10), race_list[race_idx], fill="green", font=font)

# Save the resulting image
output_img_path = args["path2"]
img.save(output_img_path)  # Save the image using PIL
print("Image saved:", output_img_path)