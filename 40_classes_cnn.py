"""
Written by: Maya Avezbakiev 318631991
==============================================================================
classification of 40 classes
------------------------------------------------------------------------------
this program is a complete pipeline for image classification, including data 
loading, preprocessing, model creation, training, evaluation, and result
 reporting.
"""
import os
import time
import numpy as np
import CodeLib as code
import tensorflow as tf
from keras.models import model_from_json
from keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

start_time = time.time()

input_folder1 = '3_ImagesLinesRemovedBW'
output_folder_A = 'development_set'
output_folder_B = 'test_set'
data_dark_lines_dir = '5_DataDarkLines'  
file_name = 'data_sets'

num_classes = 40

# Parameters
batch_size = 17  # Adjust as needed
num_epochs = 100  # Adjust as needed

train_images, train_labels, val_images, val_labels, test_images, test_labels = code.load_data_from_npz(output_folder_A,file_name)

train_labels = code.encode_labels_to_one_hot(train_labels, num_classes)
val_labels = code.encode_labels_to_one_hot(val_labels, num_classes)
test_labels = code.encode_labels_to_one_hot(test_labels, num_classes)

train_images = code.preprocess_images(train_images)
val_images = code.preprocess_images(val_images)
test_images = code.preprocess_images(test_images)

train_images = np.repeat(train_images, 3, axis = -1)
val_images = np.repeat(val_images, 3, axis = -1)
test_images = np.repeat(test_images, 3, axis = -1)

train_slide_images, train_slide_labels = code.apply_sliding_window(train_images,
                                                     train_labels,
                                                     window_size = 128,
                                                     stride=64)
val_slide_images, val_slide_labels = code.apply_sliding_window(val_images,
                                                 val_labels,
                                                 window_size = 128,
                                                 stride=64)
test_slide_images, test_slide_labels = code.apply_sliding_window(test_images,
                                                   test_labels,
                                                   window_size = 128,
                                                   stride=64)

print(f'train data shape: {train_slide_images.shape}')
print(f'val data shape: {val_slide_images.shape}')
print(f'test data shape: {test_slide_images.shape}')

json_path = 'model.json'
h5_path = 'model.h5'

# Check if both files exist
if os.path.exists(json_path) and os.path.exists(h5_path):
    # Load the model architecture from the .json file
    with open(json_path, 'r') as json_file:
        loaded_model_json = json_file.read()
    model = model_from_json(loaded_model_json)

    # Load the weights into the new model
    model.load_weights(h5_path)

    # Compile the model before using it
    model.compile(optimizer='adam', 
                         loss='categorical_crossentropy', 
                         metrics=['accuracy'],
                         run_eagerly=True)

    print("Model loaded successfully.")
else:
    print("Model files not found, creating new model")
    model = code.create_model(train_slide_images.shape[0], 128, 128, num_classes)
    
checkpointer = tf.keras.callbacks.ModelCheckpoint(filepath = 'model.h5', 
                                                          save_best_only= True)
                                
es = EarlyStopping(monitor= 'val_loss', patience = 25, restore_best_weights=True)

history = model.fit(train_slide_images,
                    train_slide_labels,
                    callbacks = [es, checkpointer],
                    epochs = 100,
                    batch_size = 17,
                    validation_data=(val_slide_images, val_slide_labels))

'''serialize model' to J'SON'''
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
'''serialize weights to HDF5'''
model.save_weights("model.h5")
code.plot_loss_accuracy(history, title='model')

final_predictions = code.sliding_window_predict(test_slide_images, test_slide_labels,model,
                                              window_size= 128, stride = 64)

# Calculate class predictions (argmax) and true labels
predicted_classes = np.argmax(final_predictions, axis=1)
true_classes = np.argmax(test_labels, axis=1)

print("test predict acc: ")
code.print_accuracy(true_classes, predicted_classes)

code.plot_loss_accuracy(history)
y_pred = model.predict(test_images)

accuracy = accuracy_score(true_classes, predicted_classes)
precision = precision_score(true_classes, predicted_classes, average='weighted')
recall = recall_score(true_classes, predicted_classes, average='weighted')
f1 = f1_score(true_classes, predicted_classes, average='weighted')

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1-score: {f1}")

print("--- %s seconds ---" % (time.time() - start_time))
