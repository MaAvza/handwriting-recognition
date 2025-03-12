"""
Written by: Maya Avezbakiev 318631991
==============================================================================
This module provides helpful functionalities for handling handwritten document
images, encoding writer identities, and preparing data for training and 
evaluation. 
It includes functions to extract unique page 
identifiers from image filenames, perform label encoding, and one-hot encoding
of writer labels. By efficiently managing the data and encoding processes, 
this module serves as a building block for developing machine learning models 
that can accurately identify and differentiate writers based on their 
distinctive handwriting styles.
"""
import os
import cv2
import shutil
import scipy.io
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.applications import MobileNetV2
from sklearn.metrics import accuracy_score
from tensorflow.keras import regularizers, layers
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization


def main_processing(image_dir, data_dark_lines_dir, num_images):
    """
    Parameters
    ----------
    image_dir (str): The directory containing the original images.
    data_dark_lines_dir (str): The directory containing matrices.
    num_images (int): The total number of images to be processed.
    
    Description
    -----------
    the main entry point for processing image data. It creates subfolders for 
    organizing data, divides images into two output folders, and defines data 
    within these main folders. Specifically:
    It creates subfolders for both output_folder_A and output_folder_B 
    to organize training, validation, and test data.
    It divides images from image_dir into output_folder_A and output_folder_B 
    based on the specified num_images_A which represents the number of classes
    we will use as a development set.
    Finally, it defines data under the main folders by calling the 
    define_data_sets function for both output_folder_A and output_folder_B.
  
    Returns:
    -------
        None.
    """
    '''Create subfolders for each output folder'''
    output_folder_A = 'development_set'
    output_folder_B = 'test_set'
    create_subfolders(output_folder_A)
    create_subfolders(output_folder_B)

    '''Divide images using divide_images function'''
    divide_images(image_dir, output_folder_A, output_folder_B, num_images)

    '''define data under main folders'''
    define_data_sets(output_folder_A, data_dark_lines_dir, num_images)
    define_data_sets(output_folder_B, data_dark_lines_dir, num_images)


def create_subfolders(output_folder):
    """
    Parameters
    ----------
    output_folder (str): The directory where subfolders 
                        (training, validation, and test) will be created.
    
    Description
    -----------
    This function creates subfolders (training, validation, and test) within 
    the specified output_folder. It checks if these subfolders already exist, 
    and if not, it creates them.
    
    Returns:
    -------
        None.
    """
    '''define sub-folders'''
    subfolders = ['training', 'validation', 'test']
    for subfolder in subfolders:
        subfolder_path = os.path.join(output_folder, subfolder)
        os.makedirs(subfolder_path, exist_ok=True)


def divide_images(input_folder, output_folder_A, output_folder_B, num_images_A):
    """
    Parameters
    ----------
    input_folder (str): The directory containing the input data to be divided.
    output_folder_A (str): The directory where the first set of images will be 
                            copied.
    output_folder_B (str): The directory where the second set of images will 
                            copied.
    num_images_A (int): The number of images to be placed in output_folder_A. 
                        The remaining images will be placed in output_folder_B
    
    Description
    -----------
    The divide_images function is responsible for dividing the images from 
    input_folder into two output folders (output_folder_A and output_folder_B)
    based on the value of num_images_A. 
    It efficiently processes the images as follows:
    It retrieves a list of image files from the input_folder
    It iterates through each image file, copying it to either output_folder_A 
    or output_folder_B based on the index and the value of num_images_A
    The copied files are given unique filenames that include the 
    original filename and an index.
    
    Returns:
    -------
        None.
    """
    '''Get a list of image files in the input folder'''
    image_files = get_existing_images(input_folder)

    '''Move the first num_images_A images to folder A, rest to folder B'''
    for idx, image_file in enumerate(image_files):
        
        src_path = os.path.join(input_folder, image_file)
        image_name = os.path.splitext(image_file)[0]
        
        if idx < num_images_A:
            dst_folder = output_folder_A
        else:
            dst_folder = output_folder_B
        
        dst_path = os.path.join(dst_folder, f'{image_name}_{input_folder}.jpg')
        shutil.copy(src_path, dst_path)


def get_existing_files(directory, extensions):
    """
    Parameters
    ----------
    directory (str): The directory to search for files.
    extensions (list of str): List of file extensions to filter by.
    
    Description
    -----------
    Get a list of existing files in a directory with specified extensions.
    
    Returns:
    -------
        list of str: List of filenames with matching extensions.
    """
    return [filename for filename in os.listdir(directory)
            if any(filename.lower().endswith(ext) for ext in extensions)]


def get_existing_images(image_dir):
    """
    Parameters
    ----------
    image_dir (str): The directory to search for image files.
    
    Description
    -----------
    Get a list of existing image files in a directory.

    Returns:
    -------
        list of str: List of image filenames with matching extensions.
    """
    return get_existing_files(image_dir, ['.jpg', '.jpeg', '.png', '.gif'])


def get_existing_matrices(mat_dir):
    """
    Parameters
    ----------
    mat_dir (str): The directory to search for MATLAB data files.
    
    Description
    -----------   
    Get a list of existing MATLAB data files (`.mat` files) in a directory.

    Returns:
    -------
        list of str: List of filenames with a `.mat` extension in the specified
        directory.
    """
    return get_existing_files(mat_dir, ['.mat'])


def define_data_sets(output_folder, data_dark_lines_dir, num_images):
    '''
    Parameters
    ----------
    output_folder : (str): The directory where output data will be saved.
    data_dark_lines_dir : (str): The directory containing matrices.
    num_images : (int) The number of images to process./alternatively, the
                       the number of classes to process.
    
    Description
    -----------
    This function processes a specified number of images and associated 
    data files, performing several operations on each image and saving the 
    results
    
    Returns
    -------
    None.

    '''
    mat_dir = data_dark_lines_dir
    test_set_dir = os.path.join(output_folder, 'test')

    image_filenames = get_existing_images(output_folder)
    mat_filenames = get_existing_matrices(mat_dir)

    '''List image files once before the loop'''
    image_files = [os.path.join(output_folder, filename) for filename in image_filenames if filename.lower().endswith('.jpg')]

    for i in range(num_images):
        
        lines = []       # define a temporary list to hold the cropped sub-images

        if i >= len(image_files) or i >= len(mat_filenames):
            break

        image_filename = image_filenames[i]
        mat_filename = mat_filenames[i]

        image_path = os.path.join(output_folder, image_filename)
        mat_path = os.path.join(mat_dir, mat_filename)
        
        if not os.path.exists(image_path) or not os.path.exists(mat_path):
            break

        image = Image.open(image_path)
        image_name = os.path.splitext(image_filename)[0]
            
        '''load matrix which corresponds to the loaded image and extract
        important information that we will use to crop the image'''
        mat = scipy.io.loadmat(mat_path)
       
        SCALE_FACTOR = mat['SCALE_FACTOR'].flatten()[0]
        peaks_indices = mat['peaks_indices'].flatten() * SCALE_FACTOR
        delta = mat['delta'].flatten()[0]
        top_test_area = mat['top_test_area'].flatten()[0]
        bottom_test_area = mat['bottom_test_area'].flatten()[0]

        line_space = bottom_test_area-top_test_area
        center_width = image.width * 6 // 8
        cropping = ((image.width - delta)//SCALE_FACTOR)+center_width
        
        # Crop test line with margins
        test_line = image.crop((line_space, top_test_area, 
                                    cropping + line_space, bottom_test_area))
        test_line_filename = f'{image_name}_testLine_{i:02d}.jpg'
        test_line_path = os.path.join(test_set_dir, test_line_filename)
        test_line.save(test_line_path)
        
        for line_index in peaks_indices:

            '''check overlap with test line'''
            if top_test_area - 30 <= line_index <= bottom_test_area - 30:
                continue
            elif top_test_area + 30 <= line_index <= bottom_test_area + 30:
                continue

            cropped_line = crop_line(image, line_index, line_space, cropping)

            '''if an image is blank or has very few characters, eliminate 
                it from our list, after running this function a few times we 
                have come to 37 chracters as a minimal limit for images we 
                append to our sets'''
            if is_valid_line(cropped_line):

                line_label = f'{image_name}_line_{line_index}.jpg'
                lines.append((cropped_line, line_label))

        separate_data(lines, output_folder)


def separate_data(lines, output_folder, training_to_validation_ratio=0.3):
    '''
    Parameters
    ----------
    lines : (list) A list of tuples, each containing an image (PIL image) 
            and its associated label (str).
    output_folder : (str) The directory where the separated data 
                    (training and validation sets) will be saved.
    training_to_validation_ratio : (float, optional) The ratio of data to be 
    used for training compared to validation. Default is 0.3 (30% for validation).

    Description
    -----------
    This function separates the provided list of image-label pairs into
    training and validation sets based on the specified ratio. It shuffles 
    the data before separation.
    It saves the images in the respective training and validation folders 
    under output_folder.
    
    Returns
    -------
    None.    
    '''
    train_folder = os.path.join(output_folder, 'training')
    validation_folder = os.path.join(output_folder, 'validation')

    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(validation_folder, exist_ok=True)

    indices = list(range(len(lines))) 

    for i in range(len(lines)):
        index = indices[i]
        line, label = lines[index]
        if i <= int(len(lines) * (1-training_to_validation_ratio)):
            subset_folder = train_folder
        else:
            subset_folder = validation_folder
        line_filename = f'{label}'  # Assign a unique name to each line image
        line_path = os.path.join(subset_folder, line_filename)

        line.save(line_path)


def is_valid_line(cropped_line):
    '''
    Parameters
    ----------
    cropped_line : (PIL image) A cropped line image to be validated.

    Description
    -----------
    This function determines whether a cropped line image is valid by checking 
    the number of contours in the image. If the number of contours is greater 
    than or equal to 37, the line is considered valid.
    
    Returns
    -------
    True if the cropped line image is valid (has enough contours).
    False otherwise.
    '''

    grey_img = ImageOps.grayscale(cropped_line)
    cv_image = cv2.cvtColor(np.array(grey_img), cv2.COLOR_RGB2BGR)
    edges = cv2.Canny(cv_image, threshold1=100, threshold2=255)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
    return len(contours) >= 37


def crop_line(image, line_index, line_space, cropping):
    '''
    Parameters
    ----------
    image (PIL image): The original image from which to crop the line.
    line_index (float): The index representing the vertical position of the 
                        line within the image.
    line_space (int): The height of the line to be cropped.
    cropping (int): The width at which to crop the line.
   
    Description
    -----------
    This function crops a line from the original image based on the provided
    parameters (line position and dimensions).
    
    Returns
    -------
    cropped_line (PIL image): The cropped line image.
    '''
    line_top = int(line_index)
    line_bottom = line_top + line_space
    cropped_line = image.crop((line_space , line_top,  
                                          cropping + line_space, line_bottom))
	
    return cropped_line


def save_datasets_as_npz(data_folder, file_name):
    '''
    Parameters
    ----------
    data_folder (str): The directory where the training, validation, 
                        and test datasets are stored.
    file_name (str): The name of the output .npz file that will contain the 
                    saved datasets.
   
    Description
    -----------
    This function saves the training, validation, and test datasets along with
    their encoded labels as a compressed .npz file in the specified data_folder. 
    The datasets are expected to be organized in subfolders within data_folder.
    
    Returns
    -------
        None.
    '''
    train_folder = os.path.join(data_folder, 'training')
    val_folder = os.path.join(data_folder, 'validation')
    test_folder = os.path.join(data_folder, 'test')

    train_images = []
    val_images = []
    test_images = []

    train_encoded_labels = get_encoded_labels(os.listdir(train_folder))
    val_encoded_labels = get_encoded_labels(os.listdir(val_folder))
    test_encoded_labels = get_encoded_labels(os.listdir(test_folder))

    train_images = load_images_from_folder(train_folder)
    val_images = load_images_from_folder(val_folder)
    test_images = load_images_from_folder(test_folder)

    save_path = os.path.join(data_folder, f'{file_name}.npz')
    np.savez_compressed(save_path,
                        train_images=np.array(train_images),
                        train_labels=np.array(train_encoded_labels),
                        val_images=np.array(val_images),
                        val_labels=np.array(val_encoded_labels),
                        test_images=np.array(test_images),
                        test_labels=np.array(test_encoded_labels))


def load_images_from_folder(folder_path, target_height=181, resize_cache=True):
    """
    Parameters
    ---------- 
    folder_path (str): The path to the folder containing the images.
    target_height (int, optional): The desired height for resizing the images.
    resize_cache (bool, optional): Whether to cache resized images on disk.

    Description
    -----------
    Load images from a folder and optionally resize them.
       
    Returns
    -------
        np.ndarray: A NumPy array containing the loaded and resized images.
    """
    # Check if resized images are cached on disk
    resized_cache_path = os.path.join(folder_path, f'resized_cache_{target_height}.npz')
    
    if resize_cache and os.path.exists(resized_cache_path):
        data = np.load(resized_cache_path)
        return data['images']

    images = os.listdir(folder_path)
    data = []

    for file in images:
        
        image_path = os.path.join(folder_path, file)
        
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Load image in grayscale
        
        target_width = 942
        # Resize the image using OpenCV
        image = cv2.resize(image, (target_width , target_height), interpolation=cv2.INTER_NEAREST)
        
        # Append the resized image to the data
        data.append(image)

    data = np.stack(data)

    if resize_cache:
        # Cache the resized images on disk for future use
        np.savez_compressed(resized_cache_path, images=data)

    return data


def get_page_identifier(image_name):
    """
    Parameters
    ---------- 
    image_name (str): The name of an image file.

    Description
    -----------
    This function extracts and returns a page identifier from the given 
    image_name. It assumes that the image name follows a specific format, 
    "linesX_Page_Y.jpg," where X and Y are numbers. 
    The page identifier is constructed by joining the first three parts of 
    the image name with underscores
       
    Returns
    -------
        page_identifier (str): The extracted page identifier.
    """
    # Assumes the format is linesX_Page_Y.jpg
    parts = image_name.split("_")[:3]
    page_identifier = "_".join(parts[:3])
    
    return page_identifier


def get_encoded_labels(images):
    """
    Parameters
    ---------- 
    images (list of str): A list of image file names.

    Description
    -----------
    This function encodes a list of image file names into numerical labels 
    using the LabelEncoder from scikit-learn. It first extracts page 
    identifiers from the image names using get_page_identifier and then
    performs label encoding on those identifiers.
    
    Returns
    -------
        encoded_labels (numpy.ndarray): An array of encoded labels 
                                        corresponding to the input image names.
    """
    label_encoder = LabelEncoder()
    page_identifiers = [get_page_identifier(image_name) for image_name in images]
    page_identifiers = [str(identifier) for identifier in page_identifiers]
    encoded_labels = label_encoder.fit_transform(page_identifiers)

    return encoded_labels


def encode_labels_to_one_hot(labels, num_classes):
    """
    Parameters
    ---------- 
    labels (numpy.ndarray): An array of encoded labels.
    num_classes (int): The total number of classes (unique labels)

    Description
    -----------
    This function one-hot encodes a set of labels using the OneHotEncoder from
    scikit-learn. It reshapes the input labels into a 2D array and then 
    performs one-hot encoding with the specified number of classes.
    
    Returns
    -------
        one_hot_labels (numpy.ndarray): A 2D array containing the 
        one-hot encoded labels. Each row corresponds to a label, 
        and each column represents a class, with 1s indicating class membership.
    """    
    encoder = OneHotEncoder(sparse_output=False, categories='auto')
    labels = labels.reshape((-1, 1))
    labels = encoder.fit_transform(labels)
    
    return labels


def load_data_from_npz(file_path, folder):
    """
    Parameters
    ---------- 
    file_path (str): The directory where the .npz file is located.
    folder (str): The name of the .npz file (without the extension) to load 
                    data from.

    Description
    -----------
    This function loads data from a specified .npz file, 
    specifically loading training images, training labels, validation images, 
    validation labels, test images, and test labels. The data is returned as
    Numpy arrays.
       
    Returns
    -------
       Tuple containing the following Numpy arrays:
       * train_images (ndarray): Training images.
       * train_labels (ndarray): Encoded labels for training data.
       * val_images (ndarray): Validation images.
       * val_labels (ndarray): Encoded labels for validation data.
       * test_images (ndarray): Test images.
       * test_labels (ndarray): Encoded labels for test data.
    """

    npz_file_path = os.path.join(file_path, f'{folder}.npz')
    '''we're loading images as Numpy arrays, which are themselves objects,
    we will set allow_pickle = True'''
    data = np.load(npz_file_path, allow_pickle=True)
    
    train_images = data['train_images']
    train_labels = data['train_labels']
    val_images = data['val_images']
    val_labels = data['val_labels']
    test_images = data['test_images']
    test_labels = data['test_labels']

    return train_images, train_labels, val_images, val_labels, test_images, test_labels


def display_samples(images, labels, num_samples=9, title=None):
    """
    Parameters
    ----------
        images (numpy.ndarray): The dataset of images with shape
                                (num_samples, 181, 942).
        labels (numpy.ndarray): The corresponding labels for the images.
        num_samples (int): The number of samples to display (default is 9).
        title (str): Optional title for the plot (default is None).

    Description
    -----------
    Display a selected number of samples from a dataset of grayscale images.

    Returns:
    -------
        None.
    """
    if num_samples <= 0:
        return

    # Ensure num_samples is not greater than the number of available samples
    num_samples = min(num_samples, len(images))

    # Create a subplot grid
    num_cols = min(num_samples, 3)
    num_rows = (num_samples // num_cols) + (num_samples % num_cols > 0)
    fig, axes = plt.subplots(num_rows, 3, figsize=(12, 2 * num_rows))

    # Flatten the images if they are not already 2D
    if len(images.shape) == 3:
        images = images.reshape(-1, 181, 942)

    for i in range(num_samples):
        ax = axes[i // 3, i % 3]
        ax.imshow(images[i], cmap='gray')
        ax.title.set_text(f"Label: {labels[i]}")
        ax.axis('off')

    if title:
        plt.suptitle(title)

    plt.tight_layout()
    plt.show()
    
    
def check_label_mismatches(one_hot_labels, expected_labels):
    """
    Parameters
    ----------
        one_hot_labels (numpy.ndarray): One-hot encoded labels as a 2D
                                        numpy array.
        expected_labels (numpy.ndarray): Expected class indices as a
                                        1D numpy array.

    Description
    -----------
    Check for mismatches between one-hot encoded labels and expected class 
    indices.

    Returns:
    -------
        list: A list of mismatched sample indices.
    """
    # Step 1: Convert one-hot encoded labels back to class indices
    class_indices = np.argmax(one_hot_labels, axis=1)

    # Step 2: Compare class indices with expected labels
    matches = (class_indices == expected_labels)

    # Step 3: Find and return mismatched sample indices
    mismatch_indices = np.where(matches == False)[0]

    return mismatch_indices


def preprocess_images(images):
    """
    Parameters
    ----------
        images (numpy.ndarray): The input images as a numpy array.
   
    Description
    -----------
        Preprocess a set of images by reshaping and normalizing them.

    Returns:
    -------
        numpy.ndarray: The preprocessed images.
    """
    # Reshape the images
    images = images.reshape((-1, 181, 942, 1))

    # Normalize the images
    images = images.astype('float32') / 255.0

    return images


def create_pretrained_resnet_model(input_shape, num_classes, pretrained_weights='imagenet'):
    """
    Create a pretrained ResNet model for transfer learning.

    Parameters:
    - input_shape (tuple): The shape of the input images, e.g., (224, 224, 3).
    - num_classes (int): The number of output classes.
    - pretrained_weights (str): Specify 'imagenet' to use ImageNet pretrained weights.

    Returns:
    - model (tensorflow.keras.Model): The pretrained ResNet model.
    """
    base_model = tf.keras.applications.ResNet50(
        include_top=False,
        weights=pretrained_weights,
        input_tensor=tf.keras.layers.Input(shape=input_shape),
    )

    # Unfreeze some of the later layers for fine-tuning (optional)
    for layer in base_model.layers[-5:]:
        layer.trainable = True

    # Customize the output layer for your number of classes
    x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs=base_model.input, outputs=outputs)

    return model


def create_custom_model(SAMPLES, IMG_HEIGHT, IMG_WIDTH, num_classes):
    """
    Parameters
    ----------
    IMG_HEIGHT (int): The heighta of the input images.
    IMG_WIDTH (int): The width of the input images.
    NUM_CLASSES (int): The number of classes (unique writers) for classification.

    Description
    -----------
    responsible for building and configuring a custom convolutional neural 
    network (CNN) model for image classification tasks. It is designed to 
    create a deep learning model that combines a pre-trained MobileNetV2
    feature extractor with additional custom layers.

    Returns:
    --------
        model (Keras Sequential model): The custom model configured for the 
                                           given task. This model combines a 
                                           MobileNetV2 feature extractor with 
                                           additional convolutional, pooling, 
                                           and dense layers.
    """
    base_model = MobileNetV2(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
                             include_top=False, 
                             weights='imagenet', 
                             classes=num_classes)

    # Freeze the weights of the pre-trained MobileNet layers
    for layer in base_model.layers:
        layer.trainable = False

    model = Sequential()

    model.add(base_model)

 # Convolutional layers
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Conv2D(128, kernel_size=(2, 2), activation='relu', kernel_regularizer=regularizers.l2(0.03)))
    # model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    # Flatten the output
    model.add(Flatten())

    # Dense layers
    model.add(Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model.add(BatchNormalization())
    model.add(Dropout(0.7))
    model.add(Dense(num_classes, activation='softmax'))

    # Compile the model with a slightly higher learning rate
    opt = Adam(learning_rate=0.0003)
    model.compile(optimizer=opt, 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'],
                  run_eagerly=True)

    model.summary()

    return model

def create_model(IMG_HEIGHT, IMG_WIDTH, NUM_CLASSES):
    """
    Parameters
    ----------
    IMG_HEIGHT (int): The heighta of the input images.
    IMG_WIDTH (int): The width of the input images.
    NUM_CLASSES (int): The number of classes (unique writers) for classification.

    Description
    -----------
    Create and compile a convolutional neural network (CNN) model for writer 
    recognition.

    Returns:
    --------
        tensorflow.keras.models.Sequential: A compiled CNN model for writer 
        recognition.
    """
    model = Sequential()

    # Add the convolutional layers
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=(128, 128, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Flatten the output of the convolutional layers
    model.add(Flatten())

    # Add dense layers with L2 regularization
    model.add(Dense(256, activation='relu', kernel_regularizer=l2(0.01)))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu', kernel_regularizer=l2(0.01)))
    model.add(Dropout(0.5))
    model.add(Dense(40, activation='softmax')) # Use softmax for multi-class classification

    opt = Adam(learning_rate=0.0003)

    model.compile(optimizer=opt,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.summary()

    return model


def apply_sliding_window(images, labels, window_size=50, stride=10):
    """
    Parameters
    ----------
        images (list of numpy arrays): A list of input images from which sliding
                                    windows will be extracted.
        labels (list or numpy array): A list or numpy array containing the 
                                    labels or target values corresponding to 
                                    the input images. These labels will be 
                                    associated with the extracted sliding windows.
        window_size (int): The size of the sliding window. It determines the
                            dimensions of the extracted windows. For example, 
                            if window_size is set to 50, the extracted windows
                            will be 50x50 pixels.
        stride (int): The step size by which the sliding window moves both 
                        horizontally and vertically across each image. It 
                        determines the overlap between consecutive windows.

    Description
    -----------
    extract sliding windows from a set of images, along with their corresponding
    labels. These windows are generated by moving a fixed-size window 
    (defined by window_size) across each image with a specified stride. 
    The function then converts these windows and their labels into numpy arrays
    and returns them as a pair of outputs.

    Returns:
    --------
        windows (numpy array): An array containing the extracted sliding windows
                                as numpy arrays.
        window_labels (numpy array): An array containing the labels or target 
                         values corresponding to each extracted sliding window. 
    """
    # Initialize lists to hold the windows and their labels
    windows = []
    window_labels = []

    # Slide the window across each image
    for image, label in zip(images, labels):
        for i in range(0, image.shape[0] - window_size, stride):
            for j in range(0, image.shape[1] - window_size, stride):
                # Get the window
                window = image[i:i+window_size, j:j+window_size]
               
                # Add the window and its label to the lists
                windows.append(window)
                window_labels.append(label)

    # Convert the lists to numpy arrays
    windows = np.array(windows)
    window_labels = np.array(window_labels)

    return windows, window_labels


def batch_generator(images, labels, batch_size):
    # Get total number of samples in the data
    samples = images.shape[0]
    
    # Create batches
    for start in range(0, samples, batch_size):
        end = min(start + batch_size, samples)
        yield images[start:end], labels[start:end]

            
def separate_data_by_class(image_paths, labels, num_classes=101):
    data_a = []
    labels_a = []
    data_b = []
    labels_b = []

    for image_path, label in zip(image_paths, labels):
        # Check if the label is less than or equal to the specified number of classes
        if label <= num_classes:
            data_a.append(image_path)
            labels_a.append(label)
        else:
            data_b.append(image_path)
            labels_b.append(label)
            
    
    return np.array(data_a), np.array(labels_a), np.array(data_b), np.array(labels_b)


from scipy import stats

def sliding_window_predict(images, labels, model, window_size=50, stride=10):
    """
    Parameters
    ----------
        images (list of numpy arrays): A list of input images for which 
                                                predictions need to be made.
        labels (list or numpy array): A list or numpy array containing the true
                    labels or target values corresponding to the input images. 
                    This parameter is not used for prediction but can be useful 
                    for evaluation purposes.
        model (Keras or TensorFlow model): A pre-trained machine learning model
                                        (e.g., a convolutional neural network) 
                                        capable of making predictions on image data.
        window_size (int, optional): The size of the sliding window that moves
                                    across each image. Default is set to 50.
        stride (int, optional): The step size by which the sliding window moves
                        horizontally across each image. Default is set to 10.

    Description
    -----------
    perform predictions on a set of images using a sliding window approach and 
    combine these predictions to provide a final prediction for each input image

    Returns:
    --------
        final_predictions (numpy array): An array containing the final 
        predictions for each input image. These predictions are obtained by
        processing the input images using the specified model and combining 
        predictions from sliding windows, typically using majority voting. 
        The shape of this array is (number_of_images,).
    """
    image_indices = []

    # Slide the window across each image
    for idx, image in enumerate(images):
        image_indices.append(idx)

    # Convert the lists to numpy arrays
    image_indices = np.array(image_indices)

    # Make predictions on all windows
    window_predictions = model.predict(images)

    # Combine the window predictions into a final prediction for each image
    final_predictions = []
    for idx in range(max(image_indices) + 1):
        # Get the predictions for this image
        image_predictions = window_predictions[image_indices == idx]
        
        # Use majority voting as an example
        final_prediction = stats.mode(np.argmax(image_predictions, axis=1))[0]
        final_predictions.append(final_prediction)

    final_predictions = np.array(final_predictions)

    return final_predictions

           
def ensemble_predict(models, image):
    """
    Parameters
    ----------
        models (list): List of trained models for different class groups.
        image (numpy.ndarray): Image data to classify.

    Description
    -----------
    Predict class probabilities for an image using an ensemble of models.

    Returns:
    --------
        numpy.ndarray: Combined class probabilities.
    """
    # Make predictions with each model
    probabilities = [model.predict(image) for model in models]

    # Combine predictions (simple averaging)
    combined_probabilities = sum(probabilities) / len(models)

    return combined_probabilities


def plot_loss_accuracy(history, title):
    """
    Parameters
    ----------
    history (tensorflow.keras.callbacks.History): The training history 
                                                obtained from model training.

    Description
    -----------
    Plot training and validation loss and accuracy over epochs.

    Returns:
    --------
        None.
    """
    # Extract the number of epochs from the training history
    epochs = range(1, len(history.history['accuracy']) + 1)

    # Create a figure for plotting
    plt.figure(figsize=(12, 6))

    # Plot training and validation accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history.history['accuracy'], 'bo', label='Training acc')
    plt.plot(epochs, history.history['val_accuracy'], 'r', label='Validation acc')
    plt.title(f'Training and validation accuracy - {title}')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plot training and validation loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history.history['loss'], 'bo', label='Training loss')
    plt.plot(epochs, history.history['val_loss'], 'r', label='Validation loss')
    plt.title(f'Training and validation loss - {title}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Ensure a tight layout for better visualization
    plt.tight_layout()

    # Display the plot
    plt.show()
    

def plot_confusion_matrix(y_true, y_pred, num_classes):
    """
    Parameters
    ----------
    y_true (numpy.ndarray): True labels as a 1D or 2D numpy array.
    y_pred (numpy.ndarray): Predicted labels as a 1D or 2D numpy array.
    num_classes (int): The number of classes (unique labels) in the classification task.

    Description
    -----------
    Plot a confusion matrix based on true and predicted labels.

    Returns:
    --------
        None.
    """

    # Ensure that y_true and y_pred are 1D arrays of integer class labels
    y_true_int = np.argmax(y_true, axis=1) if len(y_true.shape) > 1 else y_true
    y_pred_int = np.argmax(y_pred, axis=1) if len(y_pred.shape) > 1 else y_pred

    # Compute the confusion matrix
    cm = confusion_matrix(y_true_int, y_pred_int)
    
    # Display the confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=range(num_classes))
    disp.plot(cmap="Blues", values_format="d")
    plt.show()


def print_accuracy(y_true, y_pred):
    """
    Parameters
    ----------
    y_true (numpy.ndarray): True labels as a 1D numpy array.
    y_pred (numpy.ndarray): Predicted labels as a 1D numpy array.

    Description
    -----------
    Print the accuracy score based on predicted and true labels.

    Returns:
    --------
        None.
    """

    # Calculate accuracy score
    acc = accuracy_score(y_true, y_pred)
    
    # Print the accuracy score
    print(f"Accuracy: {acc:.4f}")


def display_misclassified_images(test_images, test_labels, model):
    """
    Parameters
    ----------
    test_images (numpy.ndarray): Array of test images.
    test_labels (numpy.ndarray): True labels corresponding to the test images.
    model (tensorflow.keras.models.Model): Trained neural network model for
                                            predictions.

    Description
    -----------
    Display misclassified images along with their true and predicted labels.

    Returns:
    --------
        None.
    """
    
    '''Get the predictions for the test dataset'''
    predictions = model.predict(test_images)

    ''' Find the misclassified images'''
    misclassified_indices = np.argmax(
        predictions, axis=1) != np.argmax(test_labels, axis=1)

    ''' Get the misclassified images and labels'''
    misclassified_images = test_images[misclassified_indices]
    misclassified_true_labels = np.argmax(
        test_labels[misclassified_indices], axis=1)
    misclassified_predicted_labels = np.argmax(
        predictions[misclassified_indices], axis=1)

    ''' Display the misclassified images in rows'''
    num_images = len(misclassified_images)
    fig, axes = plt.subplots(num_images, 1, figsize=(5, 2 * num_images))
    for i, ax in enumerate(axes):
        ax.imshow(misclassified_images[i].squeeze(), cmap='gray')
        ax.set_title(
            f"True: {misclassified_true_labels[i]}\nPredicted: {misclassified_predicted_labels[i]}")
        ax.axis('off')
    plt.tight_layout()
    plt.show()