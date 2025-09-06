import os
import cv2
import tensorflow as tf

CLASSES = ['goal']

def initialize_interpreter(model_path):
    ''' Initialize the TFLite interpreter '''
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

def preprocess_frame(frame, input_size, is_grayscale=False):
    ''' Preprocess the input frame to feed to the TFLite model '''
    img = tf.convert_to_tensor(frame)  # Convert to TensorFlow tensor
    resized_img = tf.image.resize(img, input_size)  # Resize to model input size
    resized_img = resized_img[tf.newaxis, :]
    resized_img = tf.cast(resized_img, dtype=tf.uint8)
    return resized_img

def detect_objects(interpreter, frame, threshold):
    ''' Returns a list of detection results for the frame '''
    signature_fn = interpreter.get_signature_runner()
    output = signature_fn(images=frame)
    count = int(np.squeeze(output['output_0']))
    scores = np.squeeze(output['output_1'])
    classes = np.squeeze(output['output_2'])
    boxes = np.squeeze(output['output_3'])

    results = []
    for i in range(count):
        if scores[i] >= threshold:
            result = {
                'bounding_box': boxes[i],
                'class_id': classes[i],
                'score': scores[i]
            }
            results.append(result)
    return results

def process_goal_detection(input_folder_path, output_folder_path, model_path, threshold):
    ''' Process the cropped images to detect goals '''
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    # Load the TFLite model
    interpreter = initialize_interpreter(model_path)
    input_details = interpreter.get_input_details()
    input_shape = input_details[0]['shape'][1:3]  # Height and width of the input

    # Get all .jpg files in the input folder
    image_files = [f for f in os.listdir(input_folder_path) if f.endswith('.jpg')]
    image_files.sort()  # Ensure the files are sorted by frame order (if needed)

    frame_count = 0

    for image_file in image_files:
        image_path = os.path.join(input_folder_path, image_file)
        frame = cv2.imread(image_path)
        
        if frame is None:
            print(f"Error reading image: {image_file}")
            continue

        # Preprocess frame
        preprocessed_frame = preprocess_frame(frame, input_shape, is_grayscale=True)

        # Inference on the preprocessed frame
        detections = detect_objects(interpreter, preprocessed_frame, threshold)

        # Annotate the frame
        annotated_frame = frame.copy()  # Create a copy of the frame for annotation
        for obj in detections:
            box = obj['bounding_box']
            x_min = int(box[1] * frame.shape[1])
            y_min = int(box[0] * frame.shape[0])
            x_max = int(box[3] * frame.shape[1])
            y_max = int(box[2] * frame.shape[0])

            # Draw bounding box and label
            cv2.rectangle(annotated_frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            label = f"ID: {obj['class_id']} Score: {obj['score']:.2f}"
            cv2.putText(annotated_frame, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Save the annotated frame as a .jpg image
        annotated_image_path = os.path.join(output_folder_path, f"annotated_{image_file}")
        cv2.imwrite(annotated_image_path, annotated_frame)

        frame_count += 1
        print(f"Processed image {frame_count}: {image_file}")

    print(f"All annotated images saved to {output_folder_path}")



# Main execution
cwd = os.getcwd()
MODEL_PATH = f'{cwd}/models'
MODEL_NAME = 'mixed_model2_1.tflite'
DETECTION_THRESHOLD = 0.2

# Define the input and output folders
# The input folder should be the folder containing the cropped images from the previous model
INPUT_FOLDER_PATH = f"{cwd}/output/cropped_images"  # Folder containing cropped images
OUTPUT_FOLDER_PATH = f"{cwd}/output/annotated_images"  # Folder to save annotated images

# Load the TFLite model
model_path = f'{MODEL_PATH}/{MODEL_NAME}'

# Process the goal detection on cropped images
process_goal_detection(INPUT_FOLDER_PATH, OUTPUT_FOLDER_PATH, model_path, threshold=DETECTION_THRESHOLD)
