import os
import cv2
import numpy as np
import tensorflow as tf

CLASSES = ['ball', 'net']

def preprocess_frame(frame, input_size):
    ''' Preprocess the input frame to feed to the TFLite model 
    '''
    img = tf.convert_to_tensor(frame)  # Convert to TensorFlow tensor
    resized_img = tf.image.resize(img, input_size)  # Resize to model input size
    resized_img = resized_img[tf.newaxis, :]
    resized_img = tf.cast(resized_img, dtype=tf.uint8)
    return resized_img


def detect_objects(interpreter, frame, threshold):
    ''' Returns a list of detection results for the frame 
    '''
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


def get_basket_roi(input_video_path, interpreter, threshold, num_frames):
    ''' Detect the basket's ROI from the initial frames of the video 
    '''
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise ValueError(f"Error opening video file: {input_video_path}")
    
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    _, input_height, input_width, _ = interpreter.get_input_details()[0]['shape']

    roi = None
    for _ in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            break
        preprocessed_frame = preprocess_frame(frame, (input_height, input_width))
        results = detect_objects(interpreter, preprocessed_frame, threshold=threshold)

        for obj in results:
            if CLASSES[int(obj['class_id'])] == 'net':
                ymin, xmin, ymax, xmax = obj['bounding_box']
                xmin = int(xmin * frame_width)
                xmax = int(xmax * frame_width)
                ymin = int(ymin * frame_height)
                ymax = int(ymax * frame_height)

                width = xmax - xmin
                height = ymax - ymin

                # Expand the bounding box
                xmin = max(0, int(xmin - 0.6 * width))  # Increase leftward expansion by 0.6 times
                xmax = min(frame_width, int(xmax + 0.6 * width))  # Increase rightward expansion by 0.6 timess
                ymin = max(0, int(ymin - 1.6 * height))  # Increase upward expansion 1.6 times
                ymax = min(frame_height, ymax)  # Keep the bottom fixed

                roi = (xmin, ymin, xmax, ymax)
                break
        if roi:
            break

    cap.release()
    if not roi:
        raise ValueError("Basket (net) not detected in the initial frames.")
    return roi


def crop_and_save_frames(input_video_path, output_folder_path, roi):
    ''' Crop each frame of the video based on the ROI and save as individual .jpg images 
    '''
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    # Open the video file
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise ValueError(f"Error opening video file: {input_video_path}")

    xmin, ymin, xmax, ymax = roi
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Crop the frame
        cropped_frame = frame[ymin:ymax, xmin:xmax]

        # Save the cropped frame as .jpg
        frame_filename = f"{output_folder_path}/frame_{frame_count:04d}.jpg"
        cv2.imwrite(frame_filename, cropped_frame)

        frame_count += 1

    cap.release()
    print(f"Cropped frames saved to {output_folder_path}")


def process_video(input_video_path, model_path, threshold, num_frames):
    ''' Main function to process the video 
    '''
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # Get the ROI
    roi = get_basket_roi(input_video_path, interpreter, threshold, num_frames)

    # Extract the video name without extension for folder naming
    video_name = os.path.splitext(os.path.basename(input_video_path))[0]
    output_folder_path = os.path.join(os.getcwd(), "output", video_name)

    # Save cropped frames as .jpg files
    crop_and_save_frames(input_video_path, output_folder_path, roi)



# Main execution
cwd = os.getcwd()
MODEL_PATH = f'{cwd}/models'
MODEL_NAME = 'ballogy_1_eff0.tflite'
DETECTION_THRESHOLD = 0.2

# Input video path
INPUT_VIDEO_PATH = f"{cwd}/videos/testing_video.mp4"

# Load the TFLite model
model_path = f'{MODEL_PATH}/{MODEL_NAME}'

# Process the video
process_video(INPUT_VIDEO_PATH, model_path, threshold=DETECTION_THRESHOLD, num_frames=300)
