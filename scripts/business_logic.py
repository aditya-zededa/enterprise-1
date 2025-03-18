import cv2, shutil, json
import os
import requests
import time
from skimage.metrics import structural_similarity as ssim
from azure.storage.blob import BlobServiceClient

new_width = 640
new_height = 640
output_dir = 'extracted_frames'
os.makedirs(output_dir, exist_ok=True)

connection_string = ''
container_name = 'retrain-images'

def cleanup_resources(video_capture):
    # Remove the output directory
    shutil.rmtree(output_dir)
    # Release the video capture object and close the OpenCV window
    video_capture.release()
    cv2.destroyAllWindows()

def log_response(response):
    log_server = 'http://20.163.29.125:8084/post'
    log_resp = requests.post(log_server, json={"response": response})

    if log_resp.status_code == 200:
        print("Cloud sink logged successfully!")
    else:
        print(f"Failed to log response. Server returned status code: {log_resp.status_code}")

def format_response(response):
    err_msg = 'No objects detected with confidence more than 0.85'
    if response.get('result') and response.get('result').get('bounding_boxes'):
        str = ','.join([f'{box.get("label")}' for box in response.get('result').get('bounding_boxes') if box.get("value") > 0.85])
        if str == '':
            print(err_msg)
            return err_msg
        print(f'Objects detected: {str}')
        return 'Objects detected: ' + str
    else:
        print('Invalid response from server: ', response)
        return err_msg
    
def upload_azure_blob(container_client, filename, image_path):
    blob_client = container_client.get_blob_client(filename)
    with open(image_path, 'rb') as data:
        blob_client.upload_blob(data, overwrite=True)
        print(f'{image_path} uploaded to Azure Blob Storage')

def send_to_monitor_server(payload):
    monitor_server = 'http://20.163.29.125:8083/post'
    resp = requests.post(monitor_server, json=payload)

    if resp.status_code == 200:
        print("sent to monitoring app successfully!")
    else:
        print(f"Failed , Monitoring server returned: {resp.json()}")

def form_monitoring_payload(response, filename):
    bounding_boxes, confidence, classes = [], [], []
    for box in response.get('result').get('bounding_boxes'):
        bounding_boxes.append([box.get('x'), box.get('y'), box.get('x') + box.get('width'), box.get('y') + box.get('height')])
        confidence.append(box.get('value'))
        classes.append(box.get('label'))

    print("predicted classes by our model: ", ','.join(classes))

    payload = {
        "image_url": "https://arziedemo.blob.core.windows.net/retrain-images/" + filename,
        "bounding_boxes": bounding_boxes,
        "confidence": confidence,
        "classes": classes,
        "timestamp": int(time.time())
    }
    return payload

def post_process_image(resized_filename, response, prev_image):
    prev_image = resized_filename
    if response:
        str_resp = format_response(response)
        if len(response.get('result').get('bounding_boxes')) > 0 and str_resp == 'No objects detected with confidence more than 0.85':
            ls = resized_filename.split('/')
            filename = ls[-1]
            upload_azure_blob(container_client, filename, resized_filename)
            payload = form_monitoring_payload(response, filename)
            send_to_monitor_server(payload)
        log_response(str_resp)
    return prev_image

def send_file_for_inference(file_path):
    server_url = 'http://10.1.0.128:1337/api/image'
    try:
        # Open the file in binary mode for reading
        with open(file_path, 'rb') as file:
            # Prepare the HTTP POST request with the file
            files = {'file': file}  # 'file' is the name of the form field expected by the server
            response = requests.post(server_url, files=files)

            # Check the response status
            if response.status_code == 200:
                print("successfull inference!")
                print(response.text)
                return json.loads(response.text)
            else:
                print(f"Failed to send file. Server returned status code: {response.status_code}")
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")
    
    return None

def compare_images_ssim(image1_path, image2_path):
    # Load images
    image1 = cv2.imread(image1_path, cv2.IMREAD_GRAYSCALE)
    image2 = cv2.imread(image2_path, cv2.IMREAD_GRAYSCALE)

    if image1 is None or image2 is None:
        print("Error: Could not read the images.")
        return False

    # Calculate SSIM
    similarity_index = ssim(image1, image2)

    # Define a threshold for SSIM (adjust this based on your requirement)
    ssim_threshold = 0.95  # Adjust this threshold as needed

    # Compare SSIM with threshold
    if similarity_index > ssim_threshold:
        print("SSIM: Images are similar.")
        return True
    else:
        print("SSIM: Images are different.")
        return False

def compare_images_mse(image1_path, image2_path):
    # Load images
    image1 = cv2.imread(image1_path)
    image2 = cv2.imread(image2_path)

    if image1 is None or image2 is None:
        print("Error: Could not read the images.")
        return False

    # Ensure images have the same dimensions
    if image1.shape != image2.shape:
        print("Error: Images have different dimensions.")
        return False

    # Calculate MSE
    mse = ((image1 - image2) ** 2).mean()

    # Define a threshold for MSE (adjust this based on your requirement)
    mse_threshold = 100  # Adjust this threshold as needed

    # Compare MSE with threshold
    if mse < mse_threshold:
        print("MSE: Images are similar.")
        return True
    else:
        print("MSE: Images are different.")
        return False

def resize_and_save(input_path, output_path, new_width, new_height):
    # Read the image from input path
    image = cv2.imread(input_path)

    if image is None:
        print(f"Error: Unable to read image from {input_path}")
        return

    # Resize the image
    resized_image = cv2.resize(image, (new_width, new_height))


    # Save the resized image to the specified output path
    cv2.imwrite(output_path, resized_image)

    print(f"Resized image saved at {output_path}")

def is_blurred(frame, threshold=100):
    """
    Check if the frame is blurred using the Laplacian variance method.
    
    Parameters:
    - frame: The frame to check.
    - threshold: The threshold for the variance of the Laplacian. Frames with a variance below this value are considered blurred.
    
    Returns:
    - True if the frame is blurred, False otherwise.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian_var < threshold

def pre_process_image(frame, frame_count, prev_image):
    ts = int(time.time())
    filename = '%s/frame_%d_%d.jpg' % (output_dir, frame_count, ts)
    resized_filename = '%s/frame_%d_%d_resized.jpg' % (output_dir, frame_count, ts)

    if is_blurred(frame):
        print("Frame is blurred. Skipping...")
        return True, resized_filename, frame_count, prev_image

    cv2.imwrite(filename, frame)
    resize_and_save(filename, resized_filename, new_width, new_height)
    if prev_image:
        images_similar = compare_images_mse(prev_image, resized_filename)
        if images_similar:
            images_similar = compare_images_ssim(prev_image, resized_filename)
            prev_image = ''
            if images_similar:
                frame_count += 1
                prev_image = resized_filename
                return True , resized_filename, frame_count, prev_image
    return False, resized_filename, frame_count, prev_image

def check_break_condition(frame_count):
    if frame_count == 400:
        print("Exiting: frame limit exhausted...")
        return True
    return False

def capture_video_frames(video_capture, fps):
    frame_count = 0
    prev_image = ''
    while True:
        if check_break_condition(frame_count):
            break
        ret, frame = video_capture.read()
        if not ret:
            break

        # Extract frames at every fifth of a second
        if frame_count % int(fps/5) == 0:
            is_similar, resized_filename, frame_count, prev_image = pre_process_image(frame, frame_count, prev_image)
            if is_similar:
                continue
            response = send_file_for_inference(resized_filename)
            prev_image = post_process_image(resized_filename, response, prev_image)

        frame_count += 1

def initialize_webcam():
    # Open the first webcam device
    try:
        video_capture = cv2.VideoCapture(0)
    except Exception as e:
        print("Error opening video device: %s" % e)

    # Check if the webcam is opened correctly
    if not video_capture.isOpened():
        print("Error: Could not open webcam.")
        exit()

    # Open the video file
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    return video_capture, fps

def initialize_azure():
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    container_client = blob_service_client.get_container_client(container_name)
    return container_client

container_client = initialize_azure()
video_capture, fps = initialize_webcam()
capture_video_frames(video_capture, fps)
cleanup_resources(video_capture)