from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import uuid
import os
import random

from ultralytics import YOLO

from arize.api import Client
from arize.utils.types import Environments, ModelTypes, ObjectDetectionLabel, Embedding
from torchvision import transforms
import torch
from PIL import Image

model = YOLO('yolov8n.pt')

app = FastAPI()

arize_client = Client(
    space_key="55d57ee",
    api_key="b7a6d45b568117b4acf",
)

node_list = ['edge-node-1', 'edge-node-2', 'edge-node-3', 'edge-node-4', 'edge-node-5', 'edge-node-6', 'edge-node-7', 'edge-node-8', 'edge-node-9', 'edge-node-10', 'edge-node-11']
version_list = ['v1', 'v2']

# Define a data model for the request body
class Event(BaseModel):
    image_url: str
    bounding_boxes: List[List[float]]
    confidence: List[float]
    classes: List[str]
    timestamp: int

def send_metrics_to_arize(metrics: dict):

    print("Sending metrics to Arize for image: ", metrics.get('url'))

    embedding_features = {
        'image_embedding': Embedding(
            vector=metrics.get('image_vector'),
            link_to_data=metrics.get('url'),
        )
    }

    object_detection_prediction = ObjectDetectionLabel(
        metrics.get('prediction_bboxes'), 
        metrics.get('prediction_categories'),
        metrics.get('prediction_scores')
    )

    object_detection_actual = ObjectDetectionLabel(
        metrics.get('actual_bboxes'),
        metrics.get('actual_categories'),
    )

    try:
        node = random.choice(node_list)
        version = random.choice(version_list)
        # Log inferences to Arize
        response = arize_client.log(
            model_id="watch-mobile-image-detection",
            model_version=version,
            environment=Environments.PRODUCTION,
            model_type=ModelTypes.OBJECT_DETECTION,
            prediction_id=metrics.get('prediction_id'),
            prediction_label=object_detection_prediction,
            actual_label=object_detection_actual,
            embedding_features=embedding_features,
            prediction_timestamp=metrics.get('prediction_ts'),
            tags={"edge_node": node},
            features=metrics.get('features')
        )
        res = response.result()
        if res.status_code == 200:
            print('Success sending Prediction!')
        else:
            print(f'arize failed with response code {res.status_code}, {res.text}')
    except Exception as e:
        print(f"Error: {e}")

def get_reference_prediction(image_url: str):
    filename = image_url.split('/')[-1]
    results = model(image_url)

    # Extract bounding boxes, classes, and confidence scores
    bounding_boxes = results[0].boxes.xyxy.tolist()
    classes = [results[0].names[cls] for cls in results[0].boxes.cls.tolist()]
    confidence = results[0].boxes.conf.tolist()

    image_vector = get_image_vector(filename)

    # Prepare the response
    response = {
        "image_url": image_url,
        "bounding_box": bounding_boxes,
        "confidence": confidence,
        "classes": classes,
        "image_vector": image_vector
    }

    # Return the response
    return response

def get_features_from_layer(model, image_tensor, layer_idx=-2):
    """
    Extract features from a specific layer.
    
    Parameters:
    - model: The loaded YOLOv5 model.
    - image_tensor: The input image tensor with shape [C, H, W] and batch dimension added.
    - layer_idx: Index of the layer from which to extract features. Default is -2, the second to last layer.
    
    Returns:
    - features: The extracted features from the specified layer.
    """
    # Ensure model is in evaluation mode
    #model.eval()
    
    # Hook to capture the output of the specified layer
    features = []
    def hook_fn(module, input, output):
        features.append(output)
    
    # Register the hook to the desired layer
    handle = model.model.model[layer_idx].register_forward_hook(hook_fn)
    
    # Perform a forward pass to get the features
    with torch.no_grad():
        _ = model(image_tensor.unsqueeze(0))  # Add batch dimension if not present
    
    # Remove the hook
    handle.remove()
    
    # Return the features
    return features[0]

def get_image_vector(filename: str):
    image_1 = Image.open(filename)
    transform = transforms.Compose([
                transforms.Resize((640, 640)),
                transforms.ToTensor()
            ])
    image_tensor = transform(image_1)
    model = YOLO('yolov8n.pt')

    res = get_features_from_layer(model, image_tensor, layer_idx=-2) 

    image_vector = res.flatten().tolist()
    if len(image_vector) > 20000:
        image_vector = image_vector[:20000]
    os.remove(filename)
    return image_vector

def form_payload_arize(predicted_data: Event, actuals: dict, features: dict):

    metrics = {
        'prediction_id': uuid.uuid4(),
        'prediction_ts': predicted_data.timestamp,
        'prediction_bboxes': predicted_data.bounding_boxes,
        'actual_bboxes': actuals.get('bounding_box'),
        'prediction_categories': predicted_data.classes,
        'actual_categories': actuals.get('classes'), 
        'prediction_scores': predicted_data.confidence,
        "image_vector": actuals.get('image_vector'),
        "url": predicted_data.image_url,
        "features": features,
    }
    return metrics

def preprocess_actuals(actuals: dict):
    temp = actuals.copy()
    temp['classes'] = []
    temp['bounding_box'] = []
    temp['confidence'] = []
    for k, cls in enumerate(actuals.get('classes')):
        if actuals.get('confidence')[k] > 0.2:
            if cls == "cup" or cls == "cell phone" or cls == "clock":
                cl = cls
                if cls == "clock":
                    cl = "watch"
                temp['classes'].append(cl)
                temp['bounding_box'].append(actuals.get('bounding_box')[k])
                temp['confidence'].append(actuals.get('confidence')[k])
    return temp

def get_image_features(predicted_data: Event):
    """
    Extract object-level features: object size, aspect ratio
    """
    object_sizes = []
    object_aspect_ratios = []

    for box in predicted_data.bounding_boxes:
        size = (box[2] - box[0]) * (box[3] - box[1])
        aspect_ratio = (box[2] - box[0]) / (box[3] - box[1])
        object_sizes.append(size)
        object_aspect_ratios.append(aspect_ratio)

    return {
        'object_sizes': object_sizes,
        'object_aspect_ratios': object_aspect_ratios,
    }

def processEvent(predicted_data: Event):
    actuals = get_reference_prediction(predicted_data.image_url)
    actuals = preprocess_actuals(actuals)
    if actuals.get('bounding_box') == []:
        return {"error": "No actuals objects detected" }
    print("ground truth classes: ", ','.join(actuals.get('classes')))
    features = get_image_features(predicted_data)
    metrics = form_payload_arize(predicted_data, actuals, features)
    send_metrics_to_arize(metrics)
    return metrics

@app.post("/post")
async def post_endpoint(data: Event):
    # Process the data (for demonstration, we just print it)
    print("Received data:", data)

    resp = processEvent(data)

    # Respond with a success message
    return {"message": "Data received successfully", "arize_metrics": resp}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8083)
