### 1. Imports and class names setup ###
import gradio as gr
import os
import torch

from model import create_vit_model
from timeit import default_timer as timer
from typing import Tuple, Dict

# Setup class names
class_names = ["Asagi", "Bekko", "Doitsu Hariwake", "Goromo", "Hikarimoyo", "Kawarimoyo", "Kinginrin",
               "Kohaku", "Matsuba", "Ogon", "Sanke", "Showa", "Shusui", "Tancho", "Utsurimono"]

### 2. Model and transforms preparation ###

# Create vit model
vit, vit_transforms = create_vit_model(
    num_classes=3, # len(class_names) would also work
)

# Load saved weights
vit.load_state_dict(
    torch.load(
        f="09_pretrained_vit_feature_extractor_15_kelas_koi.pth",
        map_location=torch.device("cpu"),  # load to CPU
    )
)

### 3. Predict function ###

# Create predict function
def predict(img) -> Tuple[Dict, float]:
    """Transforms and performs a prediction on img and returns prediction and time taken.
    """
    # Start the timer
    start_time = timer()

    # Transform the target image and add a batch dimension
    img = vit_transforms(img).unsqueeze(0)

    # Put model into evaluation mode and turn on inference mode
    vit.eval()
    with torch.inference_mode():
        # Pass the transformed image through the model and turn the prediction logits into prediction probabilities
        pred_probs = torch.softmax(vit(img), dim=1)

    # Create a prediction label and prediction probability dictionary for each prediction class (this is the required format for Gradio's output parameter)
    pred_labels_and_probs = {class_names[i]: float(pred_probs[0][i]) for i in range(len(class_names))}

    # Calculate the prediction time
    pred_time = round(timer() - start_time, 5)

    # Return the prediction dictionary and prediction time
    return pred_labels_and_probs, pred_time

### 4. Gradio app ###

# Create title, and description strings
title = "Demo Gradio (Mini) untuk Identifikasi Citra Ikan Koi"
description = "An Vit-B/16 feature extractor computer vision model to classify images of koi into [15 different classes]."

# Create examples list from "examples/" directory
example_list = [["examples/" + example] for example in os.listdir("examples")]

# Create the Gradio demo
demo = gr.Interface(fn=predict, # mapping function from input to output
                    inputs=gr.Image(type="pil"), # what are the inputs?
                    outputs=[gr.Label(num_top_classes=3, label="Predictions"), # what are the outputs?
                             gr.Number(label="Prediction time (s)")], # our fn has two outputs, therefore we have two outputs
                    # Create examples list from "examples/" directory
                    examples=example_list,
                    title=title,
                    description=description)

# Launch the demo!
demo.launch()
