# ocr-transformer-usage

This simple code opens a web page to take screenshot. Clicking on takescreen shot , waits for 5 secs then grabs the current screenshot calls a GCP(or external) hosted Omniparser and grabs the output of various detection.

For deploying omniparser , i did following
1. Grab a VM (CPU or GPU) - I did it with CPU
2. git clone https://github.com/microsoft/OmniParser.git
3. Download the model files download.sh (took inspiration from https://mer.vin/2024/10/omni-parser/) and modified few things
4. Modified weights/convert_safetensor_to_pt.py a bit
5. Created main.py
6. In omniparser.py fixed the list conversion & deleted config in omniparser
7. Opened up GCP Firewall to allow only from my ip and open up 8080 port
8. Do following to run
   - pip install -r requirements.txt
   - pip install fastapi uvicorn torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
   - uvicorn main:app --host 0.0.0.0 --port 8080
   
![Screenshot 2024-12-31 at 10 06 44 AM](https://github.com/user-attachments/assets/3c8ea2c7-7887-4f13-a5e6-0e1979c4f09c)

download.sh
```
#!/bin/bash

# Base URL for downloading model files
BASE_URL="https://huggingface.co/microsoft/OmniParser/resolve/main"

# Define folder structure and create folders
mkdir -p weights/icon_detect
mkdir -p weights/icon_caption_florence
mkdir -p weights/icon_caption_blip2
mkdir -p weights/icon_detect_v1_5
# Declare an associative array of required files with paths
declare -A model_files=(
  ["weights/icon_detect/model.safetensors"]="$BASE_URL/icon_detect/model.safetensors"
  ["weights/icon_detect/model.yaml"]="$BASE_URL/icon_detect/model.yaml"
  ["weights/icon_caption_florence/model.safetensors"]="$BASE_URL/icon_caption_florence/model.safetensors"
  ["weights/icon_caption_florence/config.json"]="$BASE_URL/icon_caption_florence/config.json"
  ["weights/icon_caption_blip2/model.safetensors"]="$BASE_URL/icon_caption_blip2/model.safetensors"
  ["weights/icon_caption_blip2/config.json"]="$BASE_URL/icon_caption_blip2/config.json"
  ["weights/icon_detect_v1_5/model_v1_5.pt"]="https://huggingface.co/microsoft/OmniParser/tree/main/icon_detect_v1_5/model_v1_5.pt"
)

# Download each file into its specified directory
for file_path in "${!model_files[@]}"; do
  wget -O "$file_path" "${model_files[$file_path]}"
done

echo "All required model and configuration files downloaded and organised."

# Run the conversion script if necessary files are present
if [ -f "weights/icon_detect/model.safetensors" ] && [ -f "weights/icon_detect/model.yaml" ]; then
  python weights/convert_safetensor_to_pt.py
  echo "Conversion to best.pt completed."
else
  echo "Error: Required files for conversion not found."
fi
```

weights/convert_safetensor_to_pt.py
```
import torch
from ultralytics.nn.tasks import DetectionModel
from safetensors.torch import load_file
import argparse
import yaml
import os

# accept args to specify v1
parser = argparse.ArgumentParser(description='add weight directory')
parser.add_argument('--weights_dir', type=str, required=True, help='Specify the path to the safetensor file', default='weights/icon_detect')
#args = parser.parse_args()

weights_dir='weights/icon_detect'
tensor_dict = load_file(os.path.join(weights_dir, "model.safetensors"))
model = DetectionModel(os.path.join(weights_dir, "model.yaml"))

model.load_state_dict(tensor_dict)
save_dict = {'model':model}

#with open(os.path.join(weights_dir, "train_args.yaml"), 'r') as file:
#    train_args = yaml.safe_load(file)
#save_dict.update(train_args)
torch.save(save_dict, os.path.join(weights_dir, "best.pt"))
```
main.py
```
from fastapi import FastAPI, File, UploadFile, HTTPException
from omniparser import Omniparser  # Adjust based on the OmniParser library
from typing import Dict
import tempfile
import os
import numpy as np

app = FastAPI()
config = {
    'som_model_path': 'weights/icon_detect/best.pt',
    'device': 'cpu',
    'caption_model_path': 'Salesforce/blip2-opt-2.7b',
    'draw_bbox_config': {
        'text_scale': 0.8,
        'text_thickness': 2,
        'text_padding': 3,
        'thickness': 3,
    },
    'BOX_TRESHOLD': 0.05
}
parser = Omniparser(config)  # Initialize OmniParser (adjust if necessary)

def convert_numpy_types(obj):
    """
    Recursively convert NumPy types to Python native types.
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()  # Convert arrays to lists
    elif isinstance(obj, np.generic):
        return obj.item()  # Convert scalars to Python scalars
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(value) for value in obj]
    return obj

@app.post("/parse-screenshot")
async def parse_document(file: UploadFile = File(...)) -> Dict:
    """
    Parse a document file using OmniParser.
    """
    print("Received request")
    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded")
    
    try:
        # Read the uploaded file and parse it using OmniParser
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[-1]) as temp_file:
            temp_file.write(await file.read())
            temp_file_path = temp_file.name
        
        # Pass the temporary file path to the parser
        parsed_data = parser.parse(temp_file_path)
        clean_data = convert_numpy_types(parsed_data)

        # Clean up the temporary file after parsing
        os.remove(temp_file_path)
        return {"parsed_data": clean_data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```
omniparser.py (partial code fix)
```
return_list = [
            {
                'shape': {
                    'x': coord[0], 
                    'y': coord[1], 
                    'width': coord[2], 
                    'height': coord[3]
                },
                'text': parsed_content_list[i]['content'],  # Access 'content' directly
                'type': parsed_content_list[i]['type'],    # Access 'type' directly
            }
            for i, (k, coord) in enumerate(label_coordinates.items()) if i < len(parsed_content_list)
        ]
```
