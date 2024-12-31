from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import pyautogui
import time
import requests
import os
import tempfile
import base64
from dotenv import load_dotenv

load_dotenv()

# GCP Endpoint Configuration
GCP_PARSE_ENDPOINT = os.getenv("GCP_IP") + "/parse-screenshot"
print (f"GCP_PARSE_ENDPOINT {GCP_PARSE_ENDPOINT}")

app = FastAPI()

# Directory for templates
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
templates = Jinja2Templates(directory=TEMPLATES_DIR)

# Serve static files if needed
#app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """
    Render the homepage with the Take Screenshot button.
    """
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/take-screenshot")
async def take_screenshot():
    """
    Wait for 5 minutes, take a screenshot, and send it to the GCP endpoint.
    """
    try:
        time.sleep(5)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
            screenshot = pyautogui.screenshot()
            screenshot.save(temp_file.name)
            temp_file_path = temp_file.name

        # Send the screenshot to the GCP endpoint
        with open(temp_file_path, 'rb') as file:
            response = requests.post(GCP_PARSE_ENDPOINT, files={'file': file})

        # Cleanup the temporary file
        os.remove(temp_file_path)


        # Return the GCP response
        if response.status_code == 200:
            print('response is successful', response.json())
            gcp_response = response.json().get('parsed_data')
            labeled_img = gcp_response.get("labeled_img")
            labels = gcp_response.get("labels")
            print("labels", labels)
            print("labeled_img", labeled_img)

            # Decode the labeled image
            decoded_image = base64.b64decode(labeled_img)
            print("Decoded image")
            labeled_img_base64 = base64.b64encode(decoded_image).decode('utf-8')
            print("Decoded image 64")

            # Return image and labels to the frontend
            return {"status": "success", "labeled_img": labeled_img_base64, "labels": labels}
        else:
            print('response is failure', response.status_code, response.text)
            return JSONResponse(content={"status": "error", "detail": response.text}, status_code=response.status_code)

    except Exception as e:
        return JSONResponse(content={"status": "error", "detail": str(e)}, status_code=500)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=5000)
