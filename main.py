import os
from io import BytesIO
import base64
from PIL import Image
import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from model import siamese, encode_image

THRESHOLD = 0.7

base_model = siamese(input_shape=(50, 50, 3))
base_model.load_weights("siamese.h5")

conv_model = base_model.get_layer(index=2)
print("Finished loading model")
print("Creating embeddings for existing users")

users = dict()

for user in os.listdir("images"):
    with Image.open(f"images/{user}") as im:
        username = user.split(".")[0]
        users[username] = encode_image(conv_model, im)

print(users.keys())

# Create Server
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def main(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/verify")
async def verify(request: Request, body: dict):
    img_base64 = body["img"]
    img = Image.open(BytesIO(base64.b64decode(img_base64)))
    img.save("test.jpg")
    return templates.TemplateResponse("index.html", {"request": request, "err": "No se encontro niguna coincidencia. Acceso denegado"})