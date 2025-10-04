from fastapi import FastAPI, File, UploadFile
from typing import Annotated
import utils
app = FastAPI()

@app.get("/")
def home():
    return {"message": "Hello App"}

@app.post("/predict")
async def predict(file: Annotated[UploadFile, File(...)]):
    return utils.get_result(image_file=file)
