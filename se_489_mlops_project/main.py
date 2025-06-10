from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from se_489_mlops_project.predict import process_sudoku
import shutil
import os
import uuid

app = FastAPI()

@app.post("/predict/")
async def predict_image(file: UploadFile = File(...)):
    temp_id = str(uuid.uuid4())
    input_path = f"temp_{temp_id}.jpg"

    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        result = process_sudoku(input_path)
        os.remove(input_path)
        return JSONResponse(content={"result": result})
    except Exception as e:
        os.remove(input_path)
        return JSONResponse(status_code=500, content={"error": str(e)})