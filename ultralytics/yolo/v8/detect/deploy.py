from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import shutil
import subprocess
from pathlib import Path
import os

app = FastAPI()

UPLOAD_DIR = Path("static/uploads")
PROCESSED_DIR = Path("static/processed")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def upload_video_form(request: Request):
    return templates.TemplateResponse("upload.html", {"request": request})

@app.post("/", response_class=HTMLResponse)
async def upload_video(request: Request, file: UploadFile = File(...)):
    file_path = UPLOAD_DIR / file.filename
    
    with file_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    result = subprocess.run(
        ["python", "predict.py", str(file_path)],
        capture_output=True,
        text=True
    )

    if result.returncode == 0:
        processed_file_name = f"{file.filename}"
        processed_file_path = PROCESSED_DIR / processed_file_name
        # Assuming the predict_test.py script outputs the processed video in the PROCESSED_DIR
        if os.path.exists(processed_file_path):
            return templates.TemplateResponse(
                "upload.html",
                {"request": request, "processed_video_url": f"/static/processed/{processed_file_name}"}
            )
        else:
            return JSONResponse(content={"error": "Processed video not found"}, status_code=500)
    else:
        return JSONResponse(content={"error": result.stderr}, status_code=500)

@app.get("/processed-videos/", response_class=HTMLResponse)
async def processed_videos(request: Request):
    video_files = [{"name": file.name, "url": f"/static/processed/{file.name}"} for file in PROCESSED_DIR.glob("*") if file.suffix == ".mp4"]
    return templates.TemplateResponse("processed_videos.html", {"request": request, "video_files": video_files})
