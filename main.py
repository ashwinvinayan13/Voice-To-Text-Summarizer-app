from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from transformers import pipeline
from fastapi import UploadFile, File

asr = pipeline("automatic-speech-recognition", model="openai/whisper-base")
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")


app = FastAPI()

class TextInput(BaseModel):
    summary: str

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def home():
    return FileResponse("static/index.html")

@app.post("/summary")
async def summarize(data: TextInput):
    result = summarizer(data.summary.strip(), min_length=10, max_length=60)
    return {"summary": result[0]['summary_text']}

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    try:
        file_location = f"temp_{file.filename}"
        with open(file_location, "wb") as f:
            f.write(await file.read())

        result = asr(
            file_location,
            return_timestamps=False,
            chunk_length_s = 30,
            generate_kwargs={"language": "en"}
        )

        return {"text": result["text"]}

    except Exception as e:
        return {"error": str(e)}

