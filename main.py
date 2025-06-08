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
    audio_bytes = await file.read()
    result = asr(audio_bytes, return_timestamps=False, generate_kwargs={"language":"en"}, chunk_length_s=30)
    return {"transcription": result["text"]}
