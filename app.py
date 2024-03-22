from fastapi import FastAPI, HTTPException
import uvicorn
from fastapi.responses import Response, JSONResponse
from starlette.responses import RedirectResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from prediction_pipeline import PredictionPipeline

text: str = "News Summarization"
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["POST"],
    allow_headers=["*"],
)

@app.get("/", tags=["authentication"])
async def index():
    return RedirectResponse(url="/docs")

@app.post("/predict")
async def predict(text):
    try:
        obj = PredictionPipeline()
        summary = obj.main(text)
        return JSONResponse(content={"status": "Prediction Successful", "summary": summary})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction Failed: {e}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
