import base64
import io
import numpy as np
import pandas as pd
import soundfile as sf
import librosa
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class AudioRequest(BaseModel):
    audio_id: str
    audio_base64: str


def decode_audio(audio_base64: str):
    """Decode base64 -> numpy array directly in memory (no temp file, no ffmpeg)."""
    if "," in audio_base64 and audio_base64.strip().startswith("data:"):
        audio_base64 = audio_base64.split(",", 1)[1]

    audio_bytes = base64.b64decode(audio_base64)
    buf = io.BytesIO(audio_bytes)

    # soundfile reads WAV/FLAC natively and fast, no subprocess needed
    y, sr = sf.read(buf, dtype="float32", always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)  # mono
    return y, sr


def extract_feature_dataframe(y: np.ndarray, sr: int, hop_length: int = 1024) -> pd.DataFrame:
    """Lighter feature set: bigger hop_length + fewer MFCCs = far fewer FFT calls."""
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    zcr = librosa.feature.zero_crossing_rate(y, hop_length=hop_length)[0]
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)[0]
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=hop_length)[0]
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=hop_length)[0]
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=8, hop_length=hop_length)  # fewer coeffs

    n_frames = min(len(rms), len(zcr), len(centroid), len(bandwidth), len(rolloff), mfcc.shape[1])

    data = {
        "rms_energy": rms[:n_frames],
        "zero_crossing_rate": zcr[:n_frames],
        "spectral_centroid": centroid[:n_frames],
        "spectral_bandwidth": bandwidth[:n_frames],
        "spectral_rolloff": rolloff[:n_frames],
    }
    for i in range(8):
        data[f"mfcc_{i+1}"] = mfcc[i][:n_frames]

    return pd.DataFrame(data)


def dataframe_stats(df: pd.DataFrame) -> dict:
    numeric_df = df.select_dtypes(include=[np.number])

    def series_mode(s: pd.Series):
        m = s.mode()
        return float(m.iloc[0]) if not m.empty else None

    mean = numeric_df.mean()
    std = numeric_df.std()
    variance = numeric_df.var()
    minimum = numeric_df.min()
    maximum = numeric_df.max()
    median = numeric_df.median()
    mode = {col: series_mode(numeric_df[col]) for col in numeric_df.columns}
    rng = maximum - minimum

    return {
        "rows": int(numeric_df.shape[0]),
        "columns": list(numeric_df.columns),
        "mean": {k: float(v) for k, v in mean.to_dict().items()},
        "std": {k: float(v) for k, v in std.to_dict().items()},
        "variance": {k: float(v) for k, v in variance.to_dict().items()},
        "min": {k: float(v) for k, v in minimum.to_dict().items()},
        "max": {k: float(v) for k, v in maximum.to_dict().items()},
        "median": {k: float(v) for k, v in median.to_dict().items()},
        "mode": mode,
        "range": {k: float(v) for k, v in rng.to_dict().items()},
        "allowed_values": {},
        "value_range": {
            col: [float(minimum[col]), float(maximum[col])] for col in numeric_df.columns
        },
        "correlation": numeric_df.corr().fillna(0).values.tolist(),
    }


@app.get("/")
def root():
    return {"status": "ok"}


@app.post("/analyze")
def analyze(req: AudioRequest):
    try:
        y, sr = decode_audio(req.audio_base64)
        df = extract_feature_dataframe(y, sr)
        if df.empty:
            raise HTTPException(status_code=400, detail="No audio frames extracted")
        return dataframe_stats(df)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to process audio: {e}")
