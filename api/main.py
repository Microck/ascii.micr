from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import base64
import io
import os
import sys
import shutil
from typing import Optional

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'temp_gradscii'))

from train import (
    train,
    create_char_bitmaps,
    load_target_image,
    DEVICE,
    CHARS,
    NUM_CHARS
)

app = FastAPI(title="gradscii-art API")

origins = os.getenv("ALLOWED_ORIGINS", "*").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)

class GenerationParams(BaseModel):
    iterations: int = Field(10000, ge=100, le=20000)
    lr: float = Field(0.01, ge=0.0001, le=1.0)
    diversity_weight: float = Field(0.01, ge=0.0, le=1.0)
    temp_start: float = Field(1.0, ge=0.1, le=10.0)
    temp_end: float = Field(0.01, ge=0.0001, le=1.0)
    optimize_alignment: bool = False
    dark_mode: bool = False
    encoding: str = "cp437"
    row_gap: int = Field(6, ge=0, le=50)

class GenerationRequest(BaseModel):
    image: str
    params: GenerationParams

class StepResponse(BaseModel):
    iteration: int
    image: str

class GenerationResponse(BaseModel):
    png: str
    text: str
    steps: Optional[list[StepResponse]] = None

os.makedirs("tmp", exist_ok=True)

@app.get("/")
async def root():
    return {"message": "gradscii-art API - POST /generate to create ASCII art"}

@app.post("/generate")
async def generate(request: GenerationRequest):
    tmp_dir = "tmp"
    os.makedirs(tmp_dir, exist_ok=True)
    image_path = os.path.join(tmp_dir, "input.png")

    try:
        if len(request.image) > 10 * 1024 * 1024 * 1.33:
            raise HTTPException(status_code=413, detail="Image too large (Max 10MB)")

        image_data = base64.b64decode(request.image)
        image_bytes = io.BytesIO(image_data)
        
        try:
            from PIL import Image
            img = Image.open(image_bytes)
            img.verify()
            image_bytes.seek(0)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid image format")

        with open(image_path, "wb") as f:
            f.write(image_bytes.getbuffer().tobytes())

        global CHARS, NUM_CHARS
        if request.params.encoding == "cp437":
            CHARS = "".join([chr(i) for i in range(128)])
            BANNED_CHARS = ['`', '\\']
            CHARS = "".join([c for c in CHARS if c not in BANNED_CHARS])
        else:
            CHARS = " !\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~"

        NUM_CHARS = len(CHARS)

        char_bitmaps = create_char_bitmaps()
        target_image = load_target_image(image_path, keep_rgb=False)
        target_image = target_image.unsqueeze(0)

        steps = []

        class ProgressCallback:
            def __init__(self):
                self.steps = steps

            def __call__(self, iteration, current_lr, temp):
                if iteration % 100 == 0:
                    from PIL import Image
                    from train import save_result

                    temp_path = os.path.join(tmp_dir, f"step_{iteration}.png")
                    save_result(
                        None,
                        char_bitmaps,
                        output_path=temp_path,
                        temperature=temp,
                        target_image=None,
                        warp_params=None,
                        dark_mode=request.params.dark_mode
                    )

                    with open(temp_path, "rb") as f:
                        step_b64 = base64.b64encode(f.read()).decode()
                        self.steps.append(StepResponse(
                            iteration=iteration,
                            image=step_b64
                        ))

        callback = ProgressCallback()

        logits, alignment_params = train(
            target_image,
            char_bitmaps,
            num_iterations=request.params.iterations,
            lr=request.params.lr,
            diversity_weight=request.params.diversity_weight,
            use_gumbel=True,
            temp_start=request.params.temp_start,
            temp_end=request.params.temp_end,
            protect_whitespace=True,
            multiscale_weight=0.0,
            optimize_alignment=request.params.optimize_alignment,
            alignment_lr=0.01,
            warp_reg_weight=0.01,
            dark_mode=request.params.dark_mode,
            prev_alignment_params=None
        )

        output_path = os.path.join(tmp_dir, "output.png")
        from train import save_result
        save_result(
            logits,
            char_bitmaps,
            output_path=output_path,
            text_path=os.path.join(tmp_dir, "output.txt"),
            utf8_path=os.path.join(tmp_dir, "output.utf8.txt"),
            temperature=0.01,
            target_image=None,
            warp_params=alignment_params,
            dark_mode=request.params.dark_mode
        )

        with open(output_path, "rb") as f:
            png_b64 = base64.b64encode(f.read()).decode()

        with open(os.path.join(tmp_dir, "output.txt"), "r", encoding=request.params.encoding) as f:
            text_content = f.read()

        shutil.rmtree(tmp_dir)

        return GenerationResponse(
            png=png_b64,
            text=text_content,
            steps=steps[:10]
        )

    except Exception as e:
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
