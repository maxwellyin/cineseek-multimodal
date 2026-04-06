from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

try:
    from . import network
except ImportError:
    import network


APP_DIR = Path(__file__).resolve().parent
app = FastAPI(title="CineSeek-MM Demo")
templates = Jinja2Templates(directory=str(APP_DIR / "templates"))


@app.get("/", response_class=HTMLResponse)
@app.get("/search", response_class=HTMLResponse)
async def search_page(request: Request):
    return templates.TemplateResponse(
        request,
        "home.html",
        {
            "query": "",
            "mode": "text",
            "image_weight": 0.05,
            "results": None,
            "error": None,
        },
    )


@app.post("/search", response_class=HTMLResponse)
async def search_submit(
    request: Request,
    query: str = Form(default=""),
    mode: str = Form(default="text"),
    image_weight: float = Form(default=0.05),
    image: UploadFile | None = File(default=None),
):
    temp_path = None
    error = None
    results = None
    try:
        if image is not None and image.filename:
            suffix = Path(image.filename).suffix or ".jpg"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                shutil.copyfileobj(image.file, tmp)
                temp_path = tmp.name
        results = network.search(query=query.strip(), image_path=temp_path, mode=mode, image_weight=image_weight)
    except Exception as exc:
        error = str(exc)
    finally:
        if temp_path:
            Path(temp_path).unlink(missing_ok=True)

    return templates.TemplateResponse(
        request,
        "home.html",
        {
            "query": query,
            "mode": mode,
            "image_weight": image_weight,
            "results": results,
            "error": error,
        },
    )
