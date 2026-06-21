import logging

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.config.config import BASE_DIR
from app.routers import agents

logging.basicConfig(level=logging.INFO)

app = FastAPI(title="Smart Prompt Router")

STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = STATIC_DIR / "templates"

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
app.include_router(agents.router)

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    # request=request, name=... works across both old and new Starlette versions
    return templates.TemplateResponse(request=request, name="index.html")


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    # Run from the project root (the folder that contains "app/") with:
    #   python -m app.main
    # or:
    #   uvicorn app.main:app --reload
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
