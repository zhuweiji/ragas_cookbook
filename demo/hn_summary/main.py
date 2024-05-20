import logging
from calendar import c
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from demo.hn_summary.feedparser_hn import parse_hn
from demo.hn_summary.project_paths import project_root

logging.basicConfig(
    format='%(name)s-%(levelname)s|%(lineno)d:  %(message)s', level=logging.INFO)
log = logging.getLogger(__name__)


hn_tech = []


@asynccontextmanager
async def lifespan(app: FastAPI):
    global hn_tech
    log.info('starting lifespan function')
    # Load the ML model
    hn_tech = parse_hn()
    log.info(hn_tech)
    yield

app = FastAPI(
    title="HackerNews Tech Summary",
    lifespan=lifespan
)

origins = [
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.mount(
    "/static", StaticFiles(directory=f"{project_root}/static"), name="static")
templates = Jinja2Templates(directory=f"{project_root}/templates")


@app.get("/", response_class=HTMLResponse)
async def render_dashboard(request: Request):
    context = {
        'request': request,
        'hn_tech': hn_tech
    }

    log.info(context)
    return templates.TemplateResponse(
        name="index.html",
        context=context,

    )
