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


responses = []


@asynccontextmanager
async def lifespan(app: FastAPI):
    global responses
    log.info('starting lifespan function')
    # Load the ML model
    responses = parse_hn()
    log.info(responses)
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
    my_list = ["Item 1", "Item 2", "Item 3"]
    context = {
        'items': my_list,
        'myid': 'hello',
        'request': request,
    }

    log.info(context)
    return templates.TemplateResponse(
        name="index.html",
        context=context,
    )
