import json
import logging
import ssl
import time
from typing import List

import feedparser
from feedparser.util import FeedParserDict
from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import BaseTool, StructuredTool, tool
from langchain_community.document_loaders import AsyncHtmlLoader
from langchain_community.document_transformers import Html2TextTransformer

from agents.basic_agents import BaseAgent
from utilities import print_long_text

log = logging.getLogger(__name__)


if hasattr(ssl, '_create_unverified_context'):
    ssl._create_default_https_context = ssl._create_unverified_context


def get_feed():
    # Define the RSS feed URL
    feed_url = "https://hnrss.org/newest?count=100"

    # Parse the RSS feed
    feed: FeedParserDict = feedparser.parse(feed_url)

    log.info(len(feed.entries))
    timestamp = time.time()
    return feed, timestamp


def strip_rss_entries(entries: List[FeedParserDict], keys):
    feed = []
    for i in entries:
        stripped_entry = {k: v for k, v in i.items() if k in keys}
        feed.append(stripped_entry)
    return feed


@tool
def search_online(urls: str):
    """Look up things online."""
    # with no_ssl_verification():
    loader = AsyncHtmlLoader(urls)
    docs = loader.load()

    html2text = Html2TextTransformer()
    return html2text.transform_documents(docs)


class RSSTechAgent(BaseAgent):
    def find_interesting_tech_from_feed(self, entry: FeedParserDict):
        article = entry['links'][0].get('href')
        if article:
            try:
                articles = search_online.invoke(article)
                entry['linked_article'] = articles
            except Exception as e:
                print(f'unable to get article: {e}')
                return

        prompt = """Read an entry in an RSS feed and determine which if it contains useful new programming technologies or tools.
Your response should only be about the intended output. Ignore anything irrelevant to the output. Your response must follow the format of the examples given.
Examples:
Two useful techs: {
    "technology": [
        {
            "name": "Kaniko",
            "summary": "Kaniko is an open-source tool that allows you to build container images in Kubernetes without needing to run a Docker daemon inside the cluster.",
            "link": ""
        },
        {
            "name": "needle-in-a-needlestack",
            "summary": "Needle in a Needlestack is a new benchmark to measure how well LLMs pay attention to the information in their context window. NIAN creates a prompt that includes thousands of limericks and the prompt asks a question about one limerick at a specific location.",
            "link": "https://github.com/llmonpy/needle-in-a-needlestack"
        }
    ],
}

No useful techs:
{"technology": []}


"""

        prompt += f"RSS feed entry:{entry}"

        return self.llm.invoke(prompt, response_format={"type": "json_object"})


def parse_hn():
    agent2 = RSSTechAgent(config='stdout')

    responses = []
    result = []
    useful_keys = ['title', 'summary', 'published', 'links', 'comments']

    feed, timestamp = get_feed()
    stripped_feed = strip_rss_entries(feed.entries, keys=useful_keys)

    for i in stripped_feed:
        responses.append(agent2.find_interesting_tech_from_feed(i))

    for i in responses:
        try:
            content = json.loads(i.content)
            if content and content.get('technology'):
                result.append(content)
        except:
            print(i.content)
            continue
    return responses
