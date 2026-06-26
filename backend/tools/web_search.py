"""Web search tools: Tavily and Firecrawl."""
from __future__ import annotations
import json
import logging
from langchain_core.tools import tool
from config import settings

logger = logging.getLogger(__name__)


def get_web_search_tool():
    """Return the best available web search tool (Tavily preferred)."""
    if settings.tavily_api_key:
        from tavily import TavilyClient
        client = TavilyClient(api_key=settings.tavily_api_key)

        @tool
        def web_search(query: str) -> str:
            """Search the web for current information using Tavily."""
            try:
                result = client.search(query=query, max_results=5)
                items = []
                for r in result.get("results", []):
                    items.append(f"**{r['title']}**\n{r['url']}\n{r.get('content','')}")
                return "\n\n---\n\n".join(items) if items else "No results found."
            except Exception as e:
                logger.error(f"Tavily search error: {e}")
                return f"Search failed: {e}"

        return web_search

    if settings.firecrawl_api_key:
        from firecrawl import FirecrawlApp
        app = FirecrawlApp(api_key=settings.firecrawl_api_key)

        @tool
        def web_search(query: str) -> str:
            """Search the web using Firecrawl."""
            try:
                result = app.search(query=query, limit=5)
                items = []
                for r in (result.get("data") or []):
                    items.append(f"**{r.get('title','')}**\n{r.get('url','')}\n{r.get('markdown','')[:500]}")
                return "\n\n---\n\n".join(items) if items else "No results found."
            except Exception as e:
                logger.error(f"Firecrawl search error: {e}")
                return f"Search failed: {e}"

        return web_search

    return None


def get_scrape_tool():
    """Return a page scraping tool (Firecrawl preferred, Tavily fallback)."""
    if settings.firecrawl_api_key:
        from firecrawl import FirecrawlApp
        app = FirecrawlApp(api_key=settings.firecrawl_api_key)

        @tool
        def web_scrape(url: str) -> str:
            """Scrape a web page and return its content as markdown."""
            try:
                result = app.scrape_url(url, formats=["markdown"])
                return result.get("markdown", "Could not extract content.")
            except Exception as e:
                logger.error(f"Firecrawl scrape error: {e}")
                return f"Scrape failed: {e}"

        return web_scrape

    return None


def get_available_tools() -> list:
    """Return list of configured web tools."""
    tools = []
    search = get_web_search_tool()
    if search:
        tools.append(search)
    scrape = get_scrape_tool()
    if scrape:
        tools.append(scrape)
    return tools
