"""
Wikipedia Content Fetcher Module
Fetches Wikipedia article content using the REST API with caching support
"""

import json
import os
import re
import time
from typing import Dict, List, Optional
from urllib.parse import unquote
import asyncio
import aiohttp
from bs4 import BeautifulSoup


class WikipediaFetcher:
    """Fetches and caches Wikipedia article content"""

    # Wikipedia REST API base URL
    API_BASE = "https://en.wikipedia.org/api/rest_v1/page/html"

    def __init__(self, cache_dir: str = "cache"):
        """
        Initialize the fetcher with optional cache directory

        Args:
            cache_dir: Directory to store cached content
        """
        self.cache_dir = cache_dir
        self.cache_file = os.path.join(cache_dir, "wiki_content.json")
        self.cache: Dict[str, str] = {}
        self._load_cache()

    def _load_cache(self):
        """Load cached content from disk"""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    self.cache = json.load(f)
            except (json.JSONDecodeError, IOError):
                self.cache = {}

    def _save_cache(self):
        """Save cache to disk"""
        os.makedirs(self.cache_dir, exist_ok=True)
        with open(self.cache_file, "w", encoding="utf-8") as f:
            json.dump(self.cache, f, ensure_ascii=False, indent=2)

    def _extract_title_from_url(self, url: str) -> Optional[str]:
        """
        Extract Wikipedia article title from URL

        Args:
            url: Wikipedia article URL

        Returns:
            Article title or None if invalid URL
        """
        # Pattern to match Wikipedia URLs
        patterns = [
            r"https?://en\.wikipedia\.org/wiki/(.+?)(?:#.*)?$",
            r"https?://en\.m\.wikipedia\.org/wiki/(.+?)(?:#.*)?$",
        ]

        for pattern in patterns:
            match = re.match(pattern, url)
            if match:
                title = unquote(match.group(1))
                return title.replace("_", " ")

        return None

    def _html_to_text(self, html_content: str) -> str:
        """
        Convert HTML content to plain text

        Args:
            html_content: Raw HTML from Wikipedia

        Returns:
            Cleaned plain text
        """
        soup = BeautifulSoup(html_content, "html.parser")

        # Remove script and style elements
        for element in soup(["script", "style", "sup", "table", "figure"]):
            element.decompose()

        # Remove reference sections and navigation
        for element in soup.find_all(class_=["mw-references-wrap", "navbox", "reflist", "infobox"]):
            element.decompose()

        # Get text
        text = soup.get_text(separator=" ", strip=True)

        # Clean up whitespace
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"\[\d+\]", "", text)  # Remove reference numbers

        return text.strip()

    def fetch_article(self, url: str) -> Optional[str]:
        """
        Fetch a single Wikipedia article content

        Args:
            url: Wikipedia article URL

        Returns:
            Article text content or None if failed
        """
        # Check cache first
        if url in self.cache:
            return self.cache[url]

        title = self._extract_title_from_url(url)
        if not title:
            return None

        # URL encode the title for API request
        api_title = title.replace(" ", "_")
        api_url = f"{self.API_BASE}/{api_title}"

        try:
            import requests
            headers = {
                "User-Agent": "RAGSystem/1.0 (Educational Project; Contact: student@example.com)"
            }
            response = requests.get(api_url, headers=headers, timeout=30)

            if response.status_code == 200:
                text = self._html_to_text(response.text)
                self.cache[url] = text
                self._save_cache()
                return text
            else:
                # Try alternative: Wikipedia API summary
                summary_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{api_title}"
                response = requests.get(summary_url, headers=headers, timeout=30)
                if response.status_code == 200:
                    data = response.json()
                    text = data.get("extract", "")
                    if text:
                        self.cache[url] = text
                        self._save_cache()
                        return text
        except Exception as e:
            print(f"Error fetching {url}: {e}")

        return None

    async def _fetch_article_async(
        self,
        session: aiohttp.ClientSession,
        url: str
    ) -> tuple[str, Optional[str]]:
        """
        Asynchronously fetch a single Wikipedia article

        Args:
            session: aiohttp session
            url: Wikipedia article URL

        Returns:
            Tuple of (url, content) or (url, None) if failed
        """
        # Check cache first
        if url in self.cache:
            return (url, self.cache[url])

        title = self._extract_title_from_url(url)
        if not title:
            return (url, None)

        api_title = title.replace(" ", "_")
        api_url = f"{self.API_BASE}/{api_title}"

        headers = {
            "User-Agent": "RAGSystem/1.0 (Educational Project; Contact: student@example.com)"
        }

        try:
            async with session.get(api_url, headers=headers, timeout=aiohttp.ClientTimeout(total=30)) as response:
                if response.status == 200:
                    html = await response.text()
                    text = self._html_to_text(html)
                    return (url, text)
                else:
                    # Try summary API
                    summary_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{api_title}"
                    async with session.get(summary_url, headers=headers) as summary_response:
                        if summary_response.status == 200:
                            data = await summary_response.json()
                            text = data.get("extract", "")
                            return (url, text) if text else (url, None)
        except Exception as e:
            print(f"Error fetching {url}: {e}")

        return (url, None)

    async def fetch_articles_async(
        self,
        urls: List[str],
        max_concurrent: int = 5
    ) -> Dict[str, str]:
        """
        Fetch multiple Wikipedia articles in parallel

        Args:
            urls: List of Wikipedia article URLs
            max_concurrent: Maximum concurrent requests

        Returns:
            Dictionary mapping URL to content
        """
        results: Dict[str, str] = {}

        # Get cached results first
        urls_to_fetch = []
        for url in urls:
            if url in self.cache:
                results[url] = self.cache[url]
            else:
                urls_to_fetch.append(url)

        if not urls_to_fetch:
            return results

        # Fetch uncached URLs
        connector = aiohttp.TCPConnector(limit=max_concurrent)
        async with aiohttp.ClientSession(connector=connector) as session:
            tasks = [self._fetch_article_async(session, url) for url in urls_to_fetch]

            # Use semaphore for rate limiting
            semaphore = asyncio.Semaphore(max_concurrent)

            async def bounded_fetch(task):
                async with semaphore:
                    return await task

            fetched = await asyncio.gather(*[bounded_fetch(task) for task in tasks])

            for url, content in fetched:
                if content:
                    results[url] = content
                    self.cache[url] = content

        # Save updated cache
        self._save_cache()

        return results

    def fetch_articles(self, urls: List[str]) -> Dict[str, str]:
        """
        Fetch multiple Wikipedia articles (synchronous wrapper for async method)

        Args:
            urls: List of Wikipedia article URLs

        Returns:
            Dictionary mapping URL to content
        """
        try:
            # Try to use existing event loop
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If already in async context, run in new thread
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, self.fetch_articles_async(urls))
                    return future.result()
            else:
                return loop.run_until_complete(self.fetch_articles_async(urls))
        except RuntimeError:
            # No event loop exists, create one
            return asyncio.run(self.fetch_articles_async(urls))

    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics"""
        return {
            "cached_articles": len(self.cache),
            "total_characters": sum(len(v) for v in self.cache.values())
        }


# Standalone function for simple usage
def fetch_wikipedia_content(urls: List[str], cache_dir: str = "cache") -> Dict[str, str]:
    """
    Convenience function to fetch Wikipedia content

    Args:
        urls: List of Wikipedia article URLs
        cache_dir: Directory to store cached content

    Returns:
        Dictionary mapping URL to content
    """
    fetcher = WikipediaFetcher(cache_dir=cache_dir)
    return fetcher.fetch_articles(urls)


if __name__ == "__main__":
    # Test the fetcher
    test_urls = [
        "https://en.wikipedia.org/wiki/Python_(programming_language)",
        "https://en.wikipedia.org/wiki/Machine_learning"
    ]

    fetcher = WikipediaFetcher()
    results = fetcher.fetch_articles(test_urls)

    for url, content in results.items():
        print(f"\n=== {url} ===")
        print(f"Content length: {len(content)} characters")
        print(f"Preview: {content[:500]}...")
