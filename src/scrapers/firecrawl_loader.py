import asyncio
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Union
import aiohttp
import backoff
from cachetools import TTLCache
from ratelimit import limits, sleep_and_retry

from ..config import (
    FIRECRAWL_API_KEY,
    RAW_CONTENT_DIR,
    FIRECRAWL_CONFIG
)

logger = logging.getLogger(__name__)

class FirecrawlError(Exception):
    """Base exception for Firecrawl API errors"""
    pass

class FirecrawlLoader:
    """
    Optimized Firecrawl API client with caching, rate limiting, and concurrent requests.
    """
    
    def __init__(
        self,
        api_key: str = FIRECRAWL_API_KEY,
        cache_ttl: int = 3600,
        cache_maxsize: int = 1000,
        max_concurrent: int = 5,
        rate_limit_calls: int = 60,
        rate_limit_period: int = 60
    ):
        """
        Initialize the Firecrawl loader with configuration.
        
        Args:
            api_key: Firecrawl API key
            cache_ttl: Cache time-to-live in seconds
            cache_maxsize: Maximum cache size
            max_concurrent: Maximum concurrent requests
            rate_limit_calls: Number of calls allowed per period
            rate_limit_period: Rate limit period in seconds
        """
        self.api_key = api_key
        self.base_url = "https://api.firecrawl.com/v1"
        self.cache = TTLCache(maxsize=cache_maxsize, ttl=cache_ttl)
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.session = None
        self.rate_limit_calls = rate_limit_calls
        self.rate_limit_period = rate_limit_period

    async def __aenter__(self):
        """Set up async context manager"""
        timeout = aiohttp.ClientTimeout(total=30)
        self.session = aiohttp.ClientSession(
            timeout=timeout,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Clean up async context manager"""
        if self.session:
            await self.session.close()

    @sleep_and_retry
    @limits(calls=60, period=60)
    @backoff.on_exception(
        backoff.expo,
        (aiohttp.ClientError, FirecrawlError),
        max_tries=3
    )
    async def _make_request(
        self,
        endpoint: str,
        method: str = "GET",
        params: Optional[Dict] = None,
        data: Optional[Dict] = None
    ) -> Dict:
        """
        Make rate-limited API request with retries and error handling.
        
        Args:
            endpoint: API endpoint
            method: HTTP method
            params: Query parameters
            data: Request body data
            
        Returns:
            API response data
            
        Raises:
            FirecrawlError: On API or connection errors
        """
        url = f"{self.base_url}/{endpoint}"
        cache_key = f"{method}:{url}:{json.dumps(params)}:{json.dumps(data)}"

        # Check cache
        if method == "GET" and cache_key in self.cache:
            logger.debug(f"Cache hit for {cache_key}")
            return self.cache[cache_key]

        async with self.semaphore:
            try:
                async with self.session.request(
                    method,
                    url,
                    params=params,
                    json=data
                ) as response:
                    if response.status == 429:
                        retry_after = int(response.headers.get("Retry-After", 60))
                        logger.warning(f"Rate limited. Waiting {retry_after}s")
                        await asyncio.sleep(retry_after)
                        raise FirecrawlError("Rate limit exceeded")

                    response.raise_for_status()
                    result = await response.json()

                    # Cache GET requests
                    if method == "GET":
                        self.cache[cache_key] = result

                    return result

            except aiohttp.ClientError as e:
                logger.error(f"Firecrawl API error: {str(e)}")
                raise FirecrawlError(f"API request failed: {str(e)}")

    async def map_website(
        self,
        url: str,
        include_subdomains: bool = True,
        ignore_sitemap: bool = False,
        limit: int = 5000
    ) -> Dict:
        """
        Map website URLs using Firecrawl's map endpoint.
        
        Args:
            url: Website URL to map
            include_subdomains: Whether to include subdomains
            ignore_sitemap: Whether to ignore sitemap
            limit: Maximum URLs to map
            
        Returns:
            Mapping results
        """
        params = {
            "url": url,
            "includeSubdomains": include_subdomains,
            "ignoreSitemap": ignore_sitemap,
            "limit": limit
        }
        
        return await self._make_request("map", params=params)

    async def scrape_content(
        self,
        url: str,
        only_main_content: bool = True,
        formats: List[str] = ["markdown"],
        timeout: int = 30000
    ) -> Dict:
        """
        Scrape content using Firecrawl's scrape endpoint.
        
        Args:
            url: URL to scrape
            only_main_content: Whether to extract only main content
            formats: Output formats
            timeout: Request timeout in ms
            
        Returns:
            Scraped content
        """
        params = {
            "url": url,
            "onlyMainContent": only_main_content,
            "formats": formats,
            "timeout": timeout
        }
        
        return await self._make_request("scrape", params=params)

    async def bulk_scrape(
        self,
        urls: List[str],
        save_to_disk: bool = True
    ) -> List[Dict]:
        """
        Scrape multiple URLs concurrently.
        
        Args:
            urls: List of URLs to scrape
            save_to_disk: Whether to save results to disk
            
        Returns:
            List of scraping results
        """
        tasks = [self.scrape_content(url) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        if save_to_disk:
            await self._save_results(urls, results)
            
        return results

    async def _save_results(
        self,
        urls: List[str],
        results: List[Dict]
    ) -> None:
        """
        Save scraping results to disk.
        
        Args:
            urls: Scraped URLs
            results: Scraping results
        """
        for url, result in zip(urls, results):
            if isinstance(result, Exception):
                logger.error(f"Failed to scrape {url}: {str(result)}")
                continue

            case_dir = RAW_CONTENT_DIR / f"case_{hash(url) % 1000}"
            case_dir.mkdir(parents=True, exist_ok=True)

            # Save raw content
            with open(case_dir / "raw_content.txt", "w") as f:
                f.write(result.get("content", ""))

            # Save metadata
            with open(case_dir / "metadata.json", "w") as f:
                json.dump({
                    "url": url,
                    "scraped_at": datetime.now().isoformat(),
                    "success": True
                }, f, indent=2)

            # Save structured content
            with open(case_dir / "structured_content.json", "w") as f:
                json.dump(result, f, indent=2)

    async def get_statistics(self) -> Dict:
        """Get API usage statistics"""
        return {
            "cache_info": {
                "size": len(self.cache),
                "maxsize": self.cache.maxsize,
                "hits": self.cache.hits,
                "misses": self.cache.misses
            },
            "rate_limits": {
                "calls": self.rate_limit_calls,
                "period": self.rate_limit_period
            }
        }
