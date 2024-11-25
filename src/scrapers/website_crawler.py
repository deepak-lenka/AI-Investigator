import time
import logging
from typing import Dict, List, Optional
from urllib.parse import urljoin
import backoff  # Add to requirements.txt
import requests
from requests.exceptions import (
    RequestException,
    ConnectionError,
    Timeout,
    TooManyRedirects,
    HTTPError
)

logger = logging.getLogger(__name__)

class WebsiteCrawlerError(Exception):
    """Base exception for website crawler errors"""
    pass

class RateLimitError(WebsiteCrawlerError):
    """Raised when rate limit is hit"""
    pass

class CrawlError(WebsiteCrawlerError):
    """Raised when crawling fails"""
    pass

class WebsiteCrawler:
    def __init__(
        self,
        base_url: str,
        max_retries: int = 3,
        timeout: int = 30,
        rate_limit_pause: float = 1.0,
        max_pages: Optional[int] = None
    ):
        """
        Initialize the website crawler with configuration parameters.
        
        Args:
            base_url: The starting URL for crawling
            max_retries: Maximum number of retry attempts for failed requests
            timeout: Request timeout in seconds
            rate_limit_pause: Pause between requests in seconds
            max_pages: Maximum number of pages to crawl (None for unlimited)
        """
        self.base_url = base_url
        self.max_retries = max_retries
        self.timeout = timeout
        self.rate_limit_pause = rate_limit_pause
        self.max_pages = max_pages
        self.session = self._create_session()
        self.crawled_urls = set()
        self.failed_urls = set()

    def _create_session(self) -> requests.Session:
        """Create and configure requests session with retry mechanism"""
        session = requests.Session()
        session.headers.update({
            'User-Agent': 'AI Case Study Analyzer Bot 1.0',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9',
        })
        return session

    @backoff.on_exception(
        backoff.expo,
        (ConnectionError, Timeout, HTTPError),
        max_tries=3,
        jitter=None
    )
    async def _make_request(self, url: str) -> Optional[requests.Response]:
        """
        Make HTTP request with retry mechanism and error handling.
        
        Args:
            url: URL to request
            
        Returns:
            Response object if successful, None otherwise
            
        Raises:
            RateLimitError: If rate limit is detected
            CrawlError: For other crawling errors
        """
        try:
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
            
            # Check for rate limiting response codes
            if response.status_code in (429, 503):
                retry_after = int(response.headers.get('Retry-After', 60))
                logger.warning(f"Rate limit hit for {url}. Waiting {retry_after}s")
                time.sleep(retry_after)
                raise RateLimitError(f"Rate limited on {url}")
                
            time.sleep(self.rate_limit_pause)  # Rate limiting pause
            return response

        except HTTPError as e:
            logger.error(f"HTTP error for {url}: {str(e)}")
            self.failed_urls.add(url)
            raise CrawlError(f"HTTP error: {str(e)}")
            
        except (ConnectionError, Timeout) as e:
            logger.error(f"Connection error for {url}: {str(e)}")
            self.failed_urls.add(url)
            raise CrawlError(f"Connection error: {str(e)}")
            
        except Exception as e:
            logger.error(f"Unexpected error crawling {url}: {str(e)}")
            self.failed_urls.add(url)
            raise CrawlError(f"Unexpected error: {str(e)}")

    async def crawl(self) -> Dict[str, List[str]]:
        """
        Crawl website starting from base URL with error handling.
        
        Returns:
            Dictionary containing:
            - 'successful_urls': List of successfully crawled URLs
            - 'failed_urls': List of URLs that failed
            - 'skipped_urls': List of URLs skipped due to filters
        """
        results = {
            'successful_urls': [],
            'failed_urls': list(self.failed_urls),
            'skipped_urls': []
        }
        
        try:
            pages_crawled = 0
            urls_to_crawl = {self.base_url}
            
            while urls_to_crawl and (self.max_pages is None or pages_crawled < self.max_pages):
                url = urls_to_crawl.pop()
                
                if url in self.crawled_urls:
                    continue
                    
                try:
                    response = await self._make_request(url)
                    if response is None:
                        continue
                        
                    self.crawled_urls.add(url)
                    results['successful_urls'].append(url)
                    
                    # Extract and queue new URLs
                    new_urls = self._extract_urls(response)
                    filtered_urls = self._filter_urls(new_urls)
                    urls_to_crawl.update(filtered_urls)
                    
                    pages_crawled += 1
                    
                except RateLimitError:
                    # Already handled in _make_request
                    continue
                    
                except CrawlError as e:
                    logger.error(f"Crawl error for {url}: {str(e)}")
                    continue

        except Exception as e:
            logger.error(f"Fatal crawling error: {str(e)}")
            raise CrawlError(f"Fatal crawling error: {str(e)}")
            
        finally:
            self.session.close()
            
        return results

    def _extract_urls(self, response: requests.Response) -> set:
        """Extract all URLs from the response"""
        # Add your URL extraction logic here
        # This is a placeholder implementation
        return set()

    def _filter_urls(self, urls: set) -> set:
        """
        Filter URLs based on criteria:
        - Same domain
        - Not already crawled
        - Matches allowed patterns
        """
        filtered = set()
        for url in urls:
            # Add your filtering logic here
            if url.startswith(self.base_url) and url not in self.crawled_urls:
                filtered.add(url)
        return filtered

    def get_statistics(self) -> Dict:
        """Return crawling statistics"""
        return {
            'total_urls_crawled': len(self.crawled_urls),
            'failed_urls': len(self.failed_urls),
            'success_rate': len(self.crawled_urls) / (len(self.crawled_urls) + len(self.failed_urls)) if self.crawled_urls else 0
        }
