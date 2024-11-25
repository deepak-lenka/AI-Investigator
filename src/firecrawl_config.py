"""Firecrawl API configuration settings"""

FIRECRAWL_CONFIG = {
    # API Settings
    "base_url": "https://api.firecrawl.com/v1",
    
    # Cache Settings
    "cache_ttl": 3600,  # Cache lifetime in seconds
    "cache_maxsize": 1000,  # Maximum cache entries
    
    # Rate Limiting
    "rate_limit_calls": 60,  # Calls allowed per period
    "rate_limit_period": 60,  # Period in seconds
    
    # Concurrent Requests
    "max_concurrent": 5,  # Maximum concurrent requests
    
    # Timeouts
    "request_timeout": 30,  # Request timeout in seconds
    "scrape_timeout": 30000,  # Scrape timeout in milliseconds
    
    # Retry Settings
    "max_retries": 3,
    "retry_delay": 1,  # Initial retry delay in seconds
    
    # Output Formats
    "default_formats": ["markdown"],
    
    # Content Settings
    "only_main_content": True,
    "include_subdomains": True,
    "ignore_sitemap": False,
    "url_limit": 5000
}
