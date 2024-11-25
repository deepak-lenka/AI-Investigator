import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Directory structure
BASE_DIR = Path(__file__).parent.parent
INPUT_DIR = BASE_DIR / "input"
RAW_DIR = BASE_DIR / "raw_content"
LOGS_DIR = BASE_DIR / "logs"
SECTIONS_DIR = BASE_DIR / "sections"
REPORTS_DIR = BASE_DIR / "reports"
REPORTS_INDIVIDUAL_DIR = REPORTS_DIR / "individual"
REPORTS_CROSS_CASE_DIR = REPORTS_DIR / "cross_case_analysis"
REPORTS_EXECUTIVE_DIR = REPORTS_DIR / "executive_dashboard"

# Create directories if they don't exist
for directory in [INPUT_DIR, RAW_DIR, LOGS_DIR, SECTIONS_DIR, REPORTS_DIR, 
                  REPORTS_INDIVIDUAL_DIR, REPORTS_CROSS_CASE_DIR, REPORTS_EXECUTIVE_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Claude settings
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
if not ANTHROPIC_API_KEY:
    raise ValueError("ANTHROPIC_API_KEY environment variable is not set")

CLAUDE_MODEL = "claude-3-5-sonnet-20241022"
CLAUDE_TEMPERATURE = 0.2
CLAUDE_MAX_TOKENS = 4096

# Web scraping settings
MAX_RETRIES = 3
REQUEST_TIMEOUT = 30
RETRY_DELAY = 1

# FireCrawl API settings
FIRECRAWL_API_KEY = os.getenv("FIRECRAWL_API_KEY")
if not FIRECRAWL_API_KEY:
    raise ValueError("FIRECRAWL_API_KEY environment variable is not set")

# Logging configuration
LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        },
    },
    'handlers': {
        'default': {
            'level': 'INFO',
            'formatter': 'standard',
            'class': 'logging.StreamHandler',
            'stream': 'ext://sys.stdout',
        },
        'file': {
            'level': 'INFO',
            'formatter': 'standard',
            'class': 'logging.FileHandler',
            'filename': LOGS_DIR / 'crawler.log',
            'mode': 'a',
        },
    },
    'loggers': {
        '': {  # root logger
            'handlers': ['default', 'file'],
            'level': 'INFO',
            'propagate': True
        }
    }
}

# Crawler configurations
CRAWLER_CONFIG = {
    'max_retries': 3,
    'timeout': 30,
    'rate_limit_pause': 1.0,
    'max_pages': 100,
    'user_agent': 'AI Case Study Analyzer Bot 1.0',
}
