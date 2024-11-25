import pandas as pd
import asyncio
import logging
from pathlib import Path
from typing import Dict, List
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from src.config import (
    INPUT_DIR,
    RAW_DIR,
    LOGS_DIR,
    LOG_FORMAT,
    SECTIONS_DIR,
    REPORTS_DIR,
    LOGGING_CONFIG
)
from src.scrapers.web_loader import WebLoader
from src.processors.claude_processor import ClaudeProcessor
from src.scrapers.website_crawler import WebsiteCrawler, WebsiteCrawlerError

# Configure logging
logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger(__name__)
console = Console()

async def load_urls_from_csv() -> List[str]:
    """Load URLs from a CSV file."""
    try:
        csv_path = Path(INPUT_DIR) / "ai case studies - Sheet1.csv"
        if not csv_path.exists():
            csv_path = Path(INPUT_DIR) / "urls.csv"

        if not csv_path.exists():
            logger.error(f"No CSV file found in {INPUT_DIR}")
            return []

        df = pd.read_csv(csv_path)
        first_column = df.columns[0]
        if first_column != 'url':
            df = df.rename(columns={first_column: 'url'})

        urls = df['url'].tolist()
        logger.info(f"Loaded {len(urls)} URLs from CSV")
        return urls

    except Exception as e:
        logger.error(f"Error loading URLs from CSV: {str(e)}")
        return []

async def process_case_study(web_loader: WebLoader, claude_processor: ClaudeProcessor, url: str, index: int, progress=None):
    """Process a single case study."""
    console.rule(f"Processing Case Study #{index + 1}")
    console.print(f"URL: {url}", style="blue")

    try:
        if progress:
            progress.update(progress.task_ids[0], description="📥 Extracting content...")
        content = await web_loader.extract_case_study(url)

        if not content:
            console.print("❌ Failed to extract content", style="red")
            return

        if progress:
            progress.update(progress.task_ids[0], description="💾 Saving raw content...")
        await web_loader.save_raw_content(index, content)

        if progress:
            progress.update(progress.task_ids[0], description="🔍 Analyzing relevance...")
        analysis = await claude_processor.analyze_enterprise_relevance(content['content'])

        if analysis.get('is_enterprise_ai'):
            console.print("\n✅ Qualified as Enterprise AI Case Study", style="green")
            table = Table(show_header=True, header_style="bold magenta")
            table.add_column("Attribute", style="cyan")
            table.add_column("Value", style="yellow")
            table.add_row("Company", analysis.get('company_details', {}).get('name', 'Unknown'))
            table.add_row("Industry", analysis.get('company_details', {}).get('industry', 'Unknown'))
            table.add_row("Technologies", ', '.join(analysis.get('ai_implementation', {}).get('technologies', [])))
            table.add_row("Confidence", f"{analysis.get('confidence_score', 0.0):.2f}")
            console.print(table)

            if progress:
                progress.update(progress.task_ids[0], description="📝 Generating executive report...")
            executive_report = await claude_processor.generate_executive_report(
                content['content'],
                analysis
            )

            if executive_report:
                if progress:
                    progress.update(progress.task_ids[0], description="💾 Saving reports...")
                if await claude_processor.save_reports(index, content, analysis, executive_report):
                    console.print("\nReports saved:", style="green")
                    console.print(f"📄 Individual report: reports/individual/case_{index}.md")
                    console.print(f"📊 Cross-case analysis: reports/cross_case_analysis.json")
                    console.print(f"📈 Executive dashboard: reports/executive_dashboard.json")
                else:
                    console.print("❌ Failed to save some reports", style="red")
            else:
                console.print("❌ Failed to generate executive report", style="red")
        else:
            console.print("\n⚠️ Not an Enterprise AI Case Study", style="yellow")
            console.print(f"Reason: {analysis.get('disqualification_reason')}")
        await asyncio.sleep(2)

    except Exception as e:
        logger.error(f"Error processing case study #{index + 1}: {str(e)}")
        console.print(f"❌ Error: {str(e)}", style="red")

async def crawl_website(base_url: str):
    """Run the website crawler for a given URL."""
    crawler = WebsiteCrawler(
        base_url=base_url,
        max_retries=3,
        timeout=30,
        rate_limit_pause=1.0,
        max_pages=100
    )
    try:
        results = await crawler.crawl()
        logger.info(f"Successfully crawled: {len(results['successful_urls'])} pages")
        logger.info(f"Failed URLs: {len(results['failed_urls'])}")
        logger.info(f"Statistics: {crawler.get_statistics()}")
        return results
    except WebsiteCrawlerError as e:
        logger.error(f"Crawling failed: {str(e)}")
        raise

async def main():
    """Main entry point for the application."""
    try:
        web_loader = WebLoader()
        claude_processor = ClaudeProcessor()
        console.print("\n=== AI Enterprise Case Study Analyzer ===", style="bold green")
        console.print("1. Analyze specific case study URLs from CSV")
        console.print("2. Analyze case studies from a company website")

        mode = console.input("\nEnter mode (1 or 2): ").strip()

        if mode == "1":
            urls = await load_urls_from_csv()
            if not urls:
                console.print("❌ No URLs found in CSV file", style="red")
                return
            for index, url in enumerate(urls):
                await process_case_study(web_loader, claude_processor, url, index)
        elif mode == "2":
            website_url = console.input("\nEnter company website URL: ").strip()
            results = await crawl_website(website_url)
            if results:
                console.print(f"✅ Found {len(results['successful_urls'])} case studies.")
                for idx, url in enumerate(results['successful_urls']):
                    await process_case_study(web_loader, claude_processor, url, idx)
        else:
            console.print("❌ Invalid mode selected", style="red")

        # Single case analysis
        case_id = "123"
        console.print(f"\nAnalyzing single case study with ID: {case_id}")
        single_case_results = await claude_processor.generate_complete_analysis(case_id)
        
        # Access specific analyses
        technical = single_case_results["technical"]
        business = single_case_results["business"]
        lessons = single_case_results["lessons"]
        
        console.print(f"\nSingle Case Analysis Results for Case ID: {case_id}")
        console.print(f"Technical: {technical}")
        console.print(f"Business: {business}")
        console.print(f"Lessons: {lessons}")
        
        # Cross-case analysis
        case_ids = ["123", "124", "125"]
        console.print(f"\nAnalyzing cross-case study with Case IDs: {', '.join(case_ids)}")
        cross_case_results = await claude_processor.generate_cross_case_analysis(case_ids)
        
        console.print(f"\nCross-case analysis results:")
        console.print(f"{cross_case_results}")

    except Exception as e:
        logger.error(f"Main process error: {str(e)}")
        console.print(f"\n❌ Error: {str(e)}", style="red")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
