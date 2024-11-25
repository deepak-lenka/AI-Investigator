import json
import logging
from typing import Dict, List, Optional, Union, Any
from pathlib import Path
import anthropic
from tenacity import retry, stop_after_attempt, wait_exponential
from datetime import datetime
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)

@dataclass
class AnalysisMetadata:
    """Metadata for tracking analysis runs"""
    case_id: str
    timestamp: str
    model_version: str
    analysis_type: str
    execution_time: float
    token_usage: Dict[str, int]

class ClaudeProcessorError(Exception):
    """Base exception for Claude processor errors"""
    pass

class ValidationError(ClaudeProcessorError):
    """Raised when response validation fails"""
    pass

class ClaudeProcessor:
    """
    Enhanced Claude processor with optimized prompts, context management, and validation.
    """
    
    def __init__(
        self,
        api_key: str = ANTHROPIC_API_KEY,
        model: str = "claude-3-sonnet-20240229",
        temperature: float = 0.2,
        max_tokens: int = 4096,
        cache_dir: Optional[Path] = None
    ):
        """
        Initialize Claude processor with configuration.
        
        Args:
            api_key: Anthropic API key
            model: Claude model version
            temperature: Response randomness (0-1)
            max_tokens: Maximum response tokens
            cache_dir: Optional directory for caching responses
        """
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.cache_dir = cache_dir or Path("./cache")
        self.cache_dir.mkdir(exist_ok=True)
        
        # Load prompt templates
        self.prompts = self._load_prompt_templates()
        
        # Initialize analysis history
        self.analysis_history: List[AnalysisMetadata] = []

    def _validate_response(self, response: Dict[str, Any], expected_schema: Dict[str, Any]) -> bool:
        """
        Validate Claude's response against expected schema
        
        Args:
            response: Response dictionary from Claude
            expected_schema: Expected structure and types
            
        Raises:
            ValidationError: If validation fails
        """
        try:
            def validate_dict(data: Dict, schema: Dict, path: str = "") -> None:
                for key, expected_type in schema.items():
                    if key not in data:
                        raise ValidationError(f"Missing key {key} at {path}")
                    
                    if isinstance(expected_type, dict):
                        if not isinstance(data[key], dict):
                            raise ValidationError(
                                f"Expected dict for {path}{key}, got {type(data[key])}"
                            )
                        validate_dict(data[key], expected_type, f"{path}{key}.")
                    elif isinstance(expected_type, list):
                        if not isinstance(data[key], list):
                            raise ValidationError(
                                f"Expected list for {path}{key}, got {type(data[key])}"
                            )
                    else:
                        if not isinstance(data[key], expected_type):
                            raise ValidationError(
                                f"Expected {expected_type} for {path}{key}, got {type(data[key])}"
                            )
            
            validate_dict(response, expected_schema)
            return True
            
        except ValidationError as e:
            logger.error(f"Validation error: {str(e)}")
            raise

    async def _make_claude_request(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        expected_schema: Optional[Dict] = None,
        cache_key: Optional[str] = None
    ) -> Dict:
        """
        Make request to Claude API with retry, caching, and validation.
        
        Args:
            prompt: Main prompt content
            system_prompt: Optional system prompt
            expected_schema: Optional response validation schema
            cache_key: Optional key for caching
            
        Returns:
            Claude's response parsed as JSON
        """
        # Check cache if cache_key provided
        if cache_key:
            cache_file = self.cache_dir / f"{cache_key}.json"
            if cache_file.exists():
                with open(cache_file) as f:
                    return json.load(f)

        start_time = datetime.now()
        
        try:
            messages = []
            if system_prompt:
                messages.append({
                    "role": "system",
                    "content": system_prompt
                })
                
            messages.append({
                "role": "user",
                "content": prompt
            })
            
            response = await self.client.messages.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            parsed_response = json.loads(response.content[0].text)
            
            # Validate response if schema provided
            if expected_schema:
                self._validate_response(parsed_response, expected_schema)
            
            # Cache response if cache_key provided
            if cache_key:
                with open(self.cache_dir / f"{cache_key}.json", "w") as f:
                    json.dump(parsed_response, f, indent=2)
            
            # Record metadata
            execution_time = (datetime.now() - start_time).total_seconds()
            metadata = AnalysisMetadata(
                case_id=cache_key or "unknown",
                timestamp=datetime.now().isoformat(),
                model_version=self.model,
                analysis_type=prompt[:50],  # First 50 chars of prompt
                execution_time=execution_time,
                token_usage={
                    "prompt_tokens": response.usage.input_tokens,
                    "completion_tokens": response.usage.output_tokens
                }
            )
            self.analysis_history.append(metadata)
            
            return parsed_response
            
        except Exception as e:
            logger.error(f"Claude API error: {str(e)}")
            raise ClaudeProcessorError(f"Claude request failed: {str(e)}")

    async def generate_synthesis_report(self, results: List[Dict]) -> Dict:
        """
        Generate a synthesis report from multiple analyses
        
        Args:
            results: List of analysis results
            
        Returns:
            Synthesis report with patterns and insights
        """
        synthesis_prompt = """
        Analyze these AI implementation results and generate a synthesis report.
        Focus on patterns, trends, and actionable insights.
        
        Analysis results:
        {results}
        
        Generate a synthesis report in JSON format with:
        1. Key patterns across implementations
        2. Common success factors
        3. Shared challenges
        4. Recommendations
        5. Industry-specific insights
        """.format(results=json.dumps(results, indent=2))
        
        expected_schema = {
            "patterns": Dict[str, List[str]],
            "success_factors": List[str],
            "challenges": List[str],
            "recommendations": Dict[str, List[str]],
            "industry_insights": Dict[str, str]
        }
        
        return await self._make_claude_request(
            prompt=synthesis_prompt,
            expected_schema=expected_schema,
            cache_key=f"synthesis_{datetime.now().strftime('%Y%m%d')}"
        )

    def export_analysis_history(self, output_path: Optional[Path] = None) -> None:
        """Export analysis history to JSON file"""
        if not output_path:
            output_path = self.cache_dir / "analysis_history.json"
            
        with open(output_path, "w") as f:
            json.dump(
                [asdict(metadata) for metadata in self.analysis_history],
                f,
                indent=2
            )

    async def analyze_implementation_risks(self, content: str) -> Dict:
        """
        Analyze potential risks and mitigation strategies
        
        Args:
            content: Case study content
            
        Returns:
            Risk analysis results
        """
        risk_prompt = """
        Analyze potential risks in this AI implementation.
        Consider technical, organizational, and ethical risks.
        Provide mitigation strategies for each risk.
        
        Content:
        {content}
        
        Respond in JSON format with:
        1. Risk categories and specific risks
        2. Impact assessment
        3. Probability assessment
        4. Mitigation strategies
        5. Monitoring recommendations
        """.format(content=content)
        
        expected_schema = {
            "risks": Dict[str, List[Dict[str, Any]]],
            "impact_assessment": Dict[str, str],
            "probability_assessment": Dict[str, float],
            "mitigation_strategies": Dict[str, List[str]],
            "monitoring": Dict[str, List[str]]
        }
        
        return await self._make_claude_request(
            prompt=risk_prompt,
            expected_schema=expected_schema
        )

    async def generate_progress_dashboard(self, case_id: str) -> Dict:
        """
        Generate a progress tracking dashboard
        
        Args:
            case_id: Case study identifier
            
        Returns:
            Dashboard data
        """
        try:
            # Load all analyses for this case
            analyses = []
            case_dir = Path(RAW_CONTENT_DIR) / f"case_{case_id}"
            analysis_file = case_dir / "claude_analysis.json"
            
            if analysis_file.exists():
                with open(analysis_file) as f:
                    analyses = json.load(f)
            
            # Generate dashboard data
            dashboard_data = {
                "case_id": case_id,
                "last_updated": datetime.now().isoformat(),
                "analysis_coverage": {
                    "technical": bool(analyses.get("technical")),
                    "business": bool(analyses.get("business")),
                    "risks": bool(analyses.get("risks")),
                    "lessons": bool(analyses.get("lessons"))
                },
                "key_metrics": self._extract_key_metrics(analyses),
                "status": self._determine_analysis_status(analyses),
                "next_steps": self._generate_next_steps(analyses)
            }
            
            # Save dashboard
            output_path = case_dir / "dashboard.json"
            with open(output_path, "w") as f:
                json.dump(dashboard_data, f, indent=2)
            
            return dashboard_data
            
        except Exception as e:
            logger.error(f"Dashboard generation failed: {str(e)}")
            raise ClaudeProcessorError(f"Dashboard generation failed: {str(e)}")

    def _extract_key_metrics(self, analyses: Dict) -> Dict:
        """Extract key metrics from analyses"""
        metrics = {}
        if "business" in analyses:
            metrics["roi"] = analyses["business"].get("quantitative_metrics", {}).get("roi")
            metrics["efficiency_gains"] = analyses["business"].get("quantitative_metrics", {}).get("efficiency_gains")
        
        if "technical" in analyses:
            metrics["implementation_status"] = analyses["technical"].get("implementation", {}).get("status")
        
        return metrics

    def _determine_analysis_status(self, analyses: Dict) -> str:
        """Determine overall analysis status"""
        required_analyses = {"technical", "business", "risks", "lessons"}
        completed = set(analyses.keys())
        
        if not completed:
            return "Not Started"
        elif completed == required_analyses:
            return "Complete"
        else:
            return f"In Progress ({len(completed)}/{len(required_analyses)})"

    def _generate_next_steps(self, analyses: Dict) -> List[str]:
        """Generate recommended next steps"""
        next_steps = []
        required_analyses = {"technical", "business", "risks", "lessons"}
        
        for analysis in required_analyses - set(analyses.keys()):
            next_steps.append(f"Complete {analysis} analysis")
            
        if "risks" in analyses:
            for risk in analyses["risks"].get("high_priority", []):
                next_steps.append(f"Address high-priority risk: {risk}")
                
        return next_steps