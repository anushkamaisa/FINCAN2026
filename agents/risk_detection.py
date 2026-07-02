"""
Risk Detection Agent

This agent identifies potential financial risks and warning signs.
It looks at the normalised data and analysis to flag issues like:
- Low cash runway
- Revenue concentration
- Rising costs
- Margin pressure

Each risk is classified as LOW/MEDIUM/HIGH severity.
"""

import json
import re
from typing import Any, Dict
from agents.base_agent import BaseAgent


class RiskDetectionAgent(BaseAgent):
    """
    Identifies and classifies financial risks.
    
    Input: Normalised data AND financial analysis
    Output: Risk assessment with severity ratings
    """
    
    def __init__(self, llm_provider: str = "claude"):
        super().__init__(
            name="Risk Detector",
            description="Identifies early-warning financial risks and classifies their severity",
            llm_provider=llm_provider
        )
    
    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Detect financial risks from the data and analysis.
        
        Args:
            input_data: Dictionary containing:
                - normalised_data: Output from DataNormalizerAgent
                - finance_analysis: Output from FinanceAnalysisAgent
        
        Returns:
            Dictionary containing risk assessment
        """
        self._log("Starting risk detection...")
        
        # Combine inputs into a clear format
        combined_data = {
            "normalised_financial_data": input_data.get("normalised_data", {}),
            "financial_analysis": input_data.get("finance_analysis", {})
        }
        data_str = json.dumps(combined_data, indent=2)
        
        prompt = self._build_prompt(
            task_description="""
Review the financial data and analysis to identify potential risks.
Focus on early-warning signs that could affect business viability.
Be specific about what the numbers show - don't invent concerns.
""",
            input_data=data_str,
            output_format="""
Return a JSON object with this exact structure:
{
    "overall_risk_level": "LOW/MEDIUM/HIGH",
    "risk_summary": "One paragraph summary of the risk landscape",
    
    "risks": [
        {
            "risk_id": "RISK_001",
            "category": "CASH_FLOW/REVENUE/EXPENSES/MARGIN/CONCENTRATION/EXTERNAL",
            "title": "Short descriptive title",
            "description": "What the risk is and why it matters",
            "severity": "LOW/MEDIUM/HIGH",
            "evidence": "Specific data points supporting this risk",
            "potential_impact": "What could happen if unaddressed",
            "timeframe": "IMMEDIATE/SHORT_TERM/MEDIUM_TERM"
        }
    ],
    
    "cash_runway_assessment": {
        "estimated_months": number or null,
        "confidence": "HIGH/MEDIUM/LOW",
        "assumptions": "How this was calculated",
        "warning_level": "SAFE/MONITOR/WARNING/CRITICAL"
    },
    
    "revenue_concentration_risk": {
        "concentration_level": "LOW/MEDIUM/HIGH",
        "details": "Explanation of revenue source dependencies"
    },
    
    "cost_rigidity_assessment": {
        "fixed_cost_ratio": number or null,
        "flexibility_rating": "FLEXIBLE/MODERATE/RIGID",
        "details": "Explanation of cost structure"
    },
    
    "external_risk_factors": [
        {
            "factor": "Description of external risk",
            "relevance": "Why this matters for this business"
        }
    ],
    
    "positive_indicators": [
        "List any positive signs that reduce risk"
    ],
    
    "monitoring_priorities": [
        {
            "metric": "What to watch",
            "threshold": "When to be concerned",
            "frequency": "How often to check"
        }
    ],
    
    "data_gaps_affecting_assessment": [
        "List any missing data that limits risk assessment accuracy"
    ]
}
""",
            additional_instructions="""
Risk assessment guidelines for NZ SMEs:

CASH RUNWAY:
- <3 months = CRITICAL
- 3-6 months = WARNING  
- 6-12 months = MONITOR
- >12 months = SAFE

REVENUE CONCENTRATION:
- >50% from one client = HIGH risk
- >30% from one client = MEDIUM risk
- Well-diversified = LOW risk

MARGIN ASSESSMENT:
- Net margin <5% = HIGH risk
- Net margin 5-10% = MEDIUM risk
- Net margin >10% = Generally healthy for NZ SMEs

Be specific and evidence-based. Don't create risks that aren't supported by the data.
If data is insufficient to assess a risk category, say so.
"""
        )
        
        # Call the LLM
        response = self.llm.generate(
            prompt=prompt,
            system_prompt=self._get_system_prompt(),
            temperature=0.2
        )
        
        # Parse the response
        risk_assessment = self._parse_response(response)
        
        # Add metadata
        risk_assessment["_agent"] = self.name
        
        risk_level = risk_assessment.get("overall_risk_level", "Unknown")
        risk_count = len(risk_assessment.get("risks", []))
        self._log(f"Risk detection complete. Overall: {risk_level}, Risks identified: {risk_count}")
        
        return risk_assessment
    
    def _parse_response(self, response: str) -> Dict[str, Any]:
        """Extract JSON from the LLM response."""
        
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response, re.DOTALL)
        
        if json_match:
            json_str = json_match.group(1)
        else:
            start = response.find('{')
            end = response.rfind('}')
            if start != -1 and end != -1:
                json_str = response[start:end + 1]
            else:
                return {
                    "error": "Could not parse risk assessment response",
                    "raw_response": response,
                    "overall_risk_level": "UNKNOWN"
                }
        
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            return {
                "error": f"JSON parsing error: {str(e)}",
                "raw_response": response,
                "overall_risk_level": "UNKNOWN"
            }
    
    def _get_system_prompt(self) -> str:
        """Custom system prompt for risk detection."""
        return """You are a financial risk analyst specialising in New Zealand small businesses.

Your role is to identify risks - things that could go wrong or cause financial stress.

Key principles:
1. Be evidence-based: only flag risks supported by the data
2. Be specific: cite actual numbers, not vague concerns
3. Be proportionate: don't overstate minor issues
4. Be practical: focus on risks the business owner can actually monitor/address
5. Be clear about uncertainty: if data is limited, say so

Common risks for NZ SMEs:
- Cash flow timing (especially with 20th of month payment cycles)
- Seasonal revenue fluctuations
- Key client dependency
- Fixed cost commitments vs variable revenue
- External factors (interest rates, supply chains)

Remember: Identify risks, don't cause panic. Frame issues constructively."""
