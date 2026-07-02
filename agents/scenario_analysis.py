"""
Scenario Analysis Agent

This agent runs simple "what if" scenarios to help the business owner
understand how changes might affect their finances.

Examples:
- What if revenue drops 20%?
- What if we increase prices by 10%?
- What if a major client leaves?

Each scenario shows estimated impact on cash runway and profitability.
"""

import json
import re
from typing import Any, Dict
from agents.base_agent import BaseAgent


class ScenarioAnalysisAgent(BaseAgent):
    """
    Simulates financial scenarios and estimates impacts.
    
    Input: Normalised data, analysis, and risk assessment
    Output: Scenario projections with explicit assumptions
    """
    
    def __init__(self, llm_provider: str = "claude"):
        super().__init__(
            name="Scenario Analyst",
            description="Simulates financial scenarios and estimates impacts on cash runway and profitability",
            llm_provider=llm_provider
        )
    
    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate and analyse financial scenarios.
        
        Args:
            input_data: Dictionary containing:
                - normalised_data: From DataNormalizerAgent
                - finance_analysis: From FinanceAnalysisAgent
                - risk_assessment: From RiskDetectionAgent
        
        Returns:
            Dictionary containing scenario analyses
        """
        self._log("Starting scenario analysis...")
        
        # Combine inputs
        combined_data = {
            "normalised_financial_data": input_data.get("normalised_data", {}),
            "financial_analysis": input_data.get("finance_analysis", {}),
            "risk_assessment": input_data.get("risk_assessment", {})
        }
        data_str = json.dumps(combined_data, indent=2)
        
        prompt = self._build_prompt(
            task_description="""
Based on the financial data and risks identified, create relevant "what if" scenarios.
These should be realistic situations the business might face.
Show clear calculations with explicit assumptions.
""",
            input_data=data_str,
            output_format="""
Return a JSON object with this exact structure:
{
    "baseline": {
        "monthly_revenue": number,
        "monthly_expenses": number,
        "monthly_net": number,
        "current_cash": number or null,
        "current_runway_months": number or null
    },
    
    "scenarios": [
        {
            "scenario_id": "SCENARIO_001",
            "name": "Short descriptive name",
            "description": "What this scenario represents",
            "relevance": "Why this scenario matters for this business",
            "trigger": "What could cause this to happen",
            "likelihood": "LOW/MEDIUM/HIGH",
            
            "changes": {
                "revenue_change_percent": number,
                "expense_change_percent": number,
                "one_off_impact": number or null
            },
            
            "projected_outcome": {
                "new_monthly_revenue": number,
                "new_monthly_expenses": number,
                "new_monthly_net": number,
                "new_runway_months": number or null,
                "profit_impact_annual": number
            },
            
            "assumptions": [
                "List all assumptions made in this calculation"
            ],
            
            "key_insight": "The most important takeaway from this scenario",
            
            "mitigation_options": [
                "Possible actions to prepare for or respond to this scenario"
            ]
        }
    ],
    
    "stress_test_summary": {
        "most_dangerous_scenario": "Which scenario poses the biggest threat",
        "survival_threshold": "How much revenue loss the business can sustain",
        "resilience_rating": "LOW/MEDIUM/HIGH"
    },
    
    "scenario_limitations": [
        "What these scenarios don't account for"
    ]
}
""",
            additional_instructions="""
Scenario guidelines:

Create 3-5 relevant scenarios. Always include:
1. DOWNSIDE: A revenue drop scenario (e.g., -20%, -30%)
2. COST PRESSURE: An expense increase scenario  
3. SPECIFIC RISK: A scenario based on the risks identified (e.g., losing biggest client)
4. UPSIDE (optional): A growth scenario to show opportunity

For each scenario:
- Show your math clearly
- State all assumptions explicitly
- Keep calculations simple and verifiable
- Use monthly figures for consistency
- Round to sensible numbers

NZ-specific considerations:
- Many SMEs see 20-30% revenue dips in Jan/Feb
- Interest rate changes affect loan costs
- GST timing can impact cash flow
"""
        )
        
        # Call the LLM
        response = self.llm.generate(
            prompt=prompt,
            system_prompt=self._get_system_prompt(),
            temperature=0.3
        )
        
        # Parse the response
        scenarios = self._parse_response(response)
        
        # Add metadata
        scenarios["_agent"] = self.name
        
        scenario_count = len(scenarios.get("scenarios", []))
        self._log(f"Scenario analysis complete. Generated {scenario_count} scenarios")
        
        return scenarios
    
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
                    "error": "Could not parse scenario response",
                    "raw_response": response,
                    "scenarios": []
                }
        
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            return {
                "error": f"JSON parsing error: {str(e)}",
                "raw_response": response,
                "scenarios": []
            }
    
    def _get_system_prompt(self) -> str:
        """Custom system prompt for scenario analysis."""
        return """You are a financial scenario analyst for New Zealand small businesses.

Your job is to create realistic "what if" scenarios that help business owners understand their financial resilience.

Key principles:
1. USE REAL NUMBERS: Base calculations on the actual data provided
2. SHOW YOUR WORK: Make calculations transparent and verifiable
3. BE REALISTIC: Create scenarios that could actually happen
4. BE HELPFUL: Focus on scenarios the owner can prepare for
5. EXPLICIT ASSUMPTIONS: State every assumption clearly

Calculation approach:
- Work in monthly figures for consistency
- Round to sensible amounts (don't show false precision)
- Cash runway = Current cash ÷ Monthly net burn
- Always sense-check results

Remember: The goal is understanding, not fear. Help the owner see their options."""
