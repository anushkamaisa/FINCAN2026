"""
Recommendation Agent

This agent takes everything learned by previous agents and produces
3-5 clear, prioritised action items for the business owner.

Each recommendation must:
- Link directly to specific analysis findings
- Be actionable (not vague advice)
- Be appropriate for a small NZ business
"""

import json
import re
from typing import Any, Dict
from agents.base_agent import BaseAgent


class RecommendationAgent(BaseAgent):
    """
    Produces prioritised recommendations based on analysis.
    
    Input: All outputs from previous agents
    Output: 3-5 actionable recommendations with rationale
    """
    
    def __init__(self, llm_provider: str = "claude"):
        super().__init__(
            name="Recommendation Engine",
            description="Produces clear, prioritised action items linked to specific analysis findings",
            llm_provider=llm_provider
        )
    
    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate recommendations from all analysis.
        
        Args:
            input_data: Dictionary containing:
                - normalised_data: From DataNormalizerAgent
                - finance_analysis: From FinanceAnalysisAgent
                - risk_assessment: From RiskDetectionAgent
                - scenario_analysis: From ScenarioAnalysisAgent
        
        Returns:
            Dictionary containing prioritised recommendations
        """
        self._log("Generating recommendations...")
        
        # Combine all inputs
        combined_data = {
            "financial_data": input_data.get("normalised_data", {}),
            "performance_analysis": input_data.get("finance_analysis", {}),
            "risks_identified": input_data.get("risk_assessment", {}),
            "scenario_projections": input_data.get("scenario_analysis", {})
        }
        data_str = json.dumps(combined_data, indent=2)
        
        prompt = self._build_prompt(
            task_description="""
Based on all the analysis, generate 3-5 prioritised recommendations.
Each recommendation must be specific, actionable, and tied to evidence.
Focus on what a small business owner can actually do.
""",
            input_data=data_str,
            output_format="""
Return a JSON object with this exact structure:
{
    "executive_summary": "2-3 sentences summarising the most important actions",
    
    "recommendations": [
        {
            "priority": 1,
            "title": "Clear, action-oriented title",
            "category": "CASH_FLOW/REVENUE/COSTS/RISK_MANAGEMENT/GROWTH/OPERATIONS",
            
            "what_to_do": "Specific action to take",
            "why_it_matters": "Brief explanation of the benefit",
            "evidence_basis": "Which analysis findings support this",
            
            "effort_level": "LOW/MEDIUM/HIGH",
            "time_to_implement": "IMMEDIATE/WEEKS/MONTHS",
            "expected_impact": "Quantified if possible, otherwise qualitative",
            
            "specific_steps": [
                "Step 1: ...",
                "Step 2: ...",
                "Step 3: ..."
            ],
            
            "success_metric": "How to know if this worked",
            "potential_obstacles": ["Things that might make this difficult"]
        }
    ],
    
    "quick_wins": [
        {
            "action": "Simple thing that can be done today/this week",
            "benefit": "What it achieves"
        }
    ],
    
    "not_recommended_right_now": [
        {
            "action": "Something that might seem appealing but isn't right for now",
            "reason": "Why to avoid it"
        }
    ],
    
    "monitoring_checklist": [
        {
            "metric": "What to track",
            "frequency": "How often",
            "target": "What good looks like"
        }
    ]
}
""",
            additional_instructions="""
Recommendation principles:

1. PRIORITISE RUTHLESSLY
   - Put cash flow and survival issues first
   - Don't overwhelm with too many actions
   - 3-5 recommendations is the sweet spot

2. BE SPECIFIC
   - Bad: "Reduce costs"
   - Good: "Review software subscriptions - you're spending $X/month. Cancel unused tools."

3. LINK TO EVIDENCE
   - Every recommendation must trace back to a specific finding
   - Don't give generic business advice

4. MATCH THE BUSINESS SIZE
   - These are 1-10 employee businesses
   - No resources for complex initiatives
   - Owner probably wears multiple hats

5. CONSIDER NZ CONTEXT
   - Bank relationships matter
   - Local supplier options
   - Industry norms

6. BE HONEST ABOUT TRADE-OFFS
   - What's the cost of implementing?
   - What won't get done if they focus here?
"""
        )
        
        # Call the LLM
        response = self.llm.generate(
            prompt=prompt,
            system_prompt=self._get_system_prompt(),
            temperature=0.3
        )
        
        # Parse the response
        recommendations = self._parse_response(response)
        
        # Add metadata
        recommendations["_agent"] = self.name
        
        rec_count = len(recommendations.get("recommendations", []))
        self._log(f"Generated {rec_count} recommendations")
        
        return recommendations
    
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
                    "error": "Could not parse recommendations response",
                    "raw_response": response,
                    "recommendations": []
                }
        
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            return {
                "error": f"JSON parsing error: {str(e)}",
                "raw_response": response,
                "recommendations": []
            }
    
    def _get_system_prompt(self) -> str:
        """Custom system prompt for recommendations."""
        return """You are a business advisor specialising in New Zealand small businesses.

Your job is to turn financial analysis into clear, actionable recommendations.

Key principles:
1. PRACTICAL: Recommendations must work for businesses with 1-10 people
2. SPECIFIC: Give actual actions, not vague guidance
3. PRIORITISED: Most important things first
4. EVIDENCE-BASED: Every recommendation tied to analysis findings
5. HONEST: Acknowledge trade-offs and limitations

Think like a trusted advisor who:
- Has seen many small businesses succeed and fail
- Knows what's realistic for an owner-operator
- Focuses on what moves the needle
- Doesn't waste time on minor optimisations when big issues exist

Avoid:
- Generic business platitudes
- Recommendations requiring significant capital
- Complex multi-year strategies
- Anything requiring specialised staff they don't have"""
