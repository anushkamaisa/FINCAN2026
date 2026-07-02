"""
Finance Analysis Agent

This agent takes normalised financial data and performs analysis to understand
the business's financial health. It looks at:
- Revenue patterns and trends
- Expense structure
- Profit margins
- Cash flow health

Output is structured insights that feed into risk detection and recommendations.
"""

import json
import re
from typing import Any, Dict
from agents.base_agent import BaseAgent


class FinanceAnalysisAgent(BaseAgent):
    """
    Analyses financial performance metrics.
    
    Input: Normalised financial data from DataNormalizerAgent
    Output: Performance analysis with trends and insights
    """
    
    def __init__(self, llm_provider: str = "claude"):
        super().__init__(
            name="Finance Analyst",
            description="Analyses financial performance including revenue, margins, expenses, and cash flow",
            llm_provider=llm_provider
        )
    
    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyse financial data and produce insights.
        
        Args:
            input_data: Normalised financial data dictionary
        
        Returns:
            Dictionary containing financial analysis
        """
        self._log("Starting financial analysis...")
        
        # Convert input to readable string for the prompt
        data_str = json.dumps(input_data, indent=2)
        
        prompt = self._build_prompt(
            task_description="""
Analyse the financial data below to assess the business's financial health.
Focus on practical insights relevant to a small NZ business owner.
Use the actual numbers provided - do not invent data.
""",
            input_data=data_str,
            output_format="""
Return a JSON object with this exact structure:
{
    "summary": {
        "overall_health": "HEALTHY/MODERATE/CONCERNING",
        "one_line_summary": "Single sentence summarising financial position"
    },
    
    "revenue_analysis": {
        "total_revenue": number,
        "revenue_concentration": "HIGH/MEDIUM/LOW (how dependent on single sources)",
        "concentration_details": "explanation of revenue sources",
        "insights": ["list of key revenue observations"]
    },
    
    "expense_analysis": {
        "total_expenses": number,
        "expense_ratio": number (expenses as % of revenue),
        "largest_expenses": [{"category": "string", "amount": number, "percent_of_total": number}],
        "fixed_vs_variable": {
            "assessment": "description of cost structure flexibility",
            "fixed_expenses_estimate": number or null,
            "variable_expenses_estimate": number or null
        },
        "insights": ["list of key expense observations"]
    },
    
    "profitability_analysis": {
        "gross_margin_percent": number or null,
        "net_margin_percent": number or null,
        "margin_assessment": "STRONG/ADEQUATE/THIN/NEGATIVE",
        "insights": ["list of profitability observations"]
    },
    
    "cash_flow_analysis": {
        "cash_position": "STRONG/ADEQUATE/TIGHT/CRITICAL",
        "monthly_burn_rate_estimate": number or null,
        "cash_runway_months": number or null,
        "insights": ["list of cash flow observations"]
    },
    
    "key_metrics": {
        "revenue_per_employee_estimate": number or null,
        "break_even_estimate": number or null,
        "other_relevant_metrics": {}
    },
    
    "trends_identified": [
        {
            "trend": "description of the trend",
            "direction": "POSITIVE/NEGATIVE/NEUTRAL",
            "significance": "HIGH/MEDIUM/LOW"
        }
    ],
    
    "data_limitations": ["List any limitations in the analysis due to missing data"]
}
""",
            additional_instructions="""
Analysis guidelines:
- Be specific: use actual numbers, not vague statements
- For NZ SMEs, typical healthy net margins are 5-15% depending on industry
- If calculating estimates, show your reasoning
- Note when you're making assumptions due to limited data
- Focus on actionable insights, not just observations
- Consider seasonal patterns common in NZ (summer holidays Dec-Jan, etc.)
"""
        )
        
        # Call the LLM
        response = self.llm.generate(
            prompt=prompt,
            system_prompt=self._get_system_prompt(),
            temperature=0.2
        )
        
        # Parse the response
        analysis = self._parse_response(response)
        
        # Add metadata
        analysis["_agent"] = self.name
        analysis["_input_data_quality"] = input_data.get("data_quality", {}).get("completeness", "Unknown")
        
        health = analysis.get("summary", {}).get("overall_health", "Unknown")
        self._log(f"Analysis complete. Overall health: {health}")
        
        return analysis
    
    def _parse_response(self, response: str) -> Dict[str, Any]:
        """Extract JSON from the LLM response."""
        
        # Look for JSON in code blocks first
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
                    "error": "Could not parse analysis response",
                    "raw_response": response
                }
        
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            return {
                "error": f"JSON parsing error: {str(e)}",
                "raw_response": response
            }
    
    def _get_system_prompt(self) -> str:
        """Custom system prompt for financial analysis."""
        return """You are a financial analyst specialising in New Zealand small businesses.

Your expertise includes:
- Analysing P&L statements, balance sheets, and cash flow
- Identifying trends and patterns in financial data
- Understanding the unique challenges of NZ SMEs (1-10 employees)
- Practical, actionable financial insights

Key principles:
1. Use actual numbers from the data - never fabricate figures
2. Clearly state when you're estimating or making assumptions
3. Focus on insights that matter for small business owners
4. Avoid technical jargon - explain in plain terms
5. Consider NZ business context (seasonality, common industries)

Remember: This is analysis only, not accounting or tax advice."""
