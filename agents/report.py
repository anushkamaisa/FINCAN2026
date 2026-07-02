"""
Report Agent

This agent is the final step. It takes all the technical analysis and
converts it into a clear, readable report for the business owner.

The report should:
- Use plain English (no finance jargon)
- Be structured logically
- Highlight what matters most
- Be appropriate for a busy SME owner
"""

import json
import re
from typing import Any, Dict
from agents.base_agent import BaseAgent


class ReportAgent(BaseAgent):
    """
    Generates a plain-English client report.
    
    Input: All outputs from previous agents
    Output: Human-readable report suitable for SME owners
    """
    
    def __init__(self, llm_provider: str = "claude"):
        super().__init__(
            name="Report Writer",
            description="Converts technical analysis into a clear, plain-English report for business owners",
            llm_provider=llm_provider
        )
    
    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a client-friendly report.
        
        Args:
            input_data: Dictionary containing all previous agent outputs
        
        Returns:
            Dictionary containing the formatted report
        """
        self._log("Generating client report...")
        
        # Combine all inputs
        combined_data = {
            "business_data": input_data.get("normalised_data", {}),
            "performance_analysis": input_data.get("finance_analysis", {}),
            "risk_assessment": input_data.get("risk_assessment", {}),
            "scenarios": input_data.get("scenario_analysis", {}),
            "recommendations": input_data.get("recommendations", {})
        }
        data_str = json.dumps(combined_data, indent=2)
        
        prompt = self._build_prompt(
            task_description="""
Create a clear, readable financial report for a small business owner.
Use plain English - avoid technical jargon.
The reader is busy and needs to quickly understand their financial position.
""",
            input_data=data_str,
            output_format="""
Return a JSON object with this exact structure:
{
    "report_title": "Financial Health Report - [Business Name]",
    "report_date": "Current date",
    "period_covered": "The period this analysis covers",
    
    "executive_summary": {
        "headline": "One sentence capturing the key message",
        "overview": "2-3 paragraph summary a busy owner can read in 60 seconds",
        "health_score": "A simple rating: GREEN/AMBER/RED with explanation"
    },
    
    "your_numbers_at_a_glance": {
        "intro": "Brief explanation of what these numbers mean",
        "key_figures": [
            {
                "label": "Plain English name",
                "value": "The number",
                "context": "Is this good/bad? Compared to what?"
            }
        ]
    },
    
    "what_we_found": {
        "intro": "Opening sentence about the analysis",
        "sections": [
            {
                "heading": "Section title",
                "content": "Plain English explanation",
                "key_point": "The main takeaway"
            }
        ]
    },
    
    "risks_to_watch": {
        "intro": "Brief explanation of why this matters",
        "risks": [
            {
                "risk": "What could go wrong",
                "severity": "How serious (use colours/words, not codes)",
                "what_this_means": "Plain English impact",
                "what_you_can_do": "Simple action"
            }
        ]
    },
    
    "what_if_scenarios": {
        "intro": "Why we looked at different scenarios",
        "scenarios": [
            {
                "scenario": "What might happen",
                "impact": "What it would mean for your business",
                "key_number": "The most important figure"
            }
        ]
    },
    
    "recommended_actions": {
        "intro": "What we suggest you focus on",
        "priority_actions": [
            {
                "action": "What to do",
                "why": "Why it matters",
                "how_to_start": "First step"
            }
        ],
        "quick_wins": ["Simple things you can do this week"]
    },
    
    "next_steps": {
        "immediate": "What to do in the next 7 days",
        "this_month": "What to tackle this month",
        "ongoing": "What to keep monitoring"
    },
    
    "closing_note": "Encouraging, professional closing paragraph",
    
    "appendix": {
        "data_sources": "What data this analysis was based on",
        "limitations": "What we couldn't analyse due to missing information",
        "disclaimer": "Standard disclaimer about this being analysis, not professional advice"
    }
}
""",
            additional_instructions="""
Writing style guide:

LANGUAGE
- Write for someone without financial training
- Avoid: "liquidity ratio", "EBITDA", "working capital"
- Use instead: "cash on hand", "profit before major costs", "money available day-to-day"
- Short sentences. Short paragraphs.
- Active voice: "You earned $50,000" not "Revenue of $50,000 was recorded"

TONE
- Professional but warm
- Confident but not arrogant
- Honest about problems, but constructive
- Encouraging where genuine

STRUCTURE
- Lead with what matters most
- Use bullet points for lists
- One idea per paragraph
- Make it scannable

NZ CONTEXT
- Use NZD with $ symbol
- Familiar references (ACC, GST, IRD)
- Kiwi business norms

IMPORTANT DISCLAIMER
Include a clear note that this is:
- Financial analysis, not professional accounting or tax advice
- The user should consult a qualified accountant for compliance matters
- Not a substitute for professional financial planning
"""
        )
        
        # Call the LLM
        response = self.llm.generate(
            prompt=prompt,
            system_prompt=self._get_system_prompt(),
            temperature=0.4  # Slightly higher for natural writing
        )
        
        # Parse the response
        report = self._parse_response(response)
        
        # Add metadata
        report["_agent"] = self.name
        
        self._log("Report generation complete")
        
        return report
    
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
                    "error": "Could not parse report response",
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
        """Custom system prompt for report writing."""
        return """You are a financial writer creating reports for New Zealand small business owners.

Your job is to translate complex financial analysis into clear, actionable reports.

Your readers:
- Run businesses with 1-10 employees
- Are experts in their trade, not in finance
- Have limited time
- Need to understand their position quickly
- Want to know what to do, not just what's happening

Writing principles:
1. CLARITY: If a 15-year-old couldn't understand it, simplify it
2. BREVITY: Every word must earn its place
3. HONESTY: Don't sugarcoat problems or oversell opportunities
4. ACTION: Always connect insights to what the owner can do
5. RESPECT: Treat the reader as intelligent but time-poor

Remember: You're writing for a person who built a business with their own hands. 
They deserve straight talk, not jargon or condescension."""
    
    def format_as_text(self, report: Dict[str, Any]) -> str:
        """
        Convert the JSON report into a readable text format.
        
        This is a helper method to produce a nicely formatted text version
        of the report that can be printed or saved to a file.
        """
        if "error" in report:
            return f"Error generating report: {report.get('error')}"
        
        lines = []
        
        # Title
        lines.append("=" * 60)
        lines.append(report.get("report_title", "Financial Health Report"))
        lines.append(f"Date: {report.get('report_date', 'N/A')}")
        lines.append(f"Period: {report.get('period_covered', 'N/A')}")
        lines.append("=" * 60)
        lines.append("")
        
        # Executive Summary
        exec_summary = report.get("executive_summary", {})
        lines.append("EXECUTIVE SUMMARY")
        lines.append("-" * 40)
        if exec_summary.get("headline"):
            lines.append(f">> {exec_summary['headline']}")
            lines.append("")
        if exec_summary.get("overview"):
            lines.append(exec_summary["overview"])
            lines.append("")
        if exec_summary.get("health_score"):
            lines.append(f"Overall Health: {exec_summary['health_score']}")
        lines.append("")
        
        # Key Numbers
        numbers = report.get("your_numbers_at_a_glance", {})
        if numbers.get("key_figures"):
            lines.append("YOUR KEY NUMBERS")
            lines.append("-" * 40)
            for fig in numbers["key_figures"]:
                lines.append(f"• {fig.get('label', 'N/A')}: {fig.get('value', 'N/A')}")
                if fig.get("context"):
                    lines.append(f"  ({fig['context']})")
            lines.append("")
        
        # Recommendations
        recs = report.get("recommended_actions", {})
        if recs.get("priority_actions"):
            lines.append("RECOMMENDED ACTIONS")
            lines.append("-" * 40)
            for i, action in enumerate(recs["priority_actions"], 1):
                lines.append(f"{i}. {action.get('action', 'N/A')}")
                if action.get("why"):
                    lines.append(f"   Why: {action['why']}")
                if action.get("how_to_start"):
                    lines.append(f"   Start: {action['how_to_start']}")
                lines.append("")
        
        # Next Steps
        next_steps = report.get("next_steps", {})
        if next_steps:
            lines.append("NEXT STEPS")
            lines.append("-" * 40)
            if next_steps.get("immediate"):
                lines.append(f"This Week: {next_steps['immediate']}")
            if next_steps.get("this_month"):
                lines.append(f"This Month: {next_steps['this_month']}")
            if next_steps.get("ongoing"):
                lines.append(f"Ongoing: {next_steps['ongoing']}")
            lines.append("")
        
        # Closing
        if report.get("closing_note"):
            lines.append("-" * 40)
            lines.append(report["closing_note"])
            lines.append("")
        
        # Disclaimer
        appendix = report.get("appendix", {})
        if appendix.get("disclaimer"):
            lines.append("-" * 40)
            lines.append("IMPORTANT NOTICE")
            lines.append(appendix["disclaimer"])
        
        return "\n".join(lines)
