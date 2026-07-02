"""
Data Normalizer Agent

This agent is the first in the pipeline. It takes raw financial data 
(from Xero, MYOB, Excel exports, etc.) and converts it into a clean,
standardised format that other agents can work with.

Think of it as the "translator" that turns messy spreadsheet data into 
structured information.
"""

import json
import re
from typing import Any, Dict
from agents.base_agent import BaseAgent


class DataNormalizerAgent(BaseAgent):
    """
    Cleans and standardises SME financial data.
    
    Input: Raw text/CSV data from accounting software
    Output: Structured financial summary with consistent formatting
    """
    
    def __init__(self, llm_provider: str = "claude"):
        super().__init__(
            name="Data Normalizer",
            description="Cleans and standardises raw financial data into a consistent format",
            llm_provider=llm_provider
        )
    
    def run(self, input_data: str) -> Dict[str, Any]:
        """
        Process raw financial data into a normalised structure.
        
        Args:
            input_data: Raw text containing financial data
        
        Returns:
            Dictionary with normalised financial data
        """
        self._log("Starting data normalisation...")
        
        # Build the prompt for the LLM
        prompt = self._build_prompt(
            task_description="""
Analyse the raw financial data below and extract key financial information.
Standardise all values to NZD and organise them into clear categories.
If any data is missing or ambiguous, note it explicitly.
""",
            input_data=input_data,
            output_format="""
Return a JSON object with this exact structure:
{
    "business_name": "string or 'Unknown'",
    "period": "string describing the time period covered",
    "currency": "NZD",
    
    "revenue": {
        "total": number,
        "breakdown": [{"source": "string", "amount": number}],
        "notes": "any relevant notes or data quality issues"
    },
    
    "expenses": {
        "total": number,
        "breakdown": [{"category": "string", "amount": number}],
        "notes": "any relevant notes or data quality issues"
    },
    
    "profit": {
        "gross_profit": number or null,
        "net_profit": number or null,
        "gross_margin_percent": number or null,
        "net_margin_percent": number or null
    },
    
    "cash_flow": {
        "opening_balance": number or null,
        "closing_balance": number or null,
        "net_change": number or null,
        "notes": "any relevant notes"
    },
    
    "assets_liabilities": {
        "total_assets": number or null,
        "total_liabilities": number or null,
        "equity": number or null
    },
    
    "data_quality": {
        "completeness": "HIGH/MEDIUM/LOW",
        "issues": ["list of any data quality issues found"],
        "assumptions_made": ["list of any assumptions made during normalisation"]
    },
    
    "raw_data_summary": "Brief description of what data was provided"
}
""",
            additional_instructions="""
- Convert all currency values to NZD (assume NZD if not specified)
- Use null for any values that cannot be determined
- Be conservative - don't guess at numbers
- Note any calculations you perform (e.g., "Net profit calculated as revenue minus expenses")
- If the data appears to be for a specific month/quarter/year, note this
- Common NZ accounting terms: GST (15%), PAYE (employee tax)
"""
        )
        
        # Call the LLM
        response = self.llm.generate(
            prompt=prompt,
            system_prompt=self._get_system_prompt(),
            temperature=0.1  # Low temperature for consistent, factual extraction
        )
        
        # Parse the response
        normalised_data = self._parse_response(response)
        
        # Add metadata
        normalised_data["_agent"] = self.name
        normalised_data["_status"] = "success" if normalised_data.get("revenue") else "partial"
        
        self._log(f"Normalisation complete. Data quality: {normalised_data.get('data_quality', {}).get('completeness', 'Unknown')}")
        
        return normalised_data
    
    def _parse_response(self, response: str) -> Dict[str, Any]:
        """
        Extract JSON from the LLM response.
        
        The LLM might wrap the JSON in markdown code blocks or add explanation,
        so we need to extract just the JSON part.
        """
        
        # Try to find JSON in the response
        # Look for content between ```json and ``` or just { and }
        
        # First, try to find a JSON code block
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response, re.DOTALL)
        
        if json_match:
            json_str = json_match.group(1)
        else:
            # Try to find raw JSON (first { to last })
            start = response.find('{')
            end = response.rfind('}')
            if start != -1 and end != -1:
                json_str = response[start:end + 1]
            else:
                # If no JSON found, return an error structure
                return {
                    "error": "Could not parse LLM response",
                    "raw_response": response,
                    "data_quality": {
                        "completeness": "LOW",
                        "issues": ["Failed to extract structured data from response"]
                    }
                }
        
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            return {
                "error": f"JSON parsing error: {str(e)}",
                "raw_response": response,
                "data_quality": {
                    "completeness": "LOW",
                    "issues": ["JSON parsing failed"]
                }
            }
    
    def _get_system_prompt(self) -> str:
        """Custom system prompt for data normalisation."""
        return """You are a financial data normalisation specialist.

Your job is to take messy, unstructured financial data and convert it into a clean, standardised JSON format.

Key principles:
1. ACCURACY: Only extract data that is clearly present. Never invent numbers.
2. CONSISTENCY: All monetary values in NZD. Use consistent naming.
3. TRANSPARENCY: Clearly note any assumptions or data quality issues.
4. COMPLETENESS: Extract as much useful information as possible.

You work with New Zealand small businesses, so be familiar with:
- NZD currency
- GST (15% goods and services tax)
- Common accounting software formats (Xero, MYOB)
- Financial year often April-March or April-March

Always return valid JSON that matches the requested format exactly."""
