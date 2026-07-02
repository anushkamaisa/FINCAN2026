"""
AI Finance Analyst - Main Entry Point

This is where you run the system from.

Usage:
    python main.py                    # Uses sample data with OpenAI
    python main.py path/to/data.txt   # Uses your own data file
    python main.py --provider claude  # Uses Claude instead of OpenAI
    python main.py --provider gemini  # Uses Google Gemini
"""

import sys
import json
from pathlib import Path
from datetime import datetime

# Ensure stdout can handle UTF-8 output on Windows terminals.
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

from manager.agent_manager import AgentManager


def load_financial_data(file_path: str = None) -> str:
    """
    Load financial data from a file or use the sample data.
    
    Args:
        file_path: Path to a text file containing financial data
    
    Returns:
        The financial data as a string
    """
    if file_path:
        path = Path(file_path)
        if not path.exists():
            print(f"Error: File not found: {file_path}")
            sys.exit(1)
        
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()
    
    # Use sample data if no file provided
    sample_path = Path(__file__).parent / "inputs" / "sample_financial_data.txt"
    if sample_path.exists():
        with open(sample_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    # Fallback minimal sample
    return """
    Sample Business
    Revenue: $100,000
    Expenses: $80,000
    Profit: $20,000
    Cash: $15,000
    """


def save_results(result: dict, output_dir: str = "outputs"):
    """
    Save analysis results to files.
    
    Creates:
    - Full JSON output with all agent data
    - Text report ready to share with client
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Timestamp for unique filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save full JSON result
    json_file = output_path / f"analysis_{timestamp}.json"
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, default=str)
    print(f"Full results saved to: {json_file}")
    
    # Save text report
    if result.get("report_text"):
        report_file = output_path / f"report_{timestamp}.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(result["report_text"])
        print(f"Client report saved to: {report_file}")
    
    return json_file, output_path


def print_quick_summary(result: dict):
    """Print a quick summary to the console."""
    print("\n" + "=" * 60)
    print("QUICK SUMMARY")
    print("=" * 60)
    
    # Overall health
    health = (result.get("finance_analysis", {})
              .get("summary", {})
              .get("overall_health", "Unknown"))
    print(f"\nOverall Financial Health: {health}")
    
    # Key risks
    risks = result.get("risk_assessment", {}).get("risks", [])
    high_risks = [r for r in risks if r.get("severity") == "HIGH"]
    if high_risks:
        print(f"\nHigh Priority Risks: {len(high_risks)}")
        for risk in high_risks[:3]:
            print(f"   - {risk.get('title', 'Unknown risk')}")
    
    # Top recommendations
    recs = result.get("recommendations", {}).get("recommendations", [])
    if recs:
        print(f"\nTop Recommendation:")
        top_rec = recs[0]
        print(f"   {top_rec.get('title', 'No recommendation')}")
        if top_rec.get('what_to_do'):
            print(f"   - {top_rec['what_to_do']}")
    
    print("\n" + "=" * 60)


def main():
    """Main entry point."""
    
    print("=" * 60)
    print("AI FINANCE ANALYST")
    print("Multi-Agent Financial Analysis System")
    print("=" * 60)
    
    # Parse command line arguments
    file_path = None
    provider = "openai"
    
    args = sys.argv[1:]
    i = 0
    while i < len(args):
        arg = args[i]
        if arg == "--provider" and i + 1 < len(args):
            provider = args[i + 1]
            i += 2
            continue
        elif not arg.startswith("--"):
            file_path = arg
        i += 1
    
    # Load data
    print(f"\nLoading financial data...")
    if file_path:
        print(f"   Source: {file_path}")
    else:
        print("   Source: Sample data (no file specified)")
    
    raw_data = load_financial_data(file_path)
    print(f"   Loaded {len(raw_data)} characters")
    
    # Create manager and run analysis
    print(f"\nUsing LLM provider: {provider}")
    print("\nStarting analysis...\n")
    
    try:
        manager = AgentManager(llm_provider=provider, verbose=True)
        result = manager.run_analysis(raw_data)
        
        # Save results
        print("\nSaving results...")
        save_results(result)
        
        # Print quick summary
        print_quick_summary(result)
        
        # Print the text report
        if result.get("report_text"):
            print("\n" + "=" * 60)
            print("CLIENT REPORT")
            print("=" * 60)
            print(result["report_text"])
        
        # Return success
        return 0
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        print("\nTroubleshooting tips:")
        print("1. Check your API key is set in .env file")
        print("2. Check you have internet connectivity")
        print("3. Try with --provider claude or --provider gemini if OpenAI isn't working")
        return 1


if __name__ == "__main__":
    sys.exit(main())
