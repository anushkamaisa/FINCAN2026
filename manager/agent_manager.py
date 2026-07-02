"""
Agent Manager

This is the central orchestrator of the multi-agent system.
It runs each agent in sequence, ensuring outputs flow correctly
from one agent to the next.

The flow is:
1. DataNormalizer → normalised_data
2. FinanceAnalysis → finance_analysis
3. RiskDetection → risk_assessment
4. ScenarioAnalysis → scenario_analysis
5. Recommendations → recommendations
6. Report → final_report

The Manager ensures each agent gets the data it needs and
collects the final result.
"""

from typing import Any, Dict, Optional
from datetime import datetime

# Import all agents
from agents.data_normalizer import DataNormalizerAgent
from agents.finance_analysis import FinanceAnalysisAgent
from agents.risk_detection import RiskDetectionAgent
from agents.scenario_analysis import ScenarioAnalysisAgent
from agents.recommendations import RecommendationAgent
from agents.report import ReportAgent


class AgentManager:
    """
    Orchestrates the multi-agent analysis pipeline.
    
    Usage:
        manager = AgentManager()
        result = manager.run_analysis(raw_financial_data)
    """
    
    def __init__(self, llm_provider: str = "claude", verbose: bool = True):
        """
        Initialise the manager and all agents.
        
        Args:
            llm_provider: Which LLM to use ("claude", "openai", "gemini", or "nvidia")
            verbose: Whether to print progress messages
        """
        self.llm_provider = llm_provider
        self.verbose = verbose
        
        # Create all agents
        self._log("Initialising agents...")
        
        self.data_normalizer = DataNormalizerAgent(llm_provider)
        self.finance_analyst = FinanceAnalysisAgent(llm_provider)
        self.risk_detector = RiskDetectionAgent(llm_provider)
        self.scenario_analyst = ScenarioAnalysisAgent(llm_provider)
        self.recommendation_engine = RecommendationAgent(llm_provider)
        self.report_writer = ReportAgent(llm_provider)
        
        self._log("All agents initialised")
    
    def run_analysis(self, raw_data: str) -> Dict[str, Any]:
        """
        Run the complete analysis pipeline.
        
        This is the main entry point. Give it raw financial data,
        and it will return a complete analysis with report.
        
        Args:
            raw_data: Raw financial data as text (CSV, exported text, etc.)
        
        Returns:
            Dictionary containing:
                - normalised_data: Cleaned financial data
                - finance_analysis: Performance analysis
                - risk_assessment: Risk evaluation
                - scenario_analysis: What-if scenarios
                - recommendations: Action items
                - report: Plain-English report
                - metadata: Information about the run
        """
        self._log("=" * 50)
        self._log("STARTING FINANCIAL ANALYSIS PIPELINE")
        self._log("=" * 50)
        
        start_time = datetime.now()
        
        # Initialise result container
        result = {
            "metadata": {
                "started_at": start_time.isoformat(),
                "llm_provider": self.llm_provider,
                "pipeline_version": "1.0.0"
            }
        }
        
        # Track any errors
        errors = []
        
        # ==========================================
        # STEP 1: Data Normalisation
        # ==========================================
        self._log("\n📊 STEP 1/6: Normalising data...")
        try:
            normalised_data = self.data_normalizer.run(raw_data)
            result["normalised_data"] = normalised_data
            
            if "error" in normalised_data:
                errors.append(f"Data normalisation: {normalised_data['error']}")
        except Exception as e:
            errors.append(f"Data normalisation failed: {str(e)}")
            result["normalised_data"] = {"error": str(e)}
        
        # ==========================================
        # STEP 2: Financial Analysis
        # ==========================================
        self._log("\n📈 STEP 2/6: Analysing financial performance...")
        try:
            finance_analysis = self.finance_analyst.run(
                result["normalised_data"]
            )
            result["finance_analysis"] = finance_analysis
            
            if "error" in finance_analysis:
                errors.append(f"Financial analysis: {finance_analysis['error']}")
        except Exception as e:
            errors.append(f"Financial analysis failed: {str(e)}")
            result["finance_analysis"] = {"error": str(e)}
        
        # ==========================================
        # STEP 3: Risk Detection
        # ==========================================
        self._log("\n⚠️ STEP 3/6: Detecting risks...")
        try:
            risk_assessment = self.risk_detector.run({
                "normalised_data": result["normalised_data"],
                "finance_analysis": result["finance_analysis"]
            })
            result["risk_assessment"] = risk_assessment
            
            if "error" in risk_assessment:
                errors.append(f"Risk detection: {risk_assessment['error']}")
        except Exception as e:
            errors.append(f"Risk detection failed: {str(e)}")
            result["risk_assessment"] = {"error": str(e)}
        
        # ==========================================
        # STEP 4: Scenario Analysis
        # ==========================================
        self._log("\n🔮 STEP 4/6: Running scenario analysis...")
        try:
            scenario_analysis = self.scenario_analyst.run({
                "normalised_data": result["normalised_data"],
                "finance_analysis": result["finance_analysis"],
                "risk_assessment": result["risk_assessment"]
            })
            result["scenario_analysis"] = scenario_analysis
            
            if "error" in scenario_analysis:
                errors.append(f"Scenario analysis: {scenario_analysis['error']}")
        except Exception as e:
            errors.append(f"Scenario analysis failed: {str(e)}")
            result["scenario_analysis"] = {"error": str(e)}
        
        # ==========================================
        # STEP 5: Recommendations
        # ==========================================
        self._log("\n💡 STEP 5/6: Generating recommendations...")
        try:
            recommendations = self.recommendation_engine.run({
                "normalised_data": result["normalised_data"],
                "finance_analysis": result["finance_analysis"],
                "risk_assessment": result["risk_assessment"],
                "scenario_analysis": result["scenario_analysis"]
            })
            result["recommendations"] = recommendations
            
            if "error" in recommendations:
                errors.append(f"Recommendations: {recommendations['error']}")
        except Exception as e:
            errors.append(f"Recommendations failed: {str(e)}")
            result["recommendations"] = {"error": str(e)}
        
        # ==========================================
        # STEP 6: Report Generation
        # ==========================================
        self._log("\n📝 STEP 6/6: Writing report...")
        try:
            report = self.report_writer.run({
                "normalised_data": result["normalised_data"],
                "finance_analysis": result["finance_analysis"],
                "risk_assessment": result["risk_assessment"],
                "scenario_analysis": result["scenario_analysis"],
                "recommendations": result["recommendations"]
            })
            result["report"] = report
            
            # Also generate a text version of the report
            result["report_text"] = self.report_writer.format_as_text(report)
            
            if "error" in report:
                errors.append(f"Report generation: {report['error']}")
        except Exception as e:
            errors.append(f"Report generation failed: {str(e)}")
            result["report"] = {"error": str(e)}
        
        # ==========================================
        # Finalise
        # ==========================================
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        result["metadata"]["completed_at"] = end_time.isoformat()
        result["metadata"]["duration_seconds"] = duration
        result["metadata"]["errors"] = errors
        result["metadata"]["success"] = len(errors) == 0
        
        self._log("\n" + "=" * 50)
        self._log(f"ANALYSIS COMPLETE")
        self._log(f"Duration: {duration:.1f} seconds")
        if errors:
            self._log(f"Errors: {len(errors)}")
            for error in errors:
                self._log(f"  - {error}")
        else:
            self._log("Status: SUCCESS ✓")
        self._log("=" * 50)
        
        return result
    
    def run_single_agent(
        self,
        agent_name: str,
        input_data: Any
    ) -> Dict[str, Any]:
        """
        Run a single agent for testing or targeted analysis.
        
        Args:
            agent_name: Name of agent to run
            input_data: Input data for that agent
        
        Returns:
            Agent output
        """
        agents = {
            "normalizer": self.data_normalizer,
            "finance": self.finance_analyst,
            "risk": self.risk_detector,
            "scenario": self.scenario_analyst,
            "recommendations": self.recommendation_engine,
            "report": self.report_writer
        }
        
        if agent_name not in agents:
            return {"error": f"Unknown agent: {agent_name}. Available: {list(agents.keys())}"}
        
        return agents[agent_name].run(input_data)
    
    def _log(self, message: str):
        """Print message if verbose mode is on."""
        if self.verbose:
            print(f"[Manager] {message}")
