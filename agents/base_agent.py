"""
Base Agent Class

All agents in the system inherit from this class.
It provides:
- A consistent interface (every agent has a 'run' method)
- Access to the LLM
- Standard prompt construction helpers
- Output validation helpers

Think of this as the "template" that all agents follow.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from llm.model import LLMClient


class BaseAgent(ABC):
    """
    Abstract base class for all agents.
    
    Every agent must implement the 'run' method which:
    - Takes some input data
    - Processes it (usually by calling the LLM)
    - Returns structured output
    """
    
    def __init__(
        self,
        name: str,
        description: str,
        llm_provider: str = "claude"
    ):
        """
        Initialise the agent.
        
        Args:
            name: Human-readable name for logging/debugging
            description: What this agent does (one sentence)
            llm_provider: Which LLM to use ("claude" or "openai")
        """
        self.name = name
        self.description = description
        self.llm = LLMClient(provider=llm_provider)
    
    @abstractmethod
    def run(self, input_data: Any) -> Dict[str, Any]:
        """
        Execute the agent's main task.
        
        This method must be implemented by each specific agent.
        
        Args:
            input_data: The data this agent needs to do its job
        
        Returns:
            A dictionary containing the agent's output
        """
        pass
    
    def _build_prompt(
        self,
        task_description: str,
        input_data: str,
        output_format: str,
        additional_instructions: Optional[str] = None
    ) -> str:
        """
        Helper to build well-structured prompts.
        
        Keeps prompts consistent and readable across all agents.
        
        Args:
            task_description: What the LLM should do
            input_data: The data to analyse
            output_format: How the output should be structured
            additional_instructions: Any extra rules or context
        
        Returns:
            A formatted prompt string
        """
        
        prompt_parts = [
            "## Task",
            task_description,
            "",
            "## Input Data",
            input_data,
            "",
            "## Required Output Format",
            output_format
        ]
        
        if additional_instructions:
            prompt_parts.extend([
                "",
                "## Additional Instructions",
                additional_instructions
            ])
        
        return "\n".join(prompt_parts)
    
    def _get_system_prompt(self) -> str:
        """
        Returns the system prompt for this agent.
        
        Override in subclasses for agent-specific behaviour.
        """
        return f"""You are {self.name}, an AI agent that is part of a financial analysis system.

Your role: {self.description}

Important rules:
1. Only use the data provided - never invent or assume financial figures
2. If data is missing or unclear, explicitly state this
3. Be specific and use actual numbers from the input
4. Do not provide tax compliance advice (GST, PAYE, etc.) - analysis only
5. Focus on practical insights for small NZ businesses (1-10 employees)
6. Use clear, non-technical language where possible"""
    
    def _validate_output(self, output: Dict[str, Any], required_keys: list) -> bool:
        """
        Check that the output contains all required fields.
        
        Args:
            output: The dictionary to validate
            required_keys: List of keys that must be present
        
        Returns:
            True if valid, False otherwise
        """
        for key in required_keys:
            if key not in output:
                print(f"WARNING: {self.name} output missing required key: {key}")
                return False
        return True
    
    def _log(self, message: str):
        """Simple logging helper."""
        print(f"[{self.name}] {message}")
