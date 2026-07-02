"""
LLM Model Abstraction Layer

This module provides a unified interface for interacting with different LLM providers.
Currently supports: OpenAI (GPT), Claude (Anthropic), Gemini (Google), and NVIDIA Nemotron.

To switch providers, change the 'provider' parameter when creating the LLMClient.

SETUP:
1. Create a .env file in the project root
2. Add your API key(s):
   OPENAI_API_KEY=your-key-here
   ANTHROPIC_API_KEY=your-key-here
   GOOGLE_API_KEY=your-key-here
   NVIDIA_API_KEY=your-key-here
"""

import os
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
class LLMClient:
    """
    A simple wrapper that provides one consistent way to call any LLM.
    
    Usage:
        client = LLMClient(provider="openai")
        response = client.generate("Analyse this data...")
    """
    
    def __init__(self, provider: str = "openai", model: Optional[str] = None):
        """
        Initialise the LLM client.
        
        Args:
                provider: Which LLM to use - "openai", "claude", "gemini", or "nvidia"
                model: Specific model name (optional, uses sensible defaults)
        """
        self.provider = provider.lower()
        
        # Set default models for each provider
        if model:
            self.model = model
        elif self.provider == "claude":
            self.model = "claude-sonnet-4-20250514"
        elif self.provider == "openai":
            self.model = "gpt-4o"
        elif self.provider == "gemini":
            self.model = "models/gemini-2.0-flash"
        elif self.provider == "nvidia":
            self.model = "nemotron-3-super"
        else:
            raise ValueError(
                f"Unknown provider: {provider}. Use 'claude', 'openai', 'gemini', or 'nvidia'."
            )
        # Initialise the appropriate client
        self._client = self._create_client()
    
    def _create_client(self):
        """Create the underlying API client based on provider."""
        
        if self.provider == "claude":
            # Check for API key
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError(
                    "ANTHROPIC_API_KEY not found. "
                    "Create a .env file with: ANTHROPIC_API_KEY=your-key-here"
                )
            
            try:
                from anthropic import Anthropic
            except ImportError as exc:
                raise ImportError(
                    "Anthropic SDK is not installed. Install it with:\n"
                    "  pip install anthropic\n"
                    "or install all project dependencies:\n"
                    "  pip install -r requirements.txt"
                ) from exc
            
            return Anthropic(api_key=api_key)
        
        elif self.provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError(
                    "OPENAI_API_KEY not found. "
                    "Create a .env file with: OPENAI_API_KEY=your-key-here"
                )
            
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise ImportError(
                    "OpenAI SDK is not installed. Install it with:\n"
                    "  pip install openai\n"
                    "or install all project dependencies:\n"
                    "  pip install -r requirements.txt"
                ) from exc

            return OpenAI(api_key=api_key)
        
        elif self.provider == "gemini":
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError(
                    "GOOGLE_API_KEY not found. "
                    "Create a .env file with: GOOGLE_API_KEY=your-key-here"
                )
            
            try:
                import google.genai as genai
            except ImportError as exc:
                raise ImportError(
                    "Google Gemini SDK is not installed. Install it with:\n"
                    "  pip install google-genai\n"
                    "or install all project dependencies:\n"
                    "  pip install -r requirements.txt"
                ) from exc

            client = genai.Client(api_key=api_key)
            return client

        elif self.provider == "nvidia":
            api_key = os.getenv("NVIDIA_API_KEY")
            if not api_key:
                raise ValueError(
                    "NVIDIA_API_KEY not found. "
                    "Create a .env file with: NVIDIA_API_KEY=your-key-here"
                )
            
            # Placeholder client object for NVIDIA Nemotron support.
            # Replace this with a real NVIDIA API client integration as needed.
            return {"api_key": api_key}
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.3
    ) -> str:
        """
        Generate a response from the LLM.
        
        Args:
            prompt: The main prompt/question to send
            system_prompt: Optional system-level instructions
            max_tokens: Maximum length of response
            temperature: Creativity level (0.0 = focused, 1.0 = creative)
        
        Returns:
            The LLM's response as a string
        """
        
        if self.provider == "claude":
            return self._generate_claude(prompt, system_prompt, max_tokens, temperature)
        elif self.provider == "openai":
            return self._generate_openai(prompt, system_prompt, max_tokens, temperature)
        elif self.provider == "gemini":
            return self._generate_gemini(prompt, system_prompt, max_tokens, temperature)
        elif self.provider == "nvidia":
            return self._generate_nvidia(prompt, system_prompt, max_tokens, temperature)
    
    def _generate_claude(
        self,
        prompt: str,
        system_prompt: Optional[str],
        max_tokens: int,
        temperature: float
    ) -> str:
        """Handle Claude API calls."""
        
        message = self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_prompt or "You are a helpful assistant.",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        # Extract text from response
        return message.content[0].text
    
    def _generate_openai(
        self,
        prompt: str,
        system_prompt: Optional[str],
        max_tokens: int,
        temperature: float
    ) -> str:
        """Handle OpenAI API calls."""
        
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        response = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        return response.choices[0].message.content
    
    def _generate_gemini(
        self,
        prompt: str,
        system_prompt: Optional[str],
        max_tokens: int,
        temperature: float
    ) -> str:
        """Handle Gemini API calls."""
        
        # Create the full prompt with system instructions if provided
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"System: {system_prompt}\n\nUser: {prompt}"
        
        try:
            response = self._client.models.generate_content(
                model=self.model,
                contents=full_prompt,
                config={
                    'temperature': temperature,
                    'max_output_tokens': max_tokens,
                }
            )
            
            if response and hasattr(response, 'text'):
                return response.text
            else:
                raise RuntimeError(f"Invalid response from Gemini API: {response}")
                
        except Exception as e:
            raise RuntimeError(f"Gemini API call failed: {str(e)}") from e

    def _generate_nvidia(
        self,
        prompt: str,
        system_prompt: Optional[str],
        max_tokens: int,
        temperature: float
    ) -> str:
        """Handle NVIDIA Nemotron API calls via NIM or compatible endpoint."""
        import requests
        import os
        
        # Get API endpoint from environment or use defaults
        api_url = os.getenv(
            "NVIDIA_API_URL",
            "https://integrate.api.nvidia.com/v1/chat/completions"
        )
        
        # Prepare headers with authorization
        headers = {
            "Authorization": f"Bearer {self._client['api_key']}",
            "Content-Type": "application/json"
        }
        
        # Build messages list following OpenAI format
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        # Prepare request payload using OpenAI-compatible format
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": 1.0
        }
        
        try:
            # Make the API request with flexible SSL handling
            # Use verify=False for development/lab environments
            verify_ssl = not api_url.startswith("http://")  # Only verify for HTTPS
            
            response = requests.post(
                api_url, 
                headers=headers, 
                json=payload, 
                timeout=60,
                verify=verify_ssl
            )
            response.raise_for_status()
            
            # Parse response in OpenAI format
            result = response.json()
            if isinstance(result, dict):
                # Standard chat completion response
                if "choices" in result and len(result["choices"]) > 0:
                    choice = result["choices"][0]
                    if "message" in choice and "content" in choice["message"]:
                        return choice["message"]["content"]
                    elif "text" in choice:
                        return choice["text"]
                # Direct text response
                elif "text" in result:
                    return result["text"]
            
            raise RuntimeError(f"Invalid response format from NVIDIA API: {result}")
                
        except requests.exceptions.RequestException as e:
            # Provide helpful error message with endpoint info
            raise RuntimeError(
                f"NVIDIA API request failed: {str(e)}\n"
                f"Endpoint: {api_url}\n"
                f"Model: {self.model}\n"
                f"Note: Set NVIDIA_API_URL environment variable to use a different endpoint."
            ) from e
        except Exception as e:
            raise RuntimeError(f"NVIDIA Nemotron API call failed: {str(e)}") from e


# ----- Convenience function for simple use cases -----

def ask_llm(
    prompt: str,
    system_prompt: Optional[str] = None,
    provider: str = "openai"
) -> str:
    """
    Quick way to ask the LLM a question without managing a client object.
    
    Usage:
        response = ask_llm("What is 2+2?")
    """
    client = LLMClient(provider=provider)
    return client.generate(prompt, system_prompt)
