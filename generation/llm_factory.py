"""
LLM Client Factory.
Initializes API connections for OpenAI and Anthropic completion clients.
"""

import os
import logging
from typing import Optional
from configs.config_manager import get_config

logger = logging.getLogger(__name__)

# Fallback checker
HAS_OPENAI = False
try:
    import openai
    HAS_OPENAI = True
except ImportError:
    pass

HAS_ANTHROPIC = False
try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    pass

class SimulatedLLMClient:
    """
    Local simulation client when third-party API keys are missing in test runs.
    """
    def create_completion(self, system: str, user: str) -> str:
        logger.warning("Simulated completion generator invoked.")
        # Isolate target user query to prevent few-shot keywords contamination
        query_part = user.split("Query:")[-1].lower() if "Query:" in user else user.lower()
        
        # Check standard query cues to simulate positive and negative paths
        if "optimizer" in query_part:
            return "The transformer architecture was optimized using the AdamW optimizer with a learning rate of 1e-4 [arxiv_1706_03762:methodology:4]."
        elif "accuracy" in query_part:
            return "We report a final validation accuracy of 92.4% on GLUE benchmarks [arxiv_1706_03762:results:6]."
        else:
            return "INSUFFICIENT_EVIDENCE"

class LLMFactory:
    def __init__(self):
        self.config = get_config()
        self.provider = "simulated"
        
        # Check if running in automated test mode to enforce simulated LLM client
        if os.getenv("TEST_MODE") == "true":
            logger.info("Automated test mode detected. Forcing SimulatedLLMClient.")
            return

        # Determine provider based on key configurations
        if self.config.openai_api_key:
            self.provider = "openai"
        elif self.config.anthropic_api_key:
            self.provider = "anthropic"
        elif self.config.mistral_api_key:
            self.provider = "mistral"
        else:
            logger.warning("No API credentials detected. Initializing SimulatedLLMClient.")


    def get_completion_client(self):
        """
        Returns configured client completion callable interface.
        """
        if self.provider == "openai" and HAS_OPENAI:
            try:
                client = openai.OpenAI(api_key=self.config.openai_api_key)
                def openai_caller(system: str, user: str) -> str:
                    response = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {"role": "system", "content": system},
                            {"role": "user", "content": user}
                        ],
                        temperature=0.0
                    )
                    return response.choices[0].message.content.strip()
                return openai_caller
            except Exception as e:
                logger.error(f"Failed to load OpenAI client: {e}. Falling back to simulation.")
                
        elif self.provider == "anthropic" and HAS_ANTHROPIC:
            try:
                client = anthropic.Anthropic(api_key=self.config.anthropic_api_key)
                def anthropic_caller(system: str, user: str) -> str:
                    response = client.messages.create(
                        model="claude-3-5-sonnet-20241022",
                        max_tokens=1024,
                        system=system,
                        messages=[{"role": "user", "content": user}],
                        temperature=0.0
                    )
                    return response.content[0].text.strip()
                return anthropic_caller
            except Exception as e:
                logger.error(f"Failed to load Anthropic client: {e}. Falling back to simulation.")

        elif self.provider == "mistral":
            try:
                import requests
                import time
                api_key = self.config.mistral_api_key
                def mistral_caller(system: str, user: str) -> str:
                    headers = {
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                        "Accept": "application/json"
                    }
                    payload = {
                        "model": "mistral-large-latest",
                        "messages": [
                            {"role": "system", "content": system},
                            {"role": "user", "content": user}
                        ],
                        "temperature": 0.0
                    }
                    
                    max_retries = 3
                    delay = 2.0
                    for attempt in range(max_retries):
                        response = requests.post(
                            "https://api.mistral.ai/v1/chat/completions",
                            headers=headers,
                            json=payload,
                            timeout=30
                        )
                        if response.status_code == 429:
                            logger.warning(f"Mistral API returned 429 (Rate Limit). Retrying in {delay}s (Attempt {attempt+1}/{max_retries})...")
                            time.sleep(delay)
                            delay *= 2  # Exponential backoff
                            continue
                            
                        response.raise_for_status()
                        data = response.json()
                        return data["choices"][0]["message"]["content"].strip()
                        
                    raise Exception("Mistral API call failed after multiple retries due to rate limit (429).")
                return mistral_caller
            except Exception as e:
                logger.error(f"Failed to configure Mistral client: {e}. Falling back to simulation.")
                
        # Default mock simulation fallback
        sim_client = SimulatedLLMClient()
        return sim_client.create_completion

