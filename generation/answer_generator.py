"""
Answer Generator.
Assembles prompting blocks, executes inference completions via LLM clients, 
and reformats citation tags to match logical bounds.
"""

import logging
from generation.prompt_loader import PromptLoader
from generation.llm_factory import LLMFactory
from generation.citation_formatter import CitationFormatter

logger = logging.getLogger(__name__)

class AnswerGenerator:
    def __init__(self):
        self.prompt_loader = PromptLoader()
        self.llm_factory = LLMFactory()
        self.formatter = CitationFormatter()
        
        # Initialize completion interface caller
        self.client_caller = self.llm_factory.get_completion_client()

    def generate_answer(self, query: str, context_str: str) -> str:
        """
        Compiles prompts, queries target LLM, and formats citations.
        
        Args:
            query (str): User question.
            context_str (str): Packed context chunks.
            
        Returns:
            str: Cleansed grounded text response.
        """
        logger.info(f"Generating answer for query: '{query}'...")
        
        system_prompt = self.prompt_loader.get_system_prompt()
        user_prompt = self.prompt_loader.compile_user_prompt(query, context_str)
        
        try:
            # Query LLM Completion API
            raw_response = self.client_caller(system_prompt, user_prompt)
            logger.debug(f"Raw API Completion returned: {raw_response}")
            
            # Reorder periods around brackets to prevent segmentation issues
            formatted_response = self.formatter.clean_citation_periods(raw_response)
            
            return formatted_response
            
        except Exception as e:
            logger.error(f"Error during LLM answer generation: {e}")
            return "INSUFFICIENT_EVIDENCE"
