"""
Generation Prompts Loader.
Contains instructions and templates for grounding answers and validating citations.
"""

import logging

logger = logging.getLogger(__name__)

SYSTEM_GROUNDING_INSTRUCTION = """You are a Research Intelligence Assistant. Analyze the query using ONLY the provided text segments.

OPERATIONAL PRINCIPLES:
1. Every factual assertion must be directly grounded in one of the provided sources.
2. For each claim you make, append the corresponding citation token in format: [PaperID:SectionName:PageNumber].
3. The period MUST be placed AFTER the citation bracket, e.g.: "This model achieves high accuracy [arxiv_1706_03762:results:6]."
4. If the provided sources do not contain sufficient evidence to answer the query, output: "INSUFFICIENT_EVIDENCE".
5. Do not use external knowledge or pre-training facts.
6. If some parts of the answer are supported and others are not, output ONLY the supported parts with citations.
7. Do not guess, speculate, or fabricate details.
"""

FEW_SHOT_EXAMPLES = """
--- EXAMPLES OF GROUNDED ANSWERS ---

Example 1:
Query: What learning rate was used for training?
Context: 
--- SOURCE: [arxiv_1706_03762:methodology:4] ---
We train our model using the AdamW optimizer with a base learning rate of 1e-4 and a batch size of 256.
-------------------------------------
Response: The model was trained using the AdamW optimizer with a learning rate of 1e-4 [arxiv_1706_03762:methodology:4].

Example 2:
Query: Who authored the baseline paper?
Context:
--- SOURCE: [paper_001:intro:1] ---
This paper describes a case for Deep Learning models.
-------------------------------------
Response: INSUFFICIENT_EVIDENCE
"""

class PromptLoader:
    def get_system_prompt(self) -> str:
        """
        Returns the structured RAG instructions prompt.
        """
        return SYSTEM_GROUNDING_INSTRUCTION.strip()

    def get_few_shot_prompt(self) -> str:
        """
        Returns few-shot examples illustrating grounding formatting.
        """
        return FEW_SHOT_EXAMPLES.strip()

    def compile_user_prompt(self, query: str, context_str: str) -> str:
        """
        Structures user prompt containing retrieval context blocks and the user query.
        """
        user_prompt = f"""
Use the following retrieved context segments to answer the query:

{context_str}

{self.get_few_shot_prompt()}

Query: {query}
Response:
"""
        return user_prompt.strip()
