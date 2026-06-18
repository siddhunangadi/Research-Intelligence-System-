"""
Section Extraction Engine.
Parses markdown texts, maps structural headers to canonical scientific sections, 
and triggers LLM fallbacks when simple parser scans fail to resolve section boundaries.
"""

import logging
import re
from typing import Optional
from configs.config_manager import get_config

logger = logging.getLogger(__name__)

CANONICAL_SECTIONS = {
    "abstract": [r"\babstract\b", r"\bsummary\b"],
    "introduction": [r"\bintroduction\b", r"\bbackground\b", r"\bmotivation\b"],
    "methodology": [r"\bmethodology\b", r"\bproposed method\b", r"\bapproach\b", r"\bsystem model\b", r"\bmathematical formulation\b"],
    "experiments": [r"\bexperiments\b", r"\bevaluation setup\b", r"\bexperimental setup\b", r"\bmethodology implementation\b"],
    "results": [r"\bresults\b", r"\bempirical evaluation\b", r"\bfindings\b"],
    "discussion": [r"\bdiscussion\b", r"\banalysis\b", r"\brelated work\b"],
    "limitations": [r"\blimitations\b", r"\bthreats to validity\b", r"\bfuture work\b"],
    "conclusion": [r"\bconclusion\b", r"\bconcluding remarks\b"],
    "references": [r"\breferences\b", r"\bbibliography\b"]
}

class SectionExtractor:
    def __init__(self):
        self.config = get_config()

    def identify_section_by_header(self, text: str) -> Optional[str]:
        """
        Maps a heading line to our canonical taxonomy using regex.
        """
        line_clean = text.lower().strip()
        for section, patterns in CANONICAL_SECTIONS.items():
            for pattern in patterns:
                if re.search(pattern, line_clean):
                    return section
        return None

    def split_by_sections(self, full_markdown: str) -> dict[str, str]:
        """
        Splits a markdown document into section blocks.
        
        Returns:
            dict[str, str]: Dictionary mapping canonical section keys to text blocks.
        """
        lines = full_markdown.split("\n")
        sections_data = {}
        
        current_section = "introduction"  # Default initial section
        current_text_lines = []
        
        heading_pattern = re.compile(r"^#{1,4}\s+(.*)")
        
        for line in lines:
            line_strip = line.strip()
            if not line_strip:
                current_text_lines.append(line)
                continue
                
            heading_match = heading_pattern.match(line_strip)
            if heading_match:
                heading_content = heading_match.group(1)
                detected_section = self.identify_section_by_header(heading_content)
                
                if detected_section:
                    # Flush previous section content
                    if current_text_lines:
                        sections_data[current_section] = sections_data.get(current_section, "") + "\n" + "\n".join(current_text_lines)
                        current_text_lines = []
                    current_section = detected_section
                    logger.debug(f"Transitioned to section: {current_section} via header: '{line_strip}'")
                    continue
            
            current_text_lines.append(line)
            
        # Flush final block
        if current_text_lines:
            sections_data[current_section] = sections_data.get(current_section, "") + "\n" + "\n".join(current_text_lines)

        # Clean spaces
        for key in list(sections_data.keys()):
            sections_data[key] = sections_data[key].strip()
            if not sections_data[key]:
                del sections_data[key]
                
        # Fallback Gateway: If we recovered fewer than 4 canonical sections, 
        # try LLM-assisted section classification if OpenAI credentials are set
        if len(sections_data) < 4 and self.config.openai_api_key:
            logger.info("Fewer than 4 canonical sections detected. Triggering LLM Section Parser fallback.")
            sections_data = self._run_llm_section_parse(full_markdown, sections_data)
            
        return sections_data

    def _run_llm_section_parse(self, text: str, initial_data: dict) -> dict:
        """
        Invokes LLM API to segment text structure when regex pattern matches fail.
        """
        try:
            # Short preview of the text structure to prevent token budget overflows
            preview = text[:6000]
            
            # Simple API simulation or mock fallback if external client call is bypassed
            # In a production context, you would instantiate an OpenAI/Anthropic client here.
            # E.g.: client.chat.completions.create(...)
            
            # Fallback to returning initial_data directly if the API call is skipped
            logger.warning("LLM API parsing fallback not fully completed: missing client call. Preserving regex parsing.")
            return initial_data
        except Exception as e:
            logger.error(f"LLM Section parsing failed: {e}. Preserving regex results.")
            return initial_data
