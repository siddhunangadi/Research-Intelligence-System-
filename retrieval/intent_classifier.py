"""
Query Intent Classifier.
Parses query strings using keyword mappings to identify target sections.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Keywords mapping queries to canonical scientific sections
INTENT_KEYWORDS = {
    "methodology": [
        "optimizer", "learning rate", "algorithm", "architecture", "layers", 
        "weights", "adamw", "sgd", "hyperparameter", "framework", "loss function",
        "network", "implementation", "train", "optimizing", "model configuration"
    ],
    "experiments": [
        "batch size", "epochs", "dataset", "evaluation", "benchmark", "validation", 
        "training setup", "hardware", "gpu", "tpu", "metric", "parameters"
    ],
    "results": [
        "accuracy", "f1", "precision", "recall", "performance", "baseline", 
        "bleu", "score", "empirical", "table", "graph", "chart", "comparison"
    ],
    "discussion": [
        "why did", "comparison to", "related work", "previous study", "literature",
        "analysis", "improvement", "ablation", "contribution"
    ],
    "limitations": [
        "limit", "drawback", "failure", "threat to validity", "future work", 
        "bottleneck", "weakness", "error analysis"
    ],
    "abstract": [
        "summarize", "main goal", "objective", "in brief", "abstract"
    ]
}

class IntentClassifier:
    def classify_intent(self, query: str) -> list[str]:
        """
        Heuristically classifies the query text and returns matching target sections.
        Returns empty list if no clear intent matches are detected (defaults to global search).
        """
        query_clean = query.lower().strip()
        matched_sections = set()
        
        for section, keywords in INTENT_KEYWORDS.items():
            for word in keywords:
                if word in query_clean:
                    matched_sections.add(section)
                    break
                    
        # If experiments or results are queried, methodology is often helpful as context
        if "experiments" in matched_sections or "results" in matched_sections:
            matched_sections.add("methodology")
            
        result = list(matched_sections)
        logger.info(f"Query '{query}' classified to target sections: {result}")
        return result
