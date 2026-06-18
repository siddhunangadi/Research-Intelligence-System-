"""
NLI-based Grounding Validator.
Uses a Natural Language Inference (NLI) model to mathematically evaluate whether 
a generated claim (hypothesis) is logically supported by its cited source context (premise).
"""

import logging

logger = logging.getLogger(__name__)

# Check dependency status
HAS_TRANSFORMERS = False
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
    HAS_TRANSFORMERS = True
except ImportError:
    logger.warning("transformers/torch not found. Running NLIValidator in Mock mode.")

class NLIValidator:
    def __init__(self, model_name: str = "cross-encoder/nli-deberta-v3-base", threshold: float = 0.70):
        self.model_name = model_name
        self.threshold = threshold
        self.tokenizer = None
        self.model = None
        self.device = "cpu"
        
        if HAS_TRANSFORMERS:
            try:
                self.device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
                logger.info(f"Loading NLI Validator model '{self.model_name}' on device: {self.device}")
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
                self.model.to(self.device)
                self.model.eval()
            except Exception as e:
                logger.error(f"Error loading NLI model: {e}. Falling back to mock validator.")
                self.model = None

    def check_entailment(self, premise: str, hypothesis: str) -> dict:
        """
        Validates logical alignment between cited context (premise) and generated claim (hypothesis).
        
        Returns:
            dict: {
                "label": "entailment" | "neutral" | "contradiction",
                "entailment_probability": float,
                "passed": bool
            }
        """
        # Truncate inputs to prevent out-of-memory or model size violations
        premise = premise.strip()
        hypothesis = hypothesis.strip()
        
        if not premise or not hypothesis:
            return {
                "label": "neutral",
                "entailment_probability": 0.0,
                "passed": False
            }

        if not self.model or not HAS_TRANSFORMERS:
            # Mock mode implementation
            # Checks if keywords or exact phrases overlap to simulate entailment
            overlap_words = set(hypothesis.lower().split()).intersection(set(premise.lower().split()))
            stop_words = {"is", "was", "the", "a", "an", "and", "or", "in", "of", "to", "for", "with", "by", "on", "at", "we", "i", "it", "this", "that"}
            significant_overlap = [w for w in overlap_words if w not in stop_words and len(w) > 3]
            
            overlap_score = len(overlap_words)
            hyp_len = len(set(hypothesis.lower().split()))
            ratio = overlap_score / max(hyp_len, 1)
            
            # Simple simulation: high overlap (~45%) maps to mock entailment
            # We dynamically calculate a probability based on the number of significant overlapping words
            hyp_content_words = set(hypothesis.lower().split()) - stop_words
            overlap_ratio = len(significant_overlap) / max(len(hyp_content_words), 1)
            
            if significant_overlap:
                simulated_prob = 0.70 + min(overlap_ratio * 0.40, 0.28)
            else:
                simulated_prob = min(ratio * 1.5, 0.65)
                
            passed = simulated_prob >= self.threshold
            label = "entailment" if passed else ("neutral" if simulated_prob > 0.2 else "contradiction")
            
            return {
                "label": label,
                "entailment_probability": float(simulated_prob),
                "passed": passed
            }

        try:
            inputs = self.tokenizer(
                premise, 
                hypothesis, 
                return_tensors="pt", 
                truncation=True, 
                max_length=512
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits[0]
                probabilities = torch.softmax(logits, dim=0)
                
                # NLI models trained on NLI datasets map classes:
                # 0 -> contradiction, 1 -> neutral, 2 -> entailment
                contradiction_prob = probabilities[0].item()
                neutral_prob = probabilities[1].item()
                entailment_prob = probabilities[2].item()
                
            labels = ["contradiction", "neutral", "entailment"]
            max_idx = torch.argmax(probabilities).item()
            predicted_label = labels[max_idx]
            passed = (predicted_label == "entailment" and entailment_prob >= self.threshold)
            
            return {
                "label": predicted_label,
                "entailment_probability": float(entailment_prob),
                "passed": passed,
                "details": {
                    "contradiction": float(contradiction_prob),
                    "neutral": float(neutral_prob),
                    "entailment": float(entailment_prob)
                }
            }
        except Exception as e:
            logger.error(f"Inference error during NLI validation: {e}")
            return {
                "label": "neutral",
                "entailment_probability": 0.0,
                "passed": False,
                "error": str(e)
            }
