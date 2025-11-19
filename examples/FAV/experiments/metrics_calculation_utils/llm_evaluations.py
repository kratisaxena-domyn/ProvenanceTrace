import time
import json
from icecream import ic
import re
from openai import OpenAI
from ..utils.config_loader import get_llm_client_config


class LLM_Evaluations:
    def __init__(self):

        # Initialize LLM client for evaluation
        llm_config = get_llm_client_config()
        self.client = OpenAI(
            base_url=llm_config["base_url"],
            api_key=llm_config["api_key"]
        )
        # Store model name for use in API calls
        self.model_name = llm_config["model_name"]
        
    def _llm_judge_sentence_support(self, claim, sentence, original_prompt):
        """Use LLM to judge if a sentence supports a claim"""
        t0 = time.time()
        system_prompt = """You are an expert fact-checker evaluating sentence-level evidence.
    Given a claim and a sentence from a document, determine if the sentence provides evidence that supports the claim.

    Return only a JSON object with:
    {
        "supports_claim": true/false,
        "support_strength": 0.0-1.0,
        "reasoning": "brief explanation"
    }"""
        
        user_prompt = f"""
    Original Question: {original_prompt}

    Claim: {claim}

    Sentence: {sentence}

    Does this sentence provide evidence that supports the claim? Consider:
    - Direct factual support
    - Logical connection to the claim
    - Relevance and specificity
    """
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=200,
                temperature=0.1
            )
            
            content = response.choices[0].message.content
            # Try to extract JSON
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                try:
                    result = json.loads(json_match.group())
                    return result
                except Exception:
                    pass
            
            # Fallback parsing
            supports = "true" in content.lower() or "support" in content.lower()
            strength_match = re.search(r"support_strength[\":\s]*([0-9.]+)", content)
            strength = float(strength_match.group(1)) if strength_match else 0.5
            reasoning_match = re.search(r"reasoning[\":\s]*([^\"]+)", content)
            reasoning = reasoning_match.group(1) if reasoning_match else "Failed to parse JSON response"

            return {
                "supports_claim": supports,
                "support_strength": strength,
                "reasoning": reasoning
            }
            
        except Exception as e:
            ic(f"LLM judge error for sentence support: {e}")
            return {
                "supports_claim": False,
                "support_strength": 0.0,
                "reasoning": "Error in evaluation"
            }
            
    def _llm_judge_document_relevance(self, claim, doc_content, original_prompt):
        """Use LLM to judge if a retrieved document is relevant to the claim."""
        import re
        t0 = time.time()
        system_prompt = (
            "You are an expert fact-checker evaluating document relevance.\n"
            "Given a claim and a retrieved document, determine if the document is relevant evidence for the claim.\n"
            "Return only a JSON object with:\n"
            "{\n"
            "  'relevant': true/false,\n"
            "  'relevance_score': 0.0-1.0,\n"
            "  'reasoning': 'brief explanation'\n"
            "}"
        )
        user_prompt = f"""
Claim: {claim}
Document: {doc_content[:1000]}...
Original Prompt: {original_prompt}
Does this document provide relevant evidence for the claim?
"""
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=200,
                temperature=0.1
            )
            content = response.choices[0].message.content
            # Try to extract JSON
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group())
                except Exception:
                    pass
            relevant = "relevant" in content.lower() or "true" in content.lower()
            score_match = re.search(r"relevance_score[\":\s]*([0-9.]+)", content)
            relevance_score = float(score_match.group(1)) if score_match else 0.5
            reasoning_match = re.search(r"reasoning[\":\s]*([^\"]+)", content)
            reasoning = reasoning_match.group(1) if reasoning_match else "Failed to parse JSON response"
            llm_judge_document_relevance_time = time.time() - t0
            ic(llm_judge_document_relevance_time)
            return {
                "relevant": relevant,
                "relevance_score": relevance_score,
                "reasoning": reasoning
            }
        except Exception as e:
            ic(f"LLM judge error for document relevance: {e}")
            return {"relevant": False, "relevance_score": 0.0, "reasoning": "Error in evaluation"}

    def _llm_judge_hallucination(self, claim, available_evidence, original_prompt):
        """Use LLM to judge if a claim is hallucinated given available evidence"""
        t0 = time.time()
        system_prompt = """You are an expert fact-checker identifying potential hallucinations.
Given a claim and the available evidence, determine if the claim is:
1. Well-supported by evidence
2. Partially supported but potentially overstated
3. Unsupported/hallucinated

Return only a JSON object with:
{
    "hallucination_risk": "low"/"medium"/"high",
    "confidence": 0.0-1.0,
    "reasoning": "brief explanation",
    "evidence_gaps": ["list of missing evidence"]
}"""
        
        evidence_summary = "\n".join([
            f"- {doc.get('content', str(doc))[:200]}..." if isinstance(doc, dict)
            else f"- {' '.join([str(item) for item in doc])[:200]}..." if isinstance(doc, list)
            else f"- {str(doc)[:200]}..."
            for doc in available_evidence[:5]
        ])
        
        user_prompt = f"""
Original Question: {original_prompt}

Claim to evaluate: {claim}

Available Evidence:
{evidence_summary}

Assess if this claim is well-supported, overstated, or potentially hallucinated.
Consider:
- Is there direct evidence for this specific claim?
- Are there logical gaps between evidence and claim?
- Are there unsupported assumptions or extrapolations?
"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=300,
                temperature=0.1
            )
            
            content = response.choices[0].message.content
            # Try to extract JSON
            import re
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group())
                except Exception:
                    pass
            risk = "medium"
            if "low" in content.lower():
                risk = "low"
            elif "high" in content.lower():
                risk = "high"
            confidence_match = re.search(r"confidence[\":\s]*([0-9.]+)", content)
            confidence = float(confidence_match.group(1)) if confidence_match else 0.5
            reasoning_match = re.search(r"reasoning[\":\s]*([^\"]+)", content)
            reasoning = reasoning_match.group(1) if reasoning_match else "Failed to parse JSON response"
            return {
                "hallucination_risk": risk,
                "confidence": confidence,
                "reasoning": reasoning,
                "evidence_gaps": []
            }
            
        except Exception as e:
            ic(f"LLM judge error for hallucination: {e}")
            return {
                "hallucination_risk": "medium",
                "confidence": 0.0,
                "reasoning": "Error in evaluation",
                "evidence_gaps": []
            }