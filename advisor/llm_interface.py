import openai
from langsmith import traceable
import re

class OpenAIInterface:
    def __init__(self, api_key):
        self.client = openai.OpenAI(api_key=api_key)
    
    @traceable(name="generate_response", run_type="llm")
    def generate_response(self, query, context):
        """Generate response using OpenAI GPT-3.5 Turbo"""
        system_prompt = """
        You are a Fantasy IPL Cricket advisor. You provide data-driven insights for fantasy cricket players.
        Your advice should be specific, actionable, and based on the latest information provided in the context.
        Include relevant statistics, matchup analysis, and injury reports when applicable.
        If you're unsure or don't have enough information, acknowledge this and suggest what additional data might help.
        
        Always provide:
        1. Specific player recommendations with reasoning
        2. Key statistics or trends that support your advice
        3. Risk factors or uncertainties to consider
        4. Confidence level in your recommendation
        """
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Based on the following information, please answer this question about Fantasy IPL Cricket: {query}\n\nContext:\n{context}"}
        ]
        
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.3,
            max_tokens=800
        )
        
        response_text = response.choices[0].message.content
        
        confidence_indicators = self._extract_confidence_indicators(response_text)
        
        token_usage = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens
        }
        
        return {
            "response": response_text,
            "confidence_indicators": confidence_indicators,
            "token_usage": token_usage,
            "model": "gpt-3.5-turbo"
        }
    
    def _extract_confidence_indicators(self, response_text):
        """Extract confidence indicators from the response text"""
        confidence_phrases = {
            "high": ["highly recommend", "strongly suggest", "definitely", "certainly", "confident"],
            "medium": ["likely", "probably", "good option", "should consider", "recommended"],
            "low": ["might", "could", "uncertain", "limited data", "not sure", "maybe"]
        }
        
        response_lower = response_text.lower()
        confidence_scores = {"high": 0, "medium": 0, "low": 0}
        
        for level, phrases in confidence_phrases.items():
            confidence_scores[level] = sum(1 for phrase in phrases if phrase in response_lower)
        
        if confidence_scores["high"] > confidence_scores["medium"] and confidence_scores["high"] > confidence_scores["low"]:
            overall_confidence = "high"
        elif confidence_scores["medium"] > confidence_scores["low"]:
            overall_confidence = "medium"
        else:
            overall_confidence = "low"
        
        return {
            "phrase_counts": confidence_scores,
            "overall_confidence": overall_confidence,
            "response_length": len(response_text),
            "contains_statistics": bool(re.search(r'\d+\.?\d*\s*(%|runs|wickets|average|strike rate)', response_lower))
        }