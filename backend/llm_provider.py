import os
from abc import ABC, abstractmethod
from typing import Optional, List
import logging

logger = logging.getLogger(__name__)

class LLMProvider(ABC):
    """Abstract base class for LLM providers"""
    
    @abstractmethod
    async def generate(self, prompt: str, context: str = "") -> str:
        """Generate a response given a prompt and optional context"""
        pass
    
    @abstractmethod
    def validate_credentials(self) -> bool:
        """Validate that credentials are set"""
        pass


class OpenAIProvider(LLMProvider):
    """OpenAI GPT provider"""
    
    def __init__(self):
        from config import settings
        self.api_key = settings.OPENAI_API_KEY
        self.model = settings.MODEL_NAME
        self.temperature = settings.TEMPERATURE
        self.max_tokens = settings.MAX_TOKENS
        
        if not self.validate_credentials():
            raise ValueError("OpenAI API key not found in environment")
    
    def validate_credentials(self) -> bool:
        return bool(self.api_key)
    
    async def generate(self, prompt: str, context: str = "") -> str:
        try:
            from openai import AsyncOpenAI
            client = AsyncOpenAI(api_key=self.api_key)
            
            messages = [
                {
                    "role": "system",
                    "content": "You are an expert Apache Spark assistant. Provide accurate, helpful answers based on the provided documentation context."
                },
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion: {prompt}"
                }
            ]
            
            response = await client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI API error: {str(e)}")
            raise


class AnthropicProvider(LLMProvider):
    """Anthropic Claude provider"""
    
    def __init__(self):
        from config import settings
        self.api_key = settings.ANTHROPIC_API_KEY
        self.temperature = settings.TEMPERATURE
        self.max_tokens = settings.MAX_TOKENS
        
        if not self.validate_credentials():
            raise ValueError("Anthropic API key not found in environment")
    
    def validate_credentials(self) -> bool:
        return bool(self.api_key)
    
    async def generate(self, prompt: str, context: str = "") -> str:
        try:
            from anthropic import AsyncAnthropic
            client = AsyncAnthropic(api_key=self.api_key)
            
            full_prompt = f"""You are an expert Apache Spark assistant. Provide accurate, helpful answers based on the provided documentation context.

Context:
{context}

Question: {prompt}"""
            
            message = await client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=self.max_tokens,
                messages=[
                    {"role": "user", "content": full_prompt}
                ]
            )
            
            return message.content[0].text
        except Exception as e:
            logger.error(f"Anthropic API error: {str(e)}")
            raise


def get_llm_provider() -> LLMProvider:
    """Factory function to get the configured LLM provider"""
    from config import settings
    
    if settings.LLM_PROVIDER == "openai":
        return OpenAIProvider()
    elif settings.LLM_PROVIDER == "anthropic":
        return AnthropicProvider()
    else:
        raise ValueError(f"Unknown LLM provider: {settings.LLM_PROVIDER}")