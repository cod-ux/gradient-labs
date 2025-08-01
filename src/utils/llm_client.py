from openai import OpenAI
from dotenv import load_dotenv
from typing import Any, Type, Optional
from pydantic import BaseModel
import os
import time
from pydantic_core import ValidationError

load_dotenv()


class LLMClient:
    """Wrapper for OpenAI API calls with structured output support."""
    
    def __init__(self, api_key: Optional[str] = None):
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
    
    def parse_response(
        self,
        prompt: str,
        response_model: Type[BaseModel],
        model: str = "gpt-4.1",
        temperature: float = 0.0,
        max_retries: int = 3
    ) -> BaseModel:
        """Generate structured response using OpenAI's structured output with retry logic."""
        for attempt in range(max_retries):
            try:
                # o3 model doesn't support temperature parameter
                if model.startswith("o3"):
                    response = self.client.responses.parse(
                        model=model,
                        input=prompt,
                        text_format=response_model
                    )
                else:
                    response = self.client.responses.parse(
                        model=model,
                        input=prompt,
                        text_format=response_model,
                        temperature=temperature
                    )
                return response.output_parsed
                
            except ValidationError as e:
                print(f"JSON validation error on attempt {attempt + 1}: {e}")
                if attempt < max_retries - 1:
                    print(f"Retrying in 2 seconds...")
                    time.sleep(2)
                else:
                    print(f"Failed after {max_retries} attempts, re-raising error")
                    raise
            except Exception as e:
                print(f"API error on attempt {attempt + 1}: {e}")
                if attempt < max_retries - 1:
                    print(f"Retrying in 2 seconds...")
                    time.sleep(2)
                else:
                    print(f"Failed after {max_retries} attempts, re-raising error")
                    raise
    
    def create_embeddings(
        self,
        texts: list[str],
        model: str = "text-embedding-3-large"
    ) -> list[list[float]]:
        """Create embeddings for a list of texts."""
        response = self.client.embeddings.create(
            input=texts,
            model=model
        )
        return [embedding.embedding for embedding in response.data]
    
    def create_single_embedding(
        self,
        text: str,
        model: str = "text-embedding-3-large"
    ) -> list[float]:
        """Create embedding for a single text."""
        response = self.client.embeddings.create(
            input=[text],
            model=model
        )
        return response.data[0].embedding