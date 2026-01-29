from pydantic import BaseModel
from typing import List

class InferenceRequest(BaseModel):
    texts : List[str]

class InferenceResponse(BaseModel):
    predictions : List[List[str]]