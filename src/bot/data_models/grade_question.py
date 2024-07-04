from langchain_core.pydantic_v1 import BaseModel, Field

class GradeQuestionType(BaseModel):
    """Binary score for question type, pleasantries or factual."""
    
    binary_score:str = Field(
        description="Questions are pleasantries or factual, 'pleasantries' or 'factual'"
    )
