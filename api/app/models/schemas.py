from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="Text to analyze for sentiment",
    )


class PredictResponse(BaseModel):
    label: str = Field(..., description="Predicted sentiment: POSITIVE or NEGATIVE")
    score: float = Field(..., ge=0.0, le=1.0, description="Confidence score of the prediction")