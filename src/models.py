from pydantic import BaseModel, Field
from typing import List, Optional, Literal

class Intent(BaseModel):
    category: Literal["question", "recommendation", "search", "greeting"] = Field(
        ..., 
        description="The category of the user's intent. 'question' for specific facts, 'recommendation' for suggestions, 'search' for looking up specific entities, 'greeting' for social interactions."
    )
    reasoning: str = Field(..., description="Brief explanation of why this category was chosen.")

class Entities(BaseModel):
    city: Optional[str] = Field(None, description="City name mentioned.")
    country: Optional[str] = Field(None, description="Country name mentioned.")
    hotel_name: Optional[str] = Field(None, description="Specific hotel name.")
    traveller_type: Optional[Literal["Solo", "Couple", "Family", "Business"]] = Field(None, description="Type of traveller.")
    min_rating: Optional[float] = Field(None, description="Minimum average review score.")
    min_stars: Optional[int] = Field(None, description="Minimum star rating.")
    min_cleanliness: Optional[float] = Field(None, description="Minimum cleanliness score.")
    min_comfort: Optional[float] = Field(None, description="Minimum comfort score.")
    min_facilities: Optional[float] = Field(None, description="Minimum facilities score.")
    age_min: Optional[int] = Field(None, description="Minimum age for demographic queries.")
    age_max: Optional[int] = Field(None, description="Maximum age for demographic queries.")
    target_country: Optional[str] = Field(None, description="Target country for visa queries.")
    current_country: Optional[str] = Field(None, description="Current country/Origin for visa queries.")
    attributes: Optional[List[str]] = Field(default_factory=list, description="Other attributes like 'pool', 'wifi'.")
    dates: Optional[List[str]] = Field(default_factory=list, description="Dates or duration mentioned.")
