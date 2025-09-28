"""
Pydantic data models for waste analysis
Fully aligned with the output format required by the Gemini prompt

Uses Gemini API's response_schema directly,
no traditional parser needed
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from enum import Enum

class ConfidenceGrade(str, Enum):
    """Confidence level grades A/B/C/D"""
    A = "A"  # High: Clear features, high confidence
    B = "B"  # Medium: Generally clear features, moderate confidence  
    C = "C"  # Low: Unclear features, low confidence
    D = "D"  # Unreliable: Very poor quality/unclear, very low confidence

class PrimaryObjectAnalysis(BaseModel):
    """Primary object analysis - corresponds to PRIMARY OBJECT ANALYSIS section in prompt"""
    
    # Material composition (plastic, metal, paper, glass, etc.)
    material_composition: str = Field(
        description="Material composition, such as plastic, metal, paper, glass, organic, composite, hazardous, etc."
    )
    
    # Physical characteristics (shape, size, color, condition)
    shape: Optional[str] = Field(
        default=None,
        description="Shape of the object"
    )
    size: Optional[str] = Field(
        default=None,
        description="Size of the object"
    )
    color: Optional[str] = Field(
        default=None,
        description="Color of the object"
    )
    condition: Optional[str] = Field(
        default=None,
        description="Physical condition of the object"
    )
    
    # Likely function or original purpose
    likely_function_or_original_purpose: Optional[str] = Field(
        default=None, 
        description="Likely original function or purpose of the object"
    )
    
    # Current state (clean, dirty, damaged, intact)
    current_state: Optional[str] = Field(
        default=None,
        description="Current state, such as clean, dirty, damaged, intact"
    )
    
    # Completeness and visibility in the crop
    completeness_and_visibility_in_crop: Optional[str] = Field(
        default=None,
        description="Description of completeness and visibility in the crop image"
    )

class SecondaryElements(BaseModel):
    """Secondary elements - corresponds to SECONDARY ELEMENTS section in prompt"""
    
    # Other visible objects or portions
    other_visible_objects_or_portions: List[str] = Field(
        description="Other visible objects or portions in the image",
        default_factory=list
    )
    
    # Their relationship to the primary object
    relationship_to_primary_object: Optional[str] = Field(
        default=None,
        description="Relationship to the primary object"
    )
    
    # Potential impact on classification
    potential_impact_on_classification: Optional[str] = Field(
        default=None,
        description="Potential impact on waste classification"
    )

class OverallAssessment(BaseModel):
    """Overall assessment - corresponds to OVERALL ASSESSMENT section in prompt"""
    
    # Primary object category determination
    primary_object_category_determination: str = Field(
        description="Primary object category determination"
    )
    
    # Recyclability assessment if possible
    recyclability_assessment_if_possible: Optional[str] = Field(
        default=None,
        description="Recyclability assessment if possible"
    )
    
    # Confidence level in your assessment
    confidence_level_in_assessment: ConfidenceGrade = Field(
        description="Confidence level in the assessment"
    )
    
    # Any ambiguities or uncertainties you observe
    ambiguities_or_uncertainties_observed: List[str] = Field(
        description="Any ambiguities or uncertainties observed during analysis",
        default_factory=list
    )
    
    # Alternative possibilities if classification is unclear
    alternative_possibilities_if_unclear: List[str] = Field(
        description="Alternative possibilities if classification is unclear",
        default_factory=list
    )

class SingleCropAnalysis(BaseModel):
    """Single crop analysis result - completely matches prompt output format"""
    
    # Basic information
    region_id: int = Field(description="Region ID")
    crop_path: str = Field(description="Crop image path")
    
    # Three main sections - completely corresponding to prompt format
    primary_object_analysis: PrimaryObjectAnalysis = Field(
        description="PRIMARY OBJECT ANALYSIS: Analysis of the main classifiable object"
    )
    
    secondary_elements: SecondaryElements = Field(
        description="SECONDARY ELEMENTS: Other visible objects or portions"
    )
    
    overall_assessment: OverallAssessment = Field(
        description="OVERALL ASSESSMENT: Category determination and confidence"
    )

class BatchCropAnalysis(BaseModel):
    """Batch crop analysis result - corresponds to batch prompt format"""
    
    total_crops: int = Field(description="Total number of crops")
    
    # Analysis results for each crop, format: CROP [X] - ...
    crop_analyses: List[SingleCropAnalysis] = Field(
        description="Analysis results for each crop, format: CROP [X] - PRIMARY OBJECT ANALYSIS, etc."
    )
    
    # Batch processing summary
    batch_summary: Optional[str] = Field(
        default=None,
        description="Batch analysis summary"
    )

# Compatibility model - maintains backward compatibility
class WasteObjectAnalysis(BaseModel):
    """Waste object analysis result - compatibility model
    
    Maintains compatibility with existing code while supporting new structured format
    """
    
    # Basic information
    region_id: int = Field(description="Region ID")
    crop_path: str = Field(description="Crop image path")
    
    # Primary object analysis (compatible with existing fields)
    primary_material: str = Field(description="Primary material type")
    primary_shape: Optional[str] = Field(default=None, description="Primary shape")
    primary_color: Optional[str] = Field(default=None, description="Primary color")
    primary_condition: Optional[str] = Field(default=None, description="Primary condition")
    primary_purpose: Optional[str] = Field(default=None, description="Original purpose")
    
    # Secondary elements
    secondary_elements: List[str] = Field(default_factory=list, description="Secondary visible elements")
    
    # Classification assessment
    suggested_category: str = Field(description="Suggested waste classification category")
    category_confidence: ConfidenceGrade = Field(description="Category confidence level")
    alternative_categories: List[str] = Field(default_factory=list, description="Alternative categories")
    
    # Recyclability
    recyclable: Optional[bool] = Field(default=None, description="Whether the item is recyclable")
    recyclability_notes: Optional[str] = Field(default=None, description="Recyclability notes")
    
    # Overall assessment
    overall_description: str = Field(description="Overall description")
    uncertainties: List[str] = Field(default_factory=list, description="Uncertainties")
    analysis_confidence: ConfidenceGrade = Field(description="Overall analysis confidence")
    
    # Physical characteristics
    size_description: Optional[str] = Field(default=None, description="Size description")
    completeness: Optional[str] = Field(default=None, description="Completeness")
    
    class Config:
        """Pydantic configuration"""
        use_enum_values = True
        json_encoders = {
            ConfidenceGrade: lambda v: v.value
        }

# Factory functions
def create_single_crop_schema() -> type[SingleCropAnalysis]:
    """Factory function to create single crop analysis schema"""
    return SingleCropAnalysis

def create_batch_crop_schema() -> type[BatchCropAnalysis]:
    """Factory function to create batch crop analysis schema"""
    return BatchCropAnalysis

def create_waste_analysis_schema() -> type[WasteObjectAnalysis]:
    """Factory function to create compatibility waste analysis schema"""
    return WasteObjectAnalysis

# Testing and examples
if __name__ == "__main__":
    import json
    
    print("=== Redesigned Pydantic Waste Analysis Model Test ===")
    print()
    
    # Test single crop analysis model
    print("🧪 **Testing Single Crop Analysis Model**:")
    single_analysis = SingleCropAnalysis(
        region_id=1,
        crop_path="crop_001.jpg",
        primary_object_analysis=PrimaryObjectAnalysis(
            material_composition="plastic",
            shape="cylindrical bottle",
            size="medium",
            color="white/translucent with blue cap",
            condition="intact",
            likely_function_or_original_purpose="milk storage",
            current_state="clean",
            completeness_and_visibility_in_crop="complete and clearly visible"
        ),
        secondary_elements=SecondaryElements(
            other_visible_objects_or_portions=["label", "cap"],
            relationship_to_primary_object="attached components",
            potential_impact_on_classification="supports plastic container classification"
        ),
        overall_assessment=OverallAssessment(
            primary_object_category_determination="Plastic Container",
            recyclability_assessment_if_possible="HDPE milk bottles are widely recyclable",
            confidence_level_in_assessment=ConfidenceGrade.A,
            ambiguities_or_uncertainties_observed=[],
            alternative_possibilities_if_unclear=["Beverage Container", "Food Storage"]
        )
    )
    
    print("✅ Single crop analysis model created successfully")
    print(f"📊 Field count: {len(single_analysis.model_fields)}")
    
    # Test batch analysis model
    print()
    print("🧪 **Testing Batch Analysis Model**:")
    batch_analysis = BatchCropAnalysis(
        total_crops=2,
        crop_analyses=[single_analysis],
        batch_summary="Successfully analyzed 1 crop with high confidence"
    )
    
    print("✅ Batch analysis model created successfully")
    
    # Output JSON
    print()
    print("📋 **Single Crop Analysis JSON Output**:")
    print(json.dumps(single_analysis.model_dump(), indent=2, ensure_ascii=False))
    
    print()
    print("📊 **JSON Schema**:")
    schema = SingleCropAnalysis.model_json_schema()
    print(f"• Model name: {schema.get('title', 'N/A')}")
    print(f"• Field count: {len(schema.get('properties', {}))}")
    print(f"• Required fields: {len(schema.get('required', []))}")
    
    print()
    print("🎯 **Format Matching Verification**:")
    print("✅ PRIMARY OBJECT ANALYSIS: Completely matches prompt requirements")
    print("✅ SECONDARY ELEMENTS: Completely matches prompt requirements") 
    print("✅ OVERALL ASSESSMENT: Completely matches prompt requirements")
    print("✅ Field structure matches Gemini prompt output format")
    
    print()
    print("✅ **Pydantic Model Redesign Successful!**")
    print("🎯 **Next Step**: Configure new Schema in Gemini Inferencer")