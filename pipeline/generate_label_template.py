#!/usr/bin/env python3
"""
Waste classification label template generator
- Generate main/sub label config for CLIP classification
- Use a simplified taxonomy to avoid confusion between Other and Unknown
- Produce concise English prompts per category while preserving main-sub hierarchy
- Optimization: every prompt mentions its main class; handle "other" subclass; improved aggregation
"""

import json

# ====== Main and sub category definitions (simplified taxonomy) ======
# Main: physical/material categories of waste, keep the count between 8 and 12
# Sub: specific item types under a main class, 3-8 per main class
main_categories = {
    "Plastics": ["bottle", "bag", "box", "container", "film", "cup", "other"],
    "Metals": ["can", "foil", "lid", "aerosol can", "other"],
    "Papers": ["newspaper", "carton", "tissue", "flyer", "envelope", "other"],
    "Glass": ["bottle", "jar", "cup", "fragment", "other"],
    "Textiles": ["cloth", "shoe", "towel", "glove", "other"],
    "E-waste": ["battery", "phone", "cable", "charger", "other"],
    "Organic": ["food", "fruit peel", "vegetable", "bone", "other"],
    "Background": ["background"],   # 独立主类：纯背景、无物体区域
    "Unknown": ["unknown"]          # 独立主类：无法识别或不属于已知类别的物体
}

def generate_prompts(main, sub):
    """
    Generate concise English prompts for each main/sub category
    
    Design principles:
    1. Objectivity: focus on object type, avoid subjective state
    2. Brevity: up to 5 prompts per class to reduce cost
    3. Hierarchy: preserve main-sub relationship
    4. Practicality: suited for crop images, follow proven patterns
    5. Main class presence: every prompt mentions the main class
    6. "Other" handling: provide catch-all prompts for "other" subclass
    
    Args:
        main (str): main class name
        sub (str): sub class name
    
    Returns:
        list: prompts for the given main/sub (max 5)
    """
    # Background: pure background/no-object regions
    if main.lower() == "background":
        return [
            "background",
            "empty space", 
            "no object",
            "background area",
            "empty"
        ]
    
    # Unknown: unrecognized or not in known categories
    if main.lower() == "unknown":
        return [
            "unknown object",
            "unclear item",
            "unrecognized",
            "unknown",
            "unclear"
        ]
    
    # Other subclass: catch-all under the main class
    if sub.lower() == "other":
        return [
            f"other {main} item",
            f"a {main} object",
            f"this is a {main} item",
            f"a piece of {main} material",
            f"{main} waste"
        ]
    
    # Regular categories: ensure every prompt mentions main class
    prompts = [
        # Basic expression (core, highest weight) - mentions main class
        f"{main} {sub}",
        f"a {sub} which is a type of {main}",
        
        # Hierarchical expression (preserve relation) - mentions main class
        f"this is a {main} item, specifically a {sub}",
        f"{sub} made of {main} material",
        
        # Generic description (effective pattern) - mentions main class
        f"a {main} object in {sub} form"
    ]
    
    # Deduplicate and limit count
    unique_prompts = list(set(prompts))
    return unique_prompts[:5]  # ensure no more than 5

def generate_label_template():
    """
    Generate the complete label template
    
    Returns:
        list: config list with all main/sub categories and prompts
    """
    label_list = []
    
    # Iterate all main/sub combinations
    for main, subs in main_categories.items():
        for sub in subs:
            label_config = {
                "main": main,
                "sub": sub,
                "prompts": generate_prompts(main, sub)
            }
            label_list.append(label_config)
    
    return label_list

def save_template(label_list, output_path="waste_labels_template.json"):
    """
    Save label template to JSON file
    
    Args:
        label_list (list): label configurations
        output_path (str): output file path
    """
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(label_list, f, indent=2, ensure_ascii=False)
    
    # Print statistics
    total_categories = len(label_list)
    total_prompts = sum(len(item["prompts"]) for item in label_list)
    main_classes = set(item["main"] for item in label_list)
    sub_classes = set(item["sub"] for item in label_list)
    
    print(f"✅ Label template generated: {output_path}")
    print(f"📊 Stats:")
    print(f"   - Main classes: {len(main_classes)}")
    print(f"   - Subclasses: {len(sub_classes)}")
    print(f"   - Total categories: {total_categories}")
    print(f"   - Total prompts: {total_prompts}")
    print(f"   - Avg prompts per category: {total_prompts/total_categories:.1f}")
    print(f"   - Main class list: {sorted(main_classes)}")
    
    # Show prompt examples
    print(f"\n📝 Prompt examples:")
    for i, item in enumerate(label_list[:6]):  # show first 6
        print(f"   {item['main']}:{item['sub']} -> {item['prompts']}")
    
    # Show examples for 'other' subclass
    other_examples = [item for item in label_list if item['sub'] == 'other']
    if other_examples:
        print(f"\n🔄 'Other' subclass examples:")
        for item in other_examples[:3]:
            print(f"   {item['main']}:other -> {item['prompts']}")

def main():
    """Main: generate and save label template"""
    print("🚀 Generating waste classification label template (optimized)...")
    print("=" * 60)
    
    # Generate label template
    label_list = generate_label_template()
    
    # Save to file
    save_template(label_list)
    
    print("\n💡 Notes:")
    print("   - Every prompt includes its main class to aid classification accuracy")
    print("   - Add an 'other' subclass for each main class to handle edge cases")
    print("   - Improved aggregation design to widen coverage")
    print("   - Keep it concise: up to 5 prompts per class")
    print("   - Clear hierarchy for CLIP understanding and classification")

if __name__ == "__main__":
    main() 