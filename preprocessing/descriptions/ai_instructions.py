INSTRUCTIONS = """
You will generate TWO distinct descriptions of the animal in the provided image: one highly detailed (high_info) and one simplified (low_info). 

Desired Response format:
high_info="your high-info description" low_info="your low-info description"

Do not include any text other than these two descriptions.

### High-Info Description (max 45 words):
Provide a richly detailed description focused primarily on the animal itself to help accurately distinguish its species. Clearly describe animal size relative to everyday references (e.g., knee-height, waist-height), precise build (heavy-set, slender, muscular), specific fur color, pattern (striped, solid, spotted), fur texture (short, long, fluffy), distinct features (ear shape, tail length and shape, snout length), explicit body posture (sitting, standing, lying), demeanor (alert, playful, calm), and noticeable accessories (collar, harness). Only briefly mention environmental context if it directly indicates species (e.g., litterbox). DO NOT explicitly name the species (cat/dog).

### Low-Info Description (max 10 words):
Briefly summarize animal size (small, medium, large), primary fur color, distinctive fur pattern, basic posture, and setting (indoors/outdoors). Omit other details. DO NOT explicitly name the species (cat/dog).

### Example:
high_info="Slender, knee-height animal with smooth, ginger fur, distinctive dark stripes across limbs, body, and face. Upright, pointed ears, short snout, expressive green eyes, relaxed front paws. Sitting calmly, alert posture, no visible accessories. Indoors, gently held on person's lap." low_info="Small ginger animal with dark stripes, sitting indoors relaxed."
"""
