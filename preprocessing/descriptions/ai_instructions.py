INSTRUCTIONS = """
You are to generate TWO descriptions of the animal in the image: one highly detailed (high_info) and one simplified (low_info). 
Your responses MUST be provided exactly in this format:

Desired Response format:
high_info="your high-info description" low_info="your low-info description"

Do not include any text other than these two descriptions.

The high-info description should contain as much semantic detail as possible within 60 words, using concise, descriptive phrases, and allowing sentence fragments if necessary, but must remain natural prose.

The low-info description should concisely convey maximum semantic detail within 25 words, allowing sentence fragments if necessary, but must remain natural prose.

Instructions for each description:

High-Info Description:
Provide a highly detailed description in continuous, natural prose. Include approximate animal size relative to everyday references (e.g., knee-height, waist-height), build (heavy-set, slender, medium build, muscular), explicit fur color and texture (short, long, shaggy, smooth, fluffy), fur pattern (solid, spotted, striped, patchy), posture and stance (standing, sitting, lying down, in motion), demeanor (calm, alert, playful, relaxed), distinctive features (ear shape and position, tail length, shape, fluffiness and position, snout length and shape), accessories (color, type, and any readable text), gaze and facial expression (direction, expression type), and environmental context (ground surface and background details). DO NOT explicitly state the species (e.g., dog or cat).

Low-Info Description:
Provide a simplified, brief description in continuous, natural prose. Briefly state animal size (small, medium, large), primary fur color, main accessory (if visible, such as harness or collar), and general setting (indoors or outdoors). DO NOT explicitly state the species (e.g., dog or cat).
"""
