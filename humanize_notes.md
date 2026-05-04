# Humanize Report Script - Remove AI patterns
# This script will be used to track improvements

AI_PATTERNS_TO_REMOVE = [
    "Moreover", "Furthermore", "Additionally",  # AI vocabulary
    "It's not just", "It is not merely",  # Negative parallelism
    "stands as a testament", "serves as a",  # Inflated symbolism
    "Experts believe", "Industry reports",  # Vague attributions
    "pivotal role", "evolving landscape",  # AI vocabulary
    "crucial", "vital", "significant" (when overused),
]

# Target: Make it sound like a real student who:
# - Sometimes writes imperfectly
# - Has opinions and reactions
# - Doesn't always use perfect transitions
# - Is honest about limitations
# - Uses "I" or "we" when appropriate
