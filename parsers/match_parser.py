def first_or_blank(values):
    return values[0] if values else ""

def phase_for_over(over):
    if 0 <= over <= 5:
        return "Powerplay"
    if 6 <= over <= 15:
        return "Middle"
    if 16 <= over <= 19:
        return "Death"
    return "Other"