from utils.helper import first_or_blank

def season_from_info(info):
    season = info.get("season")
    if isinstance(season, int):
        return season
    if isinstance(season, str) and season[:4].isdigit():
        return int(season[:4])

    date_text = first_or_blank(info.get("dates", []))
    return int(date_text[:4]) if date_text[:4].isdigit() else None