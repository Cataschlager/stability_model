"""Historical event reconstruction validation.

Tests whether the model correctly identifies instability spikes and
transmission pathways for known historical episodes.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

EPISODES = [
    {
        "name": "Arab Spring",
        "years": [2010, 2011, 2012],
        "primary": ["TUN", "EGY", "LBY", "SYR", "YEM", "BHR"],
        "secondary": ["JOR", "MAR", "DZA", "LBN", "IRQ"],
    },
    {
        "name": "Euro Sovereign Debt Crisis",
        "years": [2010, 2011, 2012],
        "primary": ["GRC", "IRL", "PRT", "ESP", "ITA"],
        "secondary": ["FRA", "DEU", "CYP", "BEL"],
    },
    {
        "name": "2014 Oil Collapse + Russia/Ukraine",
        "years": [2014, 2015],
        "primary": ["RUS", "UKR", "VEN", "NGA", "SAU"],
        "secondary": ["KAZ", "BLR", "AZE", "IRQ"],
    },
    {
        "name": "COVID-19 Onset",
        "years": [2020],
        "primary": [],  # Global
        "secondary": [],
    },
    {
        "name": "2022 Russia-Ukraine War",
        "years": [2022],
        "primary": ["RUS", "UKR"],
        "secondary": ["DEU", "POL", "EGY", "LBN", "TUN"],
    },
    {
        "name": "2023 Sahel Coup Belt",
        "years": [2023],
        "primary": ["NER", "MLI", "BFA", "GAB"],
        "secondary": ["TCD", "NGA", "GIN", "SEN"],
    },
]


def validate_episode(episode: dict, scores_panel: pd.DataFrame,
                     countries: list[str]) -> dict:
    """Validate a single historical episode.

    Checks:
    1. Primary countries' scores rose relative to prior year
    2. Secondary countries saw spillover effects

    Returns: dict with metrics
    """
    name = episode["name"]
    years = episode["years"]
    primary = [c for c in episode["primary"] if c in countries]
    secondary = [c for c in episode["secondary"] if c in countries]

    result = {"episode": name, "primary_checked": len(primary), "secondary_checked": len(secondary)}

    if not years or scores_panel.empty:
        result["status"] = "SKIP"
        return result

    # Check primary countries: did scores increase during episode years?
    primary_increases = 0
    for country in primary:
        country_scores = scores_panel[scores_panel["iso3"] == country].sort_values("year")
        for yr in years:
            curr = country_scores[country_scores["year"] == yr]
            prev = country_scores[country_scores["year"] == yr - 1]
            if not curr.empty and not prev.empty:
                if curr["composite_score"].iloc[0] > prev["composite_score"].iloc[0]:
                    primary_increases += 1

    result["primary_increases"] = primary_increases
    result["primary_increase_rate"] = primary_increases / max(len(primary) * len(years), 1)

    # Check secondary countries: did any see spillover?
    secondary_increases = 0
    for country in secondary:
        country_scores = scores_panel[scores_panel["iso3"] == country].sort_values("year")
        for yr in years:
            curr = country_scores[country_scores["year"] == yr]
            prev = country_scores[country_scores["year"] == yr - 1]
            if not curr.empty and not prev.empty:
                if curr["composite_score"].iloc[0] > prev["composite_score"].iloc[0]:
                    secondary_increases += 1

    result["secondary_increases"] = secondary_increases
    result["secondary_increase_rate"] = secondary_increases / max(len(secondary) * len(years), 1)

    # Pass if majority of primary countries show increases
    result["status"] = "PASS" if result["primary_increase_rate"] >= 0.5 else "FAIL"

    logger.info("[%s] %s — Primary %.0f%% increased, Secondary %.0f%% increased",
                 name, result["status"],
                 result["primary_increase_rate"] * 100,
                 result["secondary_increase_rate"] * 100)
    return result


def run_all_episodes(scores_panel: pd.DataFrame, countries: list[str]) -> pd.DataFrame:
    """Run validation for all historical episodes."""
    results = []
    for episode in EPISODES:
        res = validate_episode(episode, scores_panel, countries)
        results.append(res)
    return pd.DataFrame(results)
