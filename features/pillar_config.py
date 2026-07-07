"""Pillar configuration — maps every indicator to its canonical pillar.

Defines the 7-pillar taxonomy and sign-orientation rules per METHODOLOGY.md §3.5.
"""

from dataclasses import dataclass
from enum import Enum


class Pillar(Enum):
    POLITICAL_LEGITIMACY = "political_legitimacy"
    STATE_CAPACITY = "state_capacity"
    ECONOMIC_PERFORMANCE = "economic_performance"
    FISCAL_VULNERABILITY = "fiscal_vulnerability"
    SOCIAL_COHESION = "social_cohesion"
    SECURITY_VIOLENCE = "security_violence"
    ENVIRONMENTAL_STRESS = "environmental_stress"


@dataclass(frozen=True)
class IndicatorSpec:
    """Specification for a single indicator in the model."""
    name: str                   # Human-readable name
    source: str                 # Data source key (matches connector source_name)
    variable_code: str          # Original variable code in the source
    pillar: Pillar              # Canonical pillar assignment
    invert: bool                # True if original direction means higher=better (needs flipping)
    relative_weight: float = 1.0  # Within-pillar weight (default 1.0; protests get 0.5)


# ── Complete indicator registry ──────────────────────────────────────────────

ALL_INDICATORS: list[IndicatorSpec] = [
    # ── Pillar 1: Political Legitimacy ──
    IndicatorSpec("Polyarchy Index", "vdem", "vdem_polyarchy", Pillar.POLITICAL_LEGITIMACY, invert=True),
    IndicatorSpec("Liberal Democracy", "vdem", "vdem_liberal_democracy", Pillar.POLITICAL_LEGITIMACY, invert=True),
    IndicatorSpec("Deliberative Democracy", "vdem", "vdem_deliberative_democracy", Pillar.POLITICAL_LEGITIMACY, invert=True),
    IndicatorSpec("Egalitarian Democracy", "vdem", "vdem_egalitarian_democracy", Pillar.POLITICAL_LEGITIMACY, invert=True),
    IndicatorSpec("Participatory Democracy", "vdem", "vdem_participatory_democracy", Pillar.POLITICAL_LEGITIMACY, invert=True),
    IndicatorSpec("Polity2 Score", "polity5", "polity2_score", Pillar.POLITICAL_LEGITIMACY, invert=True),
    IndicatorSpec("Political Rights", "freedom_house", "fh_political_rights", Pillar.POLITICAL_LEGITIMACY, invert=False),  # Higher = worse
    IndicatorSpec("Civil Liberties", "freedom_house", "fh_civil_liberties", Pillar.POLITICAL_LEGITIMACY, invert=False),
    IndicatorSpec("State Legitimacy (FSI)", "fsi", "fsi_state_legitimacy", Pillar.POLITICAL_LEGITIMACY, invert=False),
    IndicatorSpec("Factionalized Elites (FSI)", "fsi", "fsi_factionalized_elites", Pillar.POLITICAL_LEGITIMACY, invert=False),

    # ── Pillar 2: State Capacity & Rule of Law ──
    IndicatorSpec("Govt Effectiveness (WGI)", "worldbank_wgi", "wgi_govt_effectiveness", Pillar.STATE_CAPACITY, invert=True),
    IndicatorSpec("Rule of Law (WGI)", "worldbank_wgi", "wgi_rule_of_law", Pillar.STATE_CAPACITY, invert=True),
    IndicatorSpec("Regulatory Quality (WGI)", "worldbank_wgi", "wgi_regulatory_quality", Pillar.STATE_CAPACITY, invert=True),
    IndicatorSpec("Control of Corruption (WGI)", "worldbank_wgi", "wgi_control_corruption", Pillar.STATE_CAPACITY, invert=True),
    IndicatorSpec("CPI Score", "transparency_intl", "cpi_score", Pillar.STATE_CAPACITY, invert=True),
    IndicatorSpec("Executive Constraints", "polity5", "executive_constraints", Pillar.STATE_CAPACITY, invert=True),
    IndicatorSpec("Public Services (FSI)", "fsi", "fsi_public_services", Pillar.STATE_CAPACITY, invert=False),
    IndicatorSpec("Human Rights (FSI)", "fsi", "fsi_human_rights", Pillar.STATE_CAPACITY, invert=False),

    # ── Pillar 3: Economic Performance ──
    IndicatorSpec("GDP Growth", "worldbank_wdi", "gdp_growth_pct", Pillar.ECONOMIC_PERFORMANCE, invert=True),
    IndicatorSpec("Inflation (CPI)", "worldbank_wdi", "inflation_cpi_pct", Pillar.ECONOMIC_PERFORMANCE, invert=False),
    IndicatorSpec("Unemployment Rate", "worldbank_wdi", "unemployment_pct", Pillar.ECONOMIC_PERFORMANCE, invert=False),
    IndicatorSpec("Youth Unemployment", "worldbank_wdi", "youth_unemployment_pct", Pillar.ECONOMIC_PERFORMANCE, invert=False),
    IndicatorSpec("Gini Coefficient", "worldbank_wdi", "gini_coefficient", Pillar.ECONOMIC_PERFORMANCE, invert=False),
    IndicatorSpec("Economic Decline (FSI)", "fsi", "fsi_economic_decline", Pillar.ECONOMIC_PERFORMANCE, invert=False),
    IndicatorSpec("Uneven Development (FSI)", "fsi", "fsi_uneven_development", Pillar.ECONOMIC_PERFORMANCE, invert=False),

    # ── Pillar 4: Fiscal & External Vulnerability ──
    IndicatorSpec("Fiscal Balance % GDP", "imf_weo", "fiscal_balance_pct_gdp", Pillar.FISCAL_VULNERABILITY, invert=True),  # Deficit is negative → invert
    IndicatorSpec("Gross Debt % GDP", "imf_weo", "gross_debt_pct_gdp", Pillar.FISCAL_VULNERABILITY, invert=False),
    IndicatorSpec("Current Account % GDP", "worldbank_wdi", "current_account_pct_gdp", Pillar.FISCAL_VULNERABILITY, invert=False),  # Take absolute deficit in build_indicators
    IndicatorSpec("Reserves Months Imports", "worldbank_wdi", "reserves_months_imports", Pillar.FISCAL_VULNERABILITY, invert=True),

    # ── Pillar 5: Social Cohesion & Demography ──
    IndicatorSpec("Refugee Population", "worldbank_wdi", "refugee_population", Pillar.SOCIAL_COHESION, invert=False),
    IndicatorSpec("IDPs", "worldbank_wdi", "idps", Pillar.SOCIAL_COHESION, invert=False),
    IndicatorSpec("Demographic Pressures (FSI)", "fsi", "fsi_demographic_pressures", Pillar.SOCIAL_COHESION, invert=False),
    IndicatorSpec("Human Flight (FSI)", "fsi", "fsi_human_flight", Pillar.SOCIAL_COHESION, invert=False),
    IndicatorSpec("Group Grievance (FSI)", "fsi", "fsi_group_grievance", Pillar.SOCIAL_COHESION, invert=False),
    IndicatorSpec("Refugees & IDPs (FSI)", "fsi", "fsi_refugees_idps", Pillar.SOCIAL_COHESION, invert=False),

    # ── Pillar 6: Security & Violence ──
    IndicatorSpec("UCDP Conflict Events", "ucdp", "ucdp_conflict_events", Pillar.SECURITY_VIOLENCE, invert=False),
    IndicatorSpec("ACLED Battles", "acled", "acled_battles", Pillar.SECURITY_VIOLENCE, invert=False),
    IndicatorSpec("ACLED Violence vs Civilians", "acled", "acled_violence_civilians", Pillar.SECURITY_VIOLENCE, invert=False),
    IndicatorSpec("ACLED Explosions", "acled", "acled_explosions", Pillar.SECURITY_VIOLENCE, invert=False),
    IndicatorSpec("ACLED Riots", "acled", "acled_riots", Pillar.SECURITY_VIOLENCE, invert=False),
    IndicatorSpec("ACLED Protests", "acled", "acled_protests", Pillar.SECURITY_VIOLENCE, invert=False, relative_weight=0.5),
    IndicatorSpec("Military Expend. % GDP", "sipri", "military_expenditure_pct_gdp", Pillar.SECURITY_VIOLENCE, invert=False),
    IndicatorSpec("Security Apparatus (FSI)", "fsi", "fsi_security_apparatus", Pillar.SECURITY_VIOLENCE, invert=False),
    IndicatorSpec("External Intervention (FSI)", "fsi", "fsi_external_intervention", Pillar.SECURITY_VIOLENCE, invert=False),

    # ── Pillar 7: Environmental & Resource Stress ──
    IndicatorSpec("Prevalence of Undernourishment", "fao", "prevalence_of_undernourishment", Pillar.ENVIRONMENTAL_STRESS, invert=False),
    IndicatorSpec("Dietary Energy Adequacy", "fao", "dietary_energy_adequacy", Pillar.ENVIRONMENTAL_STRESS, invert=True),
    IndicatorSpec("ND-GAIN Vulnerability", "ndgain", "ndgain_vulnerability", Pillar.ENVIRONMENTAL_STRESS, invert=False),
    IndicatorSpec("ND-GAIN Readiness", "ndgain", "ndgain_readiness", Pillar.ENVIRONMENTAL_STRESS, invert=True),
]


def get_indicators_by_pillar(pillar: Pillar) -> list[IndicatorSpec]:
    """Return all indicators assigned to a given pillar."""
    return [i for i in ALL_INDICATORS if i.pillar == pillar]


def get_indicator_by_code(variable_code: str) -> IndicatorSpec | None:
    """Look up an indicator by its variable code."""
    for i in ALL_INDICATORS:
        if i.variable_code == variable_code:
            return i
    return None


def get_variable_codes() -> list[str]:
    """Return all variable codes in the registry."""
    return [i.variable_code for i in ALL_INDICATORS]


def get_inversion_map() -> dict[str, bool]:
    """Return a dict mapping variable_code → whether to invert."""
    return {i.variable_code: i.invert for i in ALL_INDICATORS}


def get_weight_map() -> dict[str, float]:
    """Return a dict mapping variable_code → relative within-pillar weight."""
    return {i.variable_code: i.relative_weight for i in ALL_INDICATORS}
