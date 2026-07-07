"""IGO/regional organization membership data and political affinity matrix.

Computes political coupling from shared membership in international
governmental organizations. No external data downloads required.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)

# Major IGOs and regional organizations with current membership (ISO-3)
IGO_MEMBERSHIPS: dict[str, list[str]] = {
    "EU": ["AUT", "BEL", "BGR", "HRV", "CYP", "CZE", "DNK", "EST", "FIN",
           "FRA", "DEU", "GRC", "HUN", "IRL", "ITA", "LVA", "LTU", "LUX",
           "MLT", "NLD", "POL", "PRT", "ROU", "SVK", "SVN", "ESP", "SWE"],
    "NATO": ["USA", "GBR", "FRA", "DEU", "ITA", "CAN", "TUR", "NLD", "BEL",
             "LUX", "NOR", "DNK", "ISL", "PRT", "GRC", "ESP", "POL", "CZE",
             "HUN", "BGR", "ROU", "SVK", "SVN", "HRV", "ALB", "MNE", "MKD",
             "LTU", "LVA", "EST", "FIN", "SWE"],
    "ASEAN": ["BRN", "KHM", "IDN", "LAO", "MYS", "MMR", "PHL", "SGP", "THA", "VNM"],
    "AU": ["DZA", "AGO", "BEN", "BWA", "BFA", "BDI", "CMR", "CPV", "CAF",
           "TCD", "COM", "COG", "COD", "CIV", "DJI", "EGY", "GNQ", "ERI",
           "SWZ", "ETH", "GAB", "GMB", "GHA", "GIN", "GNB", "KEN", "LSO",
           "LBR", "LBY", "MDG", "MWI", "MLI", "MRT", "MUS", "MAR", "MOZ",
           "NAM", "NER", "NGA", "RWA", "STP", "SEN", "SYC", "SLE", "SOM",
           "ZAF", "SSD", "SDN", "TZA", "TGO", "TUN", "UGA", "ZMB", "ZWE"],
    "MERCOSUR": ["ARG", "BRA", "PRY", "URY", "BOL"],
    "MERCOSUR_ASSOC": ["CHL", "COL", "ECU", "GUY", "PER", "SUR"],
    "GCC": ["BHR", "KWT", "OMN", "QAT", "SAU", "ARE"],
    "SCO": ["CHN", "IND", "IRN", "KAZ", "KGZ", "PAK", "RUS", "TJK", "UZB", "BLR"],
    "ECOWAS": ["BEN", "BFA", "CPV", "CIV", "GMB", "GHA", "GIN", "GNB",
               "LBR", "MLI", "NER", "NGA", "SEN", "SLE", "TGO"],
    "BRICS": ["BRA", "RUS", "IND", "CHN", "ZAF", "EGY", "ETH", "IRN", "SAU", "ARE"],
    "G7": ["USA", "GBR", "FRA", "DEU", "ITA", "CAN", "JPN"],
    "G20": ["USA", "GBR", "FRA", "DEU", "ITA", "CAN", "JPN", "RUS", "CHN",
            "IND", "BRA", "MEX", "KOR", "AUS", "IDN", "SAU", "TUR", "ARG",
            "ZAF", "ARE"],
    "OPEC": ["DZA", "AGO", "COG", "GNQ", "GAB", "IRN", "IRQ", "KWT",
             "LBY", "NGA", "SAU", "ARE", "VEN"],
    "ARAB_LEAGUE": ["DZA", "BHR", "COM", "DJI", "EGY", "IRQ", "JOR", "KWT",
                    "LBN", "LBY", "MRT", "MAR", "OMN", "PSE", "QAT", "SAU",
                    "SOM", "SDN", "SYR", "TUN", "ARE", "YEM"],
    "COMMONWEALTH": ["AUS", "BGD", "BWA", "BRN", "CMR", "CAN", "CYP", "FJI",
                     "GHA", "GUY", "IND", "JAM", "KEN", "MYS", "MLT", "MUS",
                     "MOZ", "NAM", "NZL", "NGA", "PAK", "PNG", "RWA", "SGP",
                     "ZAF", "LKA", "SWZ", "TZA", "TTO", "TGA", "UGA", "GBR",
                     "VUT", "ZMB", "SLE", "GMB", "KIR", "WSM", "SLB", "NRU"],
    "CPTPP": ["AUS", "BRN", "CAN", "CHL", "JPN", "MYS", "MEX", "NZL",
              "PER", "SGP", "VNM", "GBR"],
    "FIVE_EYES": ["USA", "GBR", "CAN", "AUS", "NZL"],
    "QUAD": ["USA", "JPN", "AUS", "IND"],
    "RCEP": ["AUS", "BRN", "KHM", "CHN", "IDN", "JPN", "KOR", "LAO",
             "MYS", "MMR", "NZL", "PHL", "SGP", "THA", "VNM"],
    "OECD": ["AUS", "AUT", "BEL", "CAN", "CHL", "COL", "CRI", "CZE",
             "DNK", "EST", "FIN", "FRA", "DEU", "GRC", "HUN", "ISL",
             "IRL", "ISR", "ITA", "JPN", "KOR", "LVA", "LTU", "LUX",
             "MEX", "NLD", "NZL", "NOR", "POL", "PRT", "SVK", "SVN",
             "ESP", "SWE", "CHE", "TUR", "GBR", "USA"],
    "EEU": ["RUS", "BLR", "KAZ", "ARM", "KGZ"],
    "CSTO": ["RUS", "BLR", "KAZ", "ARM", "KGZ", "TJK"],
    "OAS": ["USA", "CAN", "MEX", "BRA", "ARG", "CHL", "COL", "PER",
            "VEN", "ECU", "BOL", "PRY", "URY", "GUY", "SUR", "PAN",
            "CRI", "GTM", "HND", "SLV", "NIC", "DOM", "HTI", "JAM",
            "TTO", "BHS", "BRB", "BLZ", "ATG", "DMA", "GRD", "KNA",
            "LCA", "VCT", "CUB"],
    "SADC": ["AGO", "BWA", "COD", "COM", "SWZ", "LSO", "MDG", "MWI",
             "MUS", "MOZ", "NAM", "SYC", "ZAF", "TZA", "ZMB", "ZWE"],
    "EAC": ["BDI", "COD", "KEN", "RWA", "SSD", "TZA", "UGA", "SOM"],
    "SAARC": ["AFG", "BGD", "BTN", "IND", "MDV", "NPL", "PAK", "LKA"],
    "CIS": ["RUS", "BLR", "KAZ", "UZB", "TJK", "KGZ", "ARM", "AZE", "MDA"],
    "NORDIC": ["DNK", "FIN", "ISL", "NOR", "SWE"],
    "VISEGRAD": ["CZE", "HUN", "POL", "SVK"],
    "FRANCOPHONIE": ["FRA", "BEL", "CHE", "LUX", "CAN", "SEN", "CIV",
                     "CMR", "COD", "COG", "GAB", "MLI", "NER", "BFA",
                     "TCD", "TUN", "MAR", "DZA", "MDG", "BEN", "TGO",
                     "GIN", "CAF", "DJI", "COM", "MRT", "RWA", "BDI",
                     "SYC", "VNM", "LAO", "KHM", "GRC", "ROU", "MDA",
                     "EGY", "LBN", "ARM", "GEO", "ALB", "MKD"],
}


def build_membership_vectors(countries: list[str]) -> np.ndarray:
    """Build binary membership matrix for all countries × organizations.

    Args:
        countries: List of ISO-3 codes

    Returns:
        (N, M) binary matrix where M is number of organizations
    """
    org_names = sorted(IGO_MEMBERSHIPS.keys())
    n = len(countries)
    m = len(org_names)

    membership = np.zeros((n, m), dtype=float)
    country_idx = {c: i for i, c in enumerate(countries)}

    for j, org in enumerate(org_names):
        members = IGO_MEMBERSHIPS[org]
        for iso3 in members:
            if iso3 in country_idx:
                membership[country_idx[iso3], j] = 1.0

    avg_memberships = membership.sum(axis=1).mean()
    logger.info("Membership matrix: %d countries × %d organizations, "
                 "avg %.1f memberships per country",
                 n, m, avg_memberships)
    return membership


def build_political_coupling(countries: list[str]) -> np.ndarray:
    """Build political coupling matrix from IGO co-membership.

    Uses cosine similarity of membership vectors:
        sim(i,j) = (m_i · m_j) / (||m_i|| × ||m_j||)

    Args:
        countries: List of ISO-3 codes

    Returns:
        (N, N) row-stochastic political coupling matrix
    """
    M = build_membership_vectors(countries)
    n = len(countries)

    # Compute cosine similarity
    norms = np.linalg.norm(M, axis=1, keepdims=True)
    norms[norms == 0] = 1  # Avoid division by zero
    M_norm = M / norms
    W = M_norm @ M_norm.T

    # Remove self-loops
    np.fill_diagonal(W, 0)

    # Keep only positive similarities (they should all be >= 0 for binary vectors)
    W = np.maximum(W, 0)

    # Row normalize
    row_sums = W.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    W = W / row_sums

    logger.info("Political coupling: %d×%d, density=%.3f",
                 n, n, (W > 0).mean())
    return W


def get_shared_organizations(iso_a: str, iso_b: str) -> list[str]:
    """Get list of organizations that both countries belong to.

    Useful for debugging and interpretability.

    Args:
        iso_a: First country ISO-3 code
        iso_b: Second country ISO-3 code

    Returns:
        List of organization names shared by both countries
    """
    shared = []
    for org, members in IGO_MEMBERSHIPS.items():
        if iso_a in members and iso_b in members:
            shared.append(org)
    return sorted(shared)
