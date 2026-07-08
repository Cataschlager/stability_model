"""Run all data ingestion connectors.

Usage: python -m ingestion.run_all
"""

import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def main():
    """Run all connectors in dependency order."""
    from ingestion.imf_weo import IMFWEOConnector
    from ingestion.worldbank_wdi import WorldBankWDIConnector
    from ingestion.worldbank_wgi import WorldBankWGIConnector
    from ingestion.vdem import VDemConnector
    from ingestion.bulk_connectors import (
        Polity5Connector, FreedomHouseConnector, TransparencyIntlConnector,
        FSIConnector, SIPRIConnector, NDGAINConnector,
    )
    from ingestion.api_connectors import (
        ACLEDConnector, UCDPConnector, UNPopulationConnector,
        FAOConnector, BISBankingConnector, WHOGHOConnector,
    )
    from ingestion.linkage_connectors import (
        IMFDOTSConnector, CEPIIConnector, COWAllianceConnector, UNMigrantConnector,
    )

    # Phase 1: IMF WEO must run first (establishes country universe)
    connectors_phase1 = [
        ("IMF WEO (country universe)", IMFWEOConnector),
    ]

    # Phase 2: All other connectors (can run in any order after phase 1)
    connectors_phase2 = [
        ("World Bank WDI", WorldBankWDIConnector),
        ("World Bank WGI", WorldBankWGIConnector),
        ("V-Dem", VDemConnector),
        ("Polity5", Polity5Connector),
        ("Freedom House", FreedomHouseConnector),
        ("Transparency Intl CPI", TransparencyIntlConnector),
        ("Fragile States Index", FSIConnector),
        ("SIPRI Milex", SIPRIConnector),
        ("ND-GAIN", NDGAINConnector),
        ("ACLED", ACLEDConnector),
        ("UCDP/PRIO", UCDPConnector),
        ("UN Population", UNPopulationConnector),
        ("FAO Food Security", FAOConnector),
        ("BIS Banking", BISBankingConnector),
        ("WHO GHO Health", WHOGHOConnector),
        ("IMF DOTS", IMFDOTSConnector),
        ("CEPII GeoDist", CEPIIConnector),
        ("COW Alliance", COWAllianceConnector),
        ("UN Migrant Stock", UNMigrantConnector),
    ]

    results = {}
    total = len(connectors_phase1) + len(connectors_phase2)

    logger.info("=" * 60)
    logger.info("SPECTRAL INSTABILITY MODEL - DATA INGESTION")
    logger.info("=" * 60)

    # Phase 1
    logger.info("\n--- Phase 1: Establishing Country Universe ---")
    for name, cls in connectors_phase1:
        logger.info("\n[%d/%d] %s", len(results) + 1, total, name)
        try:
            connector = cls()
            df = connector.run()
            results[name] = {"status": "✅ OK", "rows": len(df)}
        except Exception as e:
            results[name] = {"status": f"❌ FAIL: {e}", "rows": 0}
            logger.error("CRITICAL: %s failed. Cannot proceed.", name)
            # If WEO fails, we can't continue
            if "country universe" in name.lower():
                logger.error("Country universe not established. Aborting.")
                sys.exit(1)

    # Phase 2
    logger.info("\n--- Phase 2: All Data Sources ---")
    for name, cls in connectors_phase2:
        logger.info("\n[%d/%d] %s", len(results) + 1, total, name)
        try:
            connector = cls()
            df = connector.run()
            results[name] = {"status": "✅ OK", "rows": len(df)}
        except Exception as e:
            results[name] = {"status": f"❌ FAIL: {e}", "rows": 0}
            logger.error("%s failed: %s", name, e)

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("INGESTION SUMMARY")
    logger.info("=" * 60)
    successes = sum(1 for r in results.values() if r["status"].startswith("✅"))
    failures = len(results) - successes
    for name, info in results.items():
        logger.info("  %-30s %s (%d rows)", name, info["status"], info["rows"])
    logger.info("\n  %d/%d succeeded, %d failed", successes, len(results), failures)

    if failures > 0:
        logger.warning("\n⚠️ Some connectors failed. The model can still run with partial data,")
        logger.warning("  but results may be less complete. See LIMITATIONS.md.")


if __name__ == "__main__":
    main()
