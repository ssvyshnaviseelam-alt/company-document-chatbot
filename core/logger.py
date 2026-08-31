import logging


# ============================================
# Logger configuration
# ============================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)


# ============================================
# Application logger
# ============================================

logger = logging.getLogger("northstar")

