"""
mothertoken — core/pricing.py

Fetches and caches model pricing data for tokenizer cost benchmarking.
"""

import urllib.request
import json
import logging
from typing import Dict, Any

log = logging.getLogger("mothertoken")

LITELLM_PRICING_URL_TEMPLATE = "https://raw.githubusercontent.com/BerriAI/litellm/{commit}/model_prices_and_context_window.json"

def fetch_pricing_data(commit_hash: str) -> Dict[str, Any]:
    """Fetch the pricing JSON from the specified commit hash."""
    url = LITELLM_PRICING_URL_TEMPLATE.format(commit=commit_hash)
    log.info(f"Fetching pricing data from {url}")
    try:
        with urllib.request.urlopen(url) as response:
            data = json.loads(response.read().decode('utf-8'))
            return data
    except Exception as e:
        log.error(f"Failed to fetch pricing data from {url}: {e}")
        return {}

def get_model_input_cost(pricing_data: Dict[str, Any], cost_source_id: str) -> float:
    """Extract input_cost_per_token for a given cost_source_id."""
    if cost_source_id not in pricing_data:
        log.warning(f"No pricing data found for {cost_source_id}.")
        return 0.0
    
    model_info = pricing_data[cost_source_id]
    cost = model_info.get("input_cost_per_token", 0.0)
    return float(cost)
