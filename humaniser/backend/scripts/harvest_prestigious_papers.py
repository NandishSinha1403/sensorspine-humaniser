"""
harvest_prestigious_papers.py — Query arXiv API for papers from top institutions
across varied disciplines. Saves PDFs to the 'research papers' directory.
"""

import arxiv
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("harvester")

# Configuration — use relative path resolution
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
TARGET_DIR = os.path.join(PROJECT_ROOT, "research papers")

SEARCH_QUERY = '(affiliation:"MIT" OR affiliation:"Harvard" OR affiliation:"Stanford" OR affiliation:"IIT" OR affiliation:"IISc")'
MAX_RESULTS = 30

# Multi-discipline categories
CATEGORIES = ["cs.AI", "cs.LG", "stat.ML", "physics.comp-ph", "math.OC", "q-bio.QM"]


def harvest_prestigious_papers():
    if not os.path.exists(TARGET_DIR):
        os.makedirs(TARGET_DIR)
        logger.info(f"Created directory: {TARGET_DIR}")

    # Get existing filenames to avoid duplicates
    existing_files = set(os.listdir(TARGET_DIR))
    existing_ids = set()
    for f in existing_files:
        # Extract arXiv ID from filename if present
        parts = f.split("_")
        if parts:
            existing_ids.add(parts[0].replace(".pdf", ""))

    client = arxiv.Client()
    search = arxiv.Search(
        query=SEARCH_QUERY,
        max_results=MAX_RESULTS,
        sort_by=arxiv.SortCriterion.Relevance,
    )

    logger.info(f"Searching for papers with query: {SEARCH_QUERY}")

    downloaded_count = 0
    for result in client.results(search):
        short_id = result.get_short_id()

        # Skip if already downloaded
        if short_id in existing_ids:
            logger.info(f"Skipping (already exists): {result.title[:50]}")
            continue

        # Create a safe filename
        safe_title = "".join([c if c.isalnum() else "_" for c in result.title[:50]])
        filename = f"{short_id}_{safe_title}.pdf"
        filepath = os.path.join(TARGET_DIR, filename)

        if not os.path.exists(filepath):
            logger.info(f"Downloading: {result.title[:60]}...")
            try:
                result.download_pdf(dirpath=TARGET_DIR, filename=filename)
                downloaded_count += 1
            except Exception as e:
                logger.error(f"Failed to download {result.title[:40]}: {e}")

    logger.info(f"Harvest complete. Downloaded {downloaded_count} new papers.")
    logger.info(f"Total papers in directory: {len(os.listdir(TARGET_DIR))}")


if __name__ == "__main__":
    harvest_prestigious_papers()
