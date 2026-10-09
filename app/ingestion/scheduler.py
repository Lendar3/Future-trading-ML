from __future__ import annotations

import argparse

from app.ingestion.public_backfill import main as public_backfill_main


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a supported market-data ingestion job.")
    parser.add_argument(
        "--mode",
        choices=("public-backfill", "backfill"),
        default="public-backfill",
    )
    args, remaining = parser.parse_known_args()
    public_backfill_main(remaining)


if __name__ == "__main__":
    main()
