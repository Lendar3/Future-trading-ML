from __future__ import annotations

import argparse

from app.ingestion.public_backfill import main as public_backfill_main


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="public-backfill")
    args, remaining = parser.parse_known_args()
    if args.mode in {"public-backfill", "backfill"}:
        public_backfill_main(remaining)
        return
    print(f"Ingestion scheduler placeholder for mode={args.mode}.")


if __name__ == "__main__":
    main()
