"""Safe CLI entry point exposing only the scaffold status."""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Java Documentation RAG — scaffold only")
    parser.add_argument("--version", action="version", version="%(prog)s 0.1.0")
    parser.parse_args()
    parser.print_help()


if __name__ == "__main__":
    main()
