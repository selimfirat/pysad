"""Print the path to bundled agent resources without loading the MCP server."""

import argparse
from pathlib import Path


def main() -> None:
    """Print the plugin path or its skills directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skills", action="store_true", help="Print the skills directory.")
    args = parser.parse_args()
    path = Path(__file__).resolve().parent
    print(path / "skills" if args.skills else path)


if __name__ == "__main__":
    main()
