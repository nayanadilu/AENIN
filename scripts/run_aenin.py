"""Repository entry point for AENIN."""
from aenin.run import build_parser, run

if __name__ == "__main__":
    run(build_parser().parse_args())
