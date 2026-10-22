"""Future one-run acceptance entry. No build, training, or implicit approval."""
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tools'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('authorized_input')
    args=parser.parse_args()
    from export_adopted_reference import main as export
    export(['run',args.authorized_input])


if __name__=='__main__':main()
