import sys
import json 
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--key", type=str)
    args = parser.parse_args()

    print(json.load(sys.stdin)['config'][args.key])