import sonnet_basic
import argparse

s = sonnet_basic.Sonnet_Gen()
parser = argparse.ArgumentParser()
parser.add_argument('--prompt', '-p', required=True)

#args = parser.parse_args()

poem = s.gen_poem_edwin("death", print_poem=True)
