import argparse
import sys


def word_eval_parser(parent_parser):
    subparser = parent_parser.add_argument_group("word_evaluation")
    subparser.add_argument('--thresholds', nargs='+', type=float)
    subparser.add_argument('--use-full-sentence', type=str)
    subparser.add_argument('--number-of-words-left-right', type=str)
    subparser.add_argument('--skip-unknown-words', type=str)
    subparser.add_argument('--mode-of-application', type=str)

    return parent_parser

def get_softrules_argparser():
    parser = argparse.ArgumentParser(description='Read paths to config files (last takes precedence). Can also update parameters with command-line parameters', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--path', nargs='+', type=str, default = ["config/default_config.yaml"], help='Path(s) to config file(s)')
    parser.add_argument('--basepath', type=str, required=False)
    word_eval_parser(parser)
    return parser

# python -m src.argparser --basepath "testbasepath" --thresholds 0.5 0.6 0.7 0.8 0.9 0.99 0.999 --use-full-sentence false --number-of-words-left-right 2 --skip-unknown-words True --mode-of-application 'apply_rules_with_threshold'
# python -m src.argparser --basepath "testbasepath" --gensim-fname 'abc'
if __name__ == "__main__":
    parser = get_softrules_argparser()
    word_eval_parser(parser)
    args = parser.parse_args(sys.argv[1:])
    args = vars(args)
    print(args)
