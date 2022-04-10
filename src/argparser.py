import argparse
import sys


def word_eval_parser(parent_parser):
    subparser = parent_parser.add_argument_group("word_evaluation")
    subparser.add_argument('--thresholds', nargs='+', type=float, help="The threshold to use for cosine similarity word averaging evaluation")
    subparser.add_argument('--use-full-sentence', type=bool, help="Whether to use the full sentence or not for averaging. If not, we will use the words in-between the two entities (maybe some additional words to the left and o the right)")
    subparser.add_argument('--number-of-words-left-right', type=int, help='How many words left and right to use. Only when not using the full sentence')
    subparser.add_argument('--skip-unknown-words', type=bool, help="Whether to skip the unknown words when averaging or not")
    subparser.add_argument('--mode-of-application', type=str, help="Specifies the way we apply the rules. For example, we can do a cosine similarity between a sentence and all our rules and add the similarities obtained between a sentence and all the rules for a specific relation. Or we can only take the max similarity rule")
    subparser.add_argument('--rules-path', type=str, help="Where to find the rules")
    subparser.add_argument('--print-confusion-matrix', type=bool, help="Whether to print the confusion matrix or not")
    subparser.add_argument('--save-path', type=str, help='Where to save whatever is generated here. For example, where to save the rules generated.')
    subparser.add_argument('--no-relation-label', type=str, help="Specifies what is the 'no_relation' label for this dataset")

    return parent_parser

def dataset_details_parser(parent_parser):
    subparser = parent_parser.add_argument_group("dataset")
    subparser.add_argument('--dataset-path', type=str)
    subparser.add_argument('--dataset-name', type=str)

    return parent_parser

"""
Parameters for working with odinson from python
"""
def odinson_index_parser(parent_parser):
    subparser = parent_parser.add_argument_group("odinson")
    subparser.add_argument("--odinson-index-multiplicated", type=bool, help="A boolean flag, specifying if the odinson index is multiplied or not. Especially useful for rule application")
    subparser.add_argument("--odinson-index-basepath", type=str, help="The basepath to a folder containing multiple odinson indices")
    subparser.add_argument("--odinson-index-muliplication-name", type=str, help="The basename of a multiplicated index. An index will be at <odinson_index_basepath>/<odinson_index_muliplication_name>0, for example.")
    subparser.add_argument("--odinson-index-start", type=int, help="First index number (e.g. 0)")
    subparser.add_argument("--odinson-index-end", type=int, help="Last index number (e.g. 10)")
    subparser.add_argument("--odinson-index-dir", type=str, help="Path to an odinson index")

    return parent_parser

def generate_rules_parser(parent_parser):
    subparser = parent_parser.add_argument_group("rule generation")
    subparser.add_argument("--use-entities", type=bool, help="Whether to use the entities when generating a rule or not. If we use the entities, for a sentence like 'Microsoft was founded by Bill Gates', a rule can be: 'ORG was founded by PERSON'. Otherwise, it will be 'was founded by'")

    return parent_parser

def get_softrules_argparser():
    parser = argparse.ArgumentParser(description='Read paths to config files (last takes precedence). Can also update parameters with command-line parameters', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--path', nargs='+', type=str, default = ["config/default_config.yaml"], help='Path(s) to config file(s)')
    parser.add_argument('--basepath', type=str, required=False)
    word_eval_parser(parser)
    dataset_details_parser(parser)
    return parser


# python -m src.argparser --basepath "testbasepath" --thresholds 0.5 0.6 0.7 0.8 0.9 0.99 0.999 --use-full-sentence false --number-of-words-left-right 2 --skip-unknown-words True --mode-of-application 'apply_rules_with_threshold'
# python -m src.argparser --basepath "testbasepath" --gensim-fname 'abc'
# python -m src.argparser --dataset-path softrules/tacred/processed/dev.jsonl --dataset-name tacred --rules-path softrules/tacred/processed/train_rules2
if __name__ == "__main__":
    parser = get_softrules_argparser()

    args = parser.parse_args(sys.argv[1:])
    args = vars(args)
    print(args)
