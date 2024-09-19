import json
import argparse
from trainer import train

def main():
    args = setup_parser().parse_args()
    param = load_json(args.config)
    args = vars(args)  # Converting argparse Namespace to a dict.
    args.update(param)
    train(args)

def load_json(settings_path):
    with open(settings_path) as data_file:
        param = json.load(data_file)
    return param

def setup_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='stanfordcars',help="imagenetr , domainnet or cifar100_vit")
    parser.add_argument('--config', type=str, default='./exps/stanfordcars.json',
                        help='Json file of settings.')
    parser.add_argument('--topk', type=int, default=5)
    parser.add_argument('--root', type=str, default='./output')
    parser.add_argument('--only_learn_slot', default=False, action='store_true', help='only learn slots.')
    parser.add_argument('--lr', type=float, default=1e-2,
                        help='can be overwrite when specifying lr in json')
    # parser.add_argument('--device', nargs="+", type=int, default=[0],
    #                      help="The list of gpuid, ex:--gpuid 3 1. Negative value means cpu-only")

    parser.add_argument('--log_name', type=str, default='cgqa')
    parser.add_argument('--slot_log_name', type=str, default='cgqa-slot',
                        help='slot log name, can be overwrite when specifying it in json')
    parser.add_argument('--debug', default=False, action='store_true', help='print everything.')

    return parser


if __name__ == '__main__':
    main()
