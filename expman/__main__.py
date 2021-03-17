import argparse

from .exp_group import ExpGroup


def add_param(args):
    exps = ExpGroup.gather(args.run)
    for exp in exps:
        exp.add_parameter(args.param, args.value)


def mv_param(args):
    exps = ExpGroup.gather(args.run)
    for exp in exps:
        exp.rename_parameter(args.param, args.new_param)


def rm_param(args):
    exps = ExpGroup.gather(args.run)
    for exp in exps:
        exp.remove_parameter(args.param)


def command_line():
    def guess(value):
        """ try to guess a python type for the passed string parameter """
        try:
            result = eval(value)
        except (NameError, ValueError):
            result = value
        return result

    parser = argparse.ArgumentParser(description='Experiment Manager Utilities')
    subparsers = parser.add_subparsers(dest='command')
    subparsers.required = True

    parser_add = subparsers.add_parser('add-param')
    parser_add.add_argument('run', default='runs/')
    parser_add.add_argument('param', help='new param name')
    parser_add.add_argument('value', type=guess, help='new param value')
    parser_add.set_defaults(func=add_param)

    parser_rm = subparsers.add_parser('rm-param')
    parser_rm.add_argument('run', default='runs/')
    parser_rm.add_argument('param', help='param to remove')
    parser_rm.set_defaults(func=rm_param)

    parser_mv = subparsers.add_parser('mv-param')
    parser_mv.add_argument('run', default='runs/')
    parser_mv.add_argument('param', help='param to rename')
    parser_mv.add_argument('new_param', help='new param name')
    parser_mv.set_defaults(func=mv_param)

    args = parser.parse_args()
    args.func(args)


command_line()
