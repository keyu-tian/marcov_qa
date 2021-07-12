import datetime
import os


def time_str():
    return datetime.datetime.now().strftime('[%m-%d %H:%M:%S]')


def master_echo(is_master, msg: str, color='33', tail=''):
    if is_master:
        os.system(f'echo -e "\033[{color}m{msg}\033[0m{tail}"')