# -*- coding: utf-8 -*- 
# @Time : 2022/8/7 11:13 
# @Author : lepold
# @File : pprint.py

import prettytable as pt


def pretty_print(content: str):
    """
    pretty print something.

    """
    screen_width = 80
    text_width = len(content)
    box_width = text_width + 6
    left_margin = (screen_width - box_width) // 2
    print()
    print(' ' * left_margin + '+' + '-' * (text_width + 2) + '+')
    print(' ' * left_margin + '|' + ' ' * (text_width + 2) + '|')
    print(' ' * left_margin + '|' + content + ' ' * (box_width - text_width - 4) + '|')
    print(' ' * left_margin + '|' + ' ' * (text_width + 2) + '|')
    print(' ' * left_margin + '+' + '-' * (text_width + 2) + '+')
    print()


def table_print(content: dict, n_rows=None, n_columns=None):
    """
    display something in a table.

    Parameters
    ----------
    content
    n_rows
    n_columns

    Returns
    -------

    """
    assert isinstance(content, dict)
    if n_rows * n_columns < len(content):
        n_columns = 2
        n_rows = (len(content) + n_columns - 1) // n_columns
    tb = pt.PrettyTable()
    content_list = [list(x) for x in content.items()]
    content_list = sum(content_list, [])
    tb.field_names = ["name", "value"] * n_columns
    for i in range(n_rows):
        tb.add_row(content_list[n_columns * 2 * i:n_columns * 2 * (i+1)])
    print(tb)