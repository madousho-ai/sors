"""decidophobia.data.mysqldump 的测试: 从 mysqldump 导出的 .sql 里读出表的行, 不需要 MySQL.

跑:  PYTHONPATH=src .venv/bin/python tests/test_mysqldump.py
"""

from _runner import run
from decidophobia.data.mysqldump import dump_tables, sql_rows


def test_sql_rows_reads_numbers_null_and_quoted_strings():
    line = "INSERT INTO `t` VALUES (1,NULL,'abc',-2,4391.98),(2,7,'',0,NULL);"
    assert sql_rows(line) == [[1, None, "abc", -2, 4391.98], [2, 7, "", 0, None]]


def test_sql_rows_undoes_mysqldump_escapes():
    r"""mysqldump 写 \n 是换行、\\ 是一个反斜杠、\' 与 '' 是单引号、\" 是双引号、\0 是 NUL."""
    line = r"INSERT INTO `t` VALUES ('a\nb','c\\nd','it\'s','it''s','say \"hi\"','x\0y');"
    assert sql_rows(line) == [["a\nb", "c\\nd", "it's", "it's", 'say "hi"', "x\0y"]]


def test_sql_rows_keeps_commas_and_parentheses_inside_strings():
    line = "INSERT INTO `t` VALUES (1,'a),(b, c'),(2,'(d)');"
    assert sql_rows(line) == [[1, "a),(b, c"], [2, "(d)"]]


def test_dump_tables_collects_the_named_tables_across_insert_lines_and_skips_the_rest():
    """一张大表在 dump 里分成好几行 INSERT; 没点名的表 (比如带手机号的 users) 一行都不解析."""
    lines = [
        "-- MySQL dump\n",
        "INSERT INTO `posts` VALUES (1,'a'),(2,'b');\n",
        "INSERT INTO `users` VALUES (9,'+98 912 000 0000');\n",
        "INSERT INTO `posts` VALUES (3,'c');\n",
        "INSERT INTO `tags` VALUES (1,0);\n",
    ]
    assert dump_tables(lines, ("posts", "tags")) == {"posts": [[1, "a"], [2, "b"], [3, "c"]], "tags": [[1, 0]]}


def test_dump_tables_gives_an_empty_list_for_a_named_table_with_no_rows():
    assert dump_tables([], ("posts",)) == {"posts": []}


if __name__ == "__main__":
    run(globals())
