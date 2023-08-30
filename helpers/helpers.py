import re


def parse_sql_file(full_path, delimiter="-- @", name_delim="~"):
    """

    Args:
        full_path:
        delimiter:
        name_delim:

    Returns:

    """
    with open(full_path, mode="r", newline=None, encoding="utf-8") as f:
        combo_lines = f.read()

    parsed_sql = combo_lines.split(sep=delimiter)
    parsed_results = dict()

    for tmp_i, tmp_c in enumerate(parsed_sql):
        if tmp_c == "":
            continue

        tmp_c_split = tmp_c.split("\n")
        delim_index = [
            tmp_c_split.index(x)
            for x in tmp_c_split
            if re.search(pattern=name_delim, string=x) is not None
        ]

        tmp_name = re.sub(
            pattern=name_delim, repl="", string=tmp_c_split[delim_index[0]]
        )
        tmp_name = re.sub(pattern=r"\s+", repl="_", string=tmp_name)

        ind_to_remove_comm = [
            tmp_c_split.index(x)
            for x in tmp_c_split
            if re.search(pattern="--", string=x) is not None
        ]
        ind_to_remove_name = [
            tmp_c_split.index(x)
            for x in tmp_c_split
            if re.search(pattern=name_delim, string=x) is not None
        ]

        ind_to_remove = ind_to_remove_comm + ind_to_remove_name

        tmp_c_split = [i for j, i in enumerate(
            tmp_c_split) if j not in ind_to_remove]

        tmp_c_split = [i for j, i in enumerate(tmp_c_split) if i != ""]

        if len(tmp_c_split) < 1:
            continue

        tmp_c_final = "\n".join(tmp_c_split)
        parsed_results[tmp_name] = tmp_c_final

    return parsed_results
