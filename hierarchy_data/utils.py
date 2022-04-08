import numpy as np
import pandas as pd

from hierarchy_data import TSNode


def get_hierarchy(csv_file):
    df = pd.read_csv(csv_file)
    data = df.values[:, 1:]
    data = np.array([np.array(x, dtype=np.float32) for x in data])
    titles = df.columns.values[1:]
    titles = [x[2:-2] for x in titles]
    titles = [x.replace("'", "").split(",") for x in titles]
    titles = [[y.strip() for y in x] for x in titles]
    nodes = [TSNode(0, "Total", None)]
    idx_dict = {"Total": 0}
    for n, t in enumerate(titles[1:]):
        idx_dict["_".join(t)] = n + 1
        if len(t) > 1:
            parent_name = "_".join(t[1:])
        else:
            parent_name = "Total"
        curr_node = TSNode(n + 1, "_".join(t), nodes[idx_dict[parent_name]])
        nodes.append(curr_node)
        nodes[idx_dict[parent_name]].children.append(curr_node)
    return data, idx_dict, nodes
