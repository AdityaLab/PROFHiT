import numpy as np
import pandas as pd


class TSNode(object):
    def __init__(self, idx, name, parent) -> None:
        self.idx = idx
        self.name = name
        self.parent = parent
        self.children = []


class LabourHierarchyData(object):
    """
    Defines a Hierarchical Dataset
    """

    def __init__(self, data_file="data/labour/data.csv") -> None:
        self.data_file = data_file
        self.data, self.idx_dict, self.nodes = self.get_hierarchy()

    def get_hierarchy(self):
        df = pd.read_csv(self.data_file)
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
        return data.T, idx_dict, nodes


class TourismHierarchyData(object):
    def __init__(self, data_file="data/tourismlarge/data.csv") -> None:
        self.data_file = data_file
        self.data, self.idx_dict1, self.nodes1 = self.get_hierarchy1()
        self.data, self.idx_dict2, self.nodes2 = self.get_hierarchy2()

    def get_hierarchy1(self):
        df = pd.read_csv(self.data_file)
        data = df.values[:, 1:]
        data = np.array([np.array(x, dtype=np.float32) for x in data])
        titles1 = df.columns.values[1:]
        titles1 = [x.replace("All", "") for x in titles1]
        titles1 = [x.replace("Total", "") for x in titles1]
        titles = []
        heir2 = ["Hol", "Vis", "Bus", "Oth"]

        def get_title_list(x):
            if x == "":
                return ["Total"]
            ans = []
            if x[-3:] in heir2:
                return None
            ans = [y for y in reversed(x)] + ans
            return ans

        titles = [get_title_list(x) for x in titles1]
        nodes = [TSNode(0, "Total", None)]
        idx_dict = {"Total": 0}
        ct = 0
        for n, t in enumerate(titles[1:]):
            if t is None:
                continue
            ct += 1
            idx_dict["_".join(t)] = ct
            if len(t) > 1:
                parent_name = "_".join(t[1:])
            else:
                parent_name = "Total"
            curr_node = TSNode(ct, "_".join(t), nodes[idx_dict[parent_name]])
            nodes.append(curr_node)
            nodes[idx_dict[parent_name]].children.append(curr_node)
        nodes = [x for x in nodes if x is not None]
        return data.T, idx_dict, nodes

    def get_hierarchy2(self):
        df = pd.read_csv(self.data_file)
        data = df.values[:, 1:]
        data = np.array([np.array(x, dtype=np.float32) for x in data])
        titles1 = df.columns.values[1:]
        titles1 = [x.replace("All", "") for x in titles1]
        titles1 = [x.replace("Total", "") for x in titles1]
        titles = []
        heir2 = ["Hol", "Vis", "Bus", "Oth"]

        def get_title_list(x):
            if x == "":
                return ["Total"]
            ans = []
            if x[-3:] in heir2:
                ans = [x[-3:]]
                x = x[:-3]
                ans = [y for y in reversed(x)] + ans
            else:
                return None
            return ans

        titles = [get_title_list(x) for x in titles1]
        nodes = [TSNode(0, "Total", None)]
        idx_dict = {"Total": 0}
        ct = 0
        for n, t in enumerate(titles[1:]):
            if t is None:
                continue
            ct += 1
            idx_dict["_".join(t)] = ct
            if len(t) > 1:
                parent_name = "_".join(t[1:])
            else:
                parent_name = "Total"
            curr_node = TSNode(ct, "_".join(t), nodes[idx_dict[parent_name]])
            nodes.append(curr_node)
            nodes[idx_dict[parent_name]].children.append(curr_node)
        nodes = [x for x in nodes if x is not None]
        return data.T, idx_dict, nodes


class WikiHierarchyData(object):
    """
    Defines a Hierarchical Dataset
    """

    def __init__(self, data_file="data/wiki2/data.csv") -> None:
        self.data_file = data_file
        self.data, self.idx_dict, self.nodes = self.get_hierarchy()

    def get_hierarchy(self):
        df = pd.read_csv(self.data_file)
        data = df.values[:, 1:]
        data = np.array([np.array(x, dtype=np.float32) for x in data])
        titles = df.columns.values[1:]
        titles = [x.split("_") for x in titles]
        titles = [list(reversed(x)) for x in titles]
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
        return data.T, idx_dict, nodes


def normalize_data(dataset):
    for node in reversed(dataset.nodes):
        if len(node.children) > 0:
            dataset.data[node.idx, :] /= len(node.children)
    return dataset


def unnormalize_data(dataset):
    for node in reversed(dataset.nodes):
        if len(node.children) > 0:
            dataset.data[node.idx, :] *= len(node.children)
    return dataset
