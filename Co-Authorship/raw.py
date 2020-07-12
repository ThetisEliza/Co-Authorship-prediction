import os
from collections import defaultdict
import json
import csv
import re

FILES_PATH = "C:\\users\\73639\desktop\\comp90051 sml\\" \
             "assignment\\comp90051-2020-sem1-project-1"
NODE_FEATURE_FILE_NAME = "nodes.json"
TRAIN_EDGE_FILE_NAME = "train.txt"
PRED_EDGE_FILE_NAME = 'test-public.csv'

NODE_FILE = os.path.join(FILES_PATH, NODE_FEATURE_FILE_NAME)
TRAIN_FILE = os.path.join(FILES_PATH, TRAIN_EDGE_FILE_NAME)
PRED_FILE = os.path.join(FILES_PATH, PRED_EDGE_FILE_NAME)

TOTAL_AUTHOR = 4085


class RawData:
    def __init__(self):
        _, self.pred_edges = self._get_pred_edges()
        self.train_edges = self._get_train_edges()
        self.key_features = self._get_author_features()

    def _get_pred_edges(self):
        with open(PRED_FILE, 'r') as f:
            reader = csv.DictReader(f)
            id_list = []
            edge_list = []
            for row in reader:
                author1_id = row['Source']
                author2_id = row['Sink']
                edge_id = int(row['Id'])
                id_list.append(edge_id)
                edge_list.append((author1_id, author2_id))
            return id_list, edge_list

    def _get_train_edges(self):
        file_obj = open(TRAIN_FILE, 'r')
        edges_list = []
        linked_dict = defaultdict(list)

        try:
            for line in file_obj:
                strs = re.split('\\s+', line)
                author_id = strs[0]
                for id_str in strs[1:-1]:
                    if id_str not in linked_dict[author_id]:
                        edges_list.append((author_id, id_str))
                        linked_dict[id_str].append(author_id)
        finally:
            file_obj.close()
        return edges_list

    def _get_author_features(self):
        with open(NODE_FILE) as json_file:
            data = json.load(json_file)
            return data
