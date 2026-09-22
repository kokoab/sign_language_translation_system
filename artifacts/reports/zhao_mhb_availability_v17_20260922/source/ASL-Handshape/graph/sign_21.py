import sys

sys.path.extend(['../'])
from graph import tools

num_node = 21
self_link = [(i, i) for i in range(num_node)]

# inward_ori_index = [(22,23),(22,24),(22,26),(22,28),(22,30),
#                     (24,25),(26,27),(28,29),(30,31),
#                     (11,22)]
#                     #(5,12),(5,22),(6,12),(7,22),(6,22),(7,12),(12,22)]

inward_ori_index = [(1, 2), (2, 3), (4, 5), (5, 6), (7, 8),
                    (8, 9), (10, 11), (11, 12), (13, 14), (14, 15),
                    (16, 15), (17, 3), (18, 6), (19, 12), (20, 9),
                    (0, 13), (0, 1), (0, 4), (0, 10), (0, 7)
                    ]

inward = [(i, j) for (i, j) in inward_ori_index]
outward = [(j, i) for (i, j) in inward]
neighbor = inward + outward


class Graph:
    def __init__(self, labeling_mode='spatial'):
        self.A = self.get_adjacency_matrix(labeling_mode)
        self.num_node = num_node
        self.self_link = self_link
        self.inward = inward
        self.outward = outward
        self.neighbor = neighbor

    def get_adjacency_matrix(self, labeling_mode=None):
        if labeling_mode is None:
            return self.A
        if labeling_mode == 'spatial':
            A = tools.get_spatial_graph(num_node, self_link, inward, outward)
        else:
            raise ValueError()
        return A


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import os

    # os.environ['DISPLAY'] = 'localhost:11.0'
    A = Graph('spatial').get_adjacency_matrix()
    for i in A:
        plt.imshow(i, cmap='gray')
        plt.show()
    print(A)
