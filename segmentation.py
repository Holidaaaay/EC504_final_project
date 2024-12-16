import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import maxflow

class Graph:
    def __init__(self, h, w):
        self.h = h
        self.w = w
        self.source = "source"
        self.sink = "sink"
        self.edges = {}
        self.capacities = {}

    def add_edge(self, u, v, capacity):
        if (u, v) not in self.edges:
            self.edges[(u, v)] = capacity
        else:
            self.edges[(u, v)] += capacity

        if (v, u) not in self.edges:
            self.edges[(v, u)] = 0

    def bfs(self, parent):
        visited = set()
        queue = [self.source]
        visited.add(self.source)

        while queue:
            u = queue.pop(0)
            for v in self.edges:
                if v[0] == u and v[1] not in visited and self.edges[(u, v[1])] > 0:
                    queue.append(v[1])
                    visited.add(v[1])
                    parent[v[1]] = u
                    if v[1] == self.sink:
                        return True
        return False

    def max_flow(self):
        parent = {}
        max_flow = 0

        while self.bfs(parent):
            path_flow = float("Inf")
            s = self.sink
            while s != self.source:
                path_flow = min(path_flow, self.edges[(parent[s], s)])
                s = parent[s]

            v = self.sink
            while v != self.source:
                u = parent[v]
                self.edges[(u, v)] -= path_flow
                self.edges[(v, u)] += path_flow
                v = parent[v]

            max_flow += path_flow

        return max_flow

def segment_image(file_path, bg_seed):
    img = cv2.imread(file_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Image not found at {file_path}")
    print(f"Image loaded with shape: {img.shape}")

    h, w, c = img.shape
    pixels = img.reshape((-1, 3))

    kmeans = KMeans(n_clusters=2, random_state=0)
    labels = kmeans.fit_predict(pixels)
    centers = kmeans.cluster_centers_

    distances = np.linalg.norm(pixels[:, None, :] - centers[None, :, :], axis=2)
    likelihoods = 1 / (1 + distances)
    normalized_scores = likelihoods / likelihoods.sum(axis=1, keepdims=True)
    foreground_scores = normalized_scores[:, 0]
    foreground_likelihood_map = foreground_scores.reshape((h, w))

    sigma_I = 30
    sigma_S = 5
    penalties = np.zeros((h, w, 4))

    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    grad_x = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    gradient_magnitude = cv2.normalize(gradient_magnitude, None, 0, 1, cv2.NORM_MINMAX)

    texture = cv2.Laplacian(gray_img, cv2.CV_64F)
    texture = cv2.normalize(np.abs(texture), None, 0, 1, cv2.NORM_MINMAX)

    for idx, (dx, dy) in enumerate([(0, -1), (0, 1), (-1, 0), (1, 0)]):
        valid_x_start = max(-dx, 0)
        valid_x_end = h + min(-dx, 0)
        valid_y_start = max(-dy, 0)
        valid_y_end = w + min(-dy, 0)

        shifted_x_start = valid_x_start + dx
        shifted_x_end = valid_x_end + dx
        shifted_y_start = valid_y_start + dy
        shifted_y_end = valid_y_end + dy

        diff = img[valid_x_start:valid_x_end, valid_y_start:valid_y_end] - \
               img[shifted_x_start:shifted_x_end, shifted_y_start:shifted_y_end]
        diff_squared = np.sum(diff ** 2, axis=-1)

        log_penalty = -diff_squared / (sigma_I ** 2)
        log_penalty = np.clip(log_penalty, -500, 0)
        color_penalty = np.exp(log_penalty)

        grad_diff = np.abs(gradient_magnitude[valid_x_start:valid_x_end, valid_y_start:valid_y_end] - \
                           gradient_magnitude[shifted_x_start:shifted_x_end, shifted_y_start:shifted_y_end])
        grad_penalty = np.exp(-grad_diff / (sigma_S ** 2))

        texture_diff = np.abs(texture[valid_x_start:valid_x_end, valid_y_start:valid_y_end] - \
                              texture[shifted_x_start:shifted_x_end, shifted_y_start:shifted_y_end])
        texture_penalty = np.exp(-texture_diff / (sigma_S ** 2))

        penalties[valid_x_start:valid_x_end, valid_y_start:valid_y_end, idx] = color_penalty * grad_penalty * texture_penalty

    def build_graph(likelihood, penalties, bg_seed):
        g = maxflow.Graph[float]()
        node_ids = g.add_grid_nodes((h, w))

        bg_x, bg_y = bg_seed

        for y in range(h):
            for x in range(w):
                fg_capacity = likelihood[y, x]
                bg_capacity = 1 - likelihood[y, x]

                if (y, x) == (bg_y, bg_x):
                    g.add_tedge(node_ids[y, x], 0, float('inf'))
                else:
                    g.add_tedge(node_ids[y, x], fg_capacity, bg_capacity)

        for idx, (dx, dy) in enumerate([(0, -1), (0, 1), (-1, 0), (1, 0)]):
            valid_x_start = max(-dx, 0)
            valid_x_end = h + min(-dx, 0)
            valid_y_start = max(-dy, 0)
            valid_y_end = w + min(-dy, 0)

            shifted_x_start = valid_x_start + dx
            shifted_x_end = valid_x_end + dx
            shifted_y_start = valid_y_start + dy
            shifted_y_end = valid_y_end + dy

            edge_penalties = penalties[valid_x_start:valid_x_end, valid_y_start:valid_y_end, idx]

            g.add_grid_edges(
                node_ids[valid_x_start:valid_x_end, valid_y_start:valid_y_end],
                weights=edge_penalties,
                symmetric=True
            )

        return g, node_ids

    graph, node_ids = build_graph(foreground_likelihood_map, penalties, bg_seed)

    graph.maxflow()
    segmentation = graph.get_grid_segments(node_ids)
    segmented_img = np.logical_not(segmentation).reshape((h, w))

    foreground_img = img.copy()
    foreground_img[~segmented_img] = [255, 255, 255]

    cv2.imwrite("foreground_result.png", foreground_img)
    print("Foreground result saved as 'foreground_result.png'")

    return foreground_img

