import numpy as np
import matplotlib.pyplot as plt


def compute_iou(boxes, centroids):
    inter_w = np.minimum(boxes[:, None, 0], centroids[None, :, 0])
    inter_h = np.minimum(boxes[:, None, 1], centroids[None, :, 1])
    inter = inter_w * inter_h

    area_boxes = boxes[:, 0] * boxes[:, 1]
    area_centroids = centroids[:, 0] * centroids[:, 1]

    union = area_boxes[:, None] + area_centroids[None, :] - inter
    union = np.maximum(union, 1e-8)

    iou = inter / union
    return iou


def kmeans_plusplus(boxes, k, max_iters=100, random_state=None):
    if random_state:
        np.random.seed(random_state)

    n = boxes.shape[0]
    centroids = [boxes[np.random.choice(n)]]

    for _ in range(k - 1):
        iou_matrix = compute_iou(boxes, np.array(centroids))
        max_iou = np.max(iou_matrix, axis=1)
        D = 1 - max_iou

        denominator = np.sum(D ** 2)
        if denominator == 0:
            probabilities = np.ones(n) / n
        else:
            probabilities = D ** 2 / denominator

        next_idx = np.random.choice(n, p=probabilities)
        centroids.append(boxes[next_idx])

    centroids = np.array(centroids)

    for _ in range(max_iters):
        iou_matrix = compute_iou(boxes, centroids)
        labels = np.argmax(iou_matrix, axis=1)

        new_centroids = np.zeros_like(centroids)
        for i in range(k):
            cluster_points = boxes[labels == i]
            if len(cluster_points) == 0:
                new_centroids[i] = boxes[np.random.choice(n)]
            else:
                new_centroids[i] = cluster_points.mean(axis=1)

        if np.allclose(centroids, new_centroids, atol=1e-6):
            break
        centroids = new_centroids

    return centroids


def compute_miou(boxes, centroids):
    iou_matrix = compute_iou(boxes, centroids)
    max_iou_per_box = np.max(iou_matrix, axis=1)
    return np.mean(max_iou_per_box)


if __name__ == "__main__":
    boxes = np.array([[10, 20], [30, 40], [50, 60], [70, 80]])

    k_values = range(1, 11)
    mious = []

    for k in k_values:
        centroids = kmeans_plusplus(boxes, k, random_state=42)
        miou = compute_miou(boxes, centroids)
        mious.append(miou)
        print(f"k={k}, mIoU={miou:.4f}")

    plt.figure(figsize=(10, 6))
    plt.plot(k_values, mious, marker='o', linestyle='-', color='b')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Average IoU')
    plt.title('Average IoU vs. Number of Clusters')
    plt.grid(True)
    plt.xticks(k_values)
    plt.show()
