from sklearn.cluster import KMeans
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 替换为你的图片路径
image_path = "puppy2.jpg"

# 加载图片并转换为RGB格式
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 将图片重塑为像素点的二维数组
pixel_values = image_rgb.reshape((-1, 3))
pixel_values = np.float32(pixel_values)

# 设置KMeans聚类的参数
k = 2  # 聚类数量
kmeans = KMeans(n_clusters=k, random_state=0, n_init=10)
labels = kmeans.fit_predict(pixel_values)
centers = np.uint8(kmeans.cluster_centers_)

# 将标签映射到聚类中心生成分割后的图片
segmented_image = centers[labels.flatten()]
segmented_image = segmented_image.reshape(image_rgb.shape)

# 保存分割后的图片
output_path = "segmented_image.jpg"
cv2.imwrite(output_path, cv2.cvtColor(segmented_image, cv2.COLOR_RGB2BGR))

# 显示原图和分割图
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(image_rgb)
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("Segmented Image (2 Clusters)")
plt.imshow(segmented_image)
plt.axis("off")

plt.show()

print(f"Segmented image saved as {output_path}")
