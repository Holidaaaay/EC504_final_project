import cv2
import numpy as np

# 创建一个简单的黑白测试图像
def create_test_image(filename="test_image.png"):
    img = np.ones((200, 200), dtype=np.uint8) * 255  # 白色背景
    cv2.rectangle(img, (50, 50), (150, 150), 0, -1)  # 黑色正方形
    img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)  # 转换为 BGR 图像
    cv2.imwrite(filename, img_bgr)  # 保存图像
    print(f"Test image saved as {filename}")
    return img_bgr

# 生成测试图片并保存
test_img = create_test_image()
cv2.imshow("Test Image", test_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
