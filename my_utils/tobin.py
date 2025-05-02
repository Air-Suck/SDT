import os
import cv2


def binary_images_in_directory(input_dir, output_dir):
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 遍历输入目录下的所有文件
    for filename in os.listdir(input_dir):
        # 检查文件是否是 PNG 图像
        if filename.endswith(".png"):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)

            # 读取图像
            image = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)

            # 检查是否成功读取图像
            if image is None:
                print(f"Failed to read {input_path}")
                continue

            # # 提取 Alpha 通道（如果存在）
            # if image.shape[2] == 4:  # BGRA 图像
            #     print("the shape of image is 4")
            #     alpha_channel = image[:, :, 3]
            # else:  # 如果没有 Alpha 通道，直接使用灰度图像
            alpha_channel = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # 二值化图像
            _, binary_image = cv2.threshold(alpha_channel, 200, 255, cv2.THRESH_BINARY)

            # 反转图像，使前景为黑色，背景为白色
            # output = cv2.bitwise_not(binary_image)

            # 保存二值化后的图像
            cv2.imwrite(output_path, binary_image)
            print(f"Processed and saved: {output_path}")


# 示例调用
input_directory = "input/handwrite_3"  # 替换为你的输入目录路径
output_directory = "output"  # 替换为你的输出目录路径
binary_images_in_directory(input_directory, output_directory)
