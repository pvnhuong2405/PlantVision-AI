import cv2
import numpy as np

def change_black_to_white(input_path, output_path, threshold=10):
    """
    Đổi nền đen thành trắng
    threshold: độ tối được xem là nền (0-255)
    """

    img = cv2.imread(input_path)

    # tạo mask cho pixel gần màu đen
    mask = np.all(img < threshold, axis=2)

    # đổi sang trắng
    img[mask] = [255,255,255]

    cv2.imwrite(output_path, img)


# test
input_img = "D:\\Project_PlantDisease\\Pretrain_SAM\\SAM_Output\\crops1\\crop_0.jpg"
output_img = "D:\\Project_PlantDisease\\IMG_TEST_CNN_AfterSAM\\crop_bacterial_spot_whiteb7.jpg"

change_black_to_white(input_img, output_img)

print("Saved:", output_img)