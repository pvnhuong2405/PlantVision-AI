import os
import shutil

src_root = "D:/Project_PlantDisease/Train_CNN_Classes/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/valid"
list_change = ["Apple__", "Blueberry__", "Cherry_", "Corn_", "Grape__", "Orange__", "Peach__", "Pepper,_bell__", "Potato__", "Raspberry__", "Soybean__", "Strawberry__", "Tomato__"]
dst_root = "D:/Project_PlantDisease/Train_CNN_Classes/Dataset_new/valid"


# tạo folder apple nếu chưa có
os.makedirs(dst_root, exist_ok=True)

for folder in os.listdir(src_root):
    folder_path = os.path.join(src_root, folder)
    # startswith không nhận trực tiếp list, nó chỉ nhận tuple -> chuyển sang tuple
    if os.path.isdir(folder_path) and folder.startswith(tuple(list_change)):

        # Tạo thư mục đích theo tên cây
        #  lấy name trước __
        plant_name = folder.split("___")[0]

        plant_dst = os.path.join(dst_root, plant_name)
        os.makedirs(plant_dst, exist_ok=True)
        print("Processing:", folder)

        for file in os.listdir(folder_path):
            src_file = os.path.join(folder_path, file)

            if os.path.isfile(src_file):
                dst_file = os.path.join(plant_dst, file)

                # tránh trùng tên file
                base, ext = os.path.splitext(file)
                counter = 1
                while os.path.exists(dst_file):
                    dst_file = os.path.join(plant_dst, f"{base}_{counter}{ext}")
                    counter += 1

                shutil.copy2(src_file, dst_file)
print("Done")
