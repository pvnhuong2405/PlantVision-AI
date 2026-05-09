from ultralytics import YOLO


#"D:/Project_PlantDisease/Train_YOLOv11/Test"
model = YOLO("D:/Project_PlantDisease/Train_YOLOv11/weights_yolov11s/weights_yolov11s/best.pt")
results = model.predict("D:/Project_PlantDisease/Pretrain_SAM/IMG_TEST_SAM/bacterial_spot.jpg",
                        conf = 0.45,
                        imgsz = 768,
                        save = False, # SAM trên img gốc 
                        save_txt = True,
                        project = "D:/Project_PlantDisease/Train_YOLOv11",
                        name = "img_reference",
                        device = 0,
                        exist_ok=True


                        )

