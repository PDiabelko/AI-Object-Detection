from ultralytics import YOLO

if __name__ == "__main__":
    # load a pretrained model
    model = YOLO('yolo11s.pt')

    # train the model
    results = model.train(data='coco128.yaml', epochs=10, imgsz=640, patience=20)

