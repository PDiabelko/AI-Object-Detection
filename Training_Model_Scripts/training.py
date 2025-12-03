from ultralytics import YOLO

if __name__ == "__main__":
    # load a pretrained model
    model = YOLO('yolo11n.pt')

    # train the model
    results = model.train(data='coco128.yaml', epochs=3, imgsz=640)

