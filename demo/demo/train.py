from ultralytics import YOLO
import torch

def train2():
        # Load a model
        model = YOLO("ultralytics/cfg/models/11/yolo11.yaml")  # build a new model from YAML

        train_results = model.train(
            data="ultralytics/cfg/datasets/coco.yaml",  # data YAML
            epochs=260,
            imgsz=640, 
            device=[0,1], 
            batch=32,
            workers=0,
            save_period=10,
            exist_ok=True,
        )
        print(train_results)

if __name__ == "__main__":
    # if torch.cuda.is_available():
    #     print("CUDA is available. Using GPU.")
    # else:
    #     print("CUDA is not available. Using CPU.")
    train2()