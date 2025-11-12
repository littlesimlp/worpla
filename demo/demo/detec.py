from ultralytics import YOLO
import cv2
import numpy as np
import csv
from pyproj import Proj  # 用来作坐标转换的函数 UTM-经纬度

p1 = Proj(
    proj="utm",
    zone=48,
    ellps="WGS84",
    south=False,
    north=True,
    errcheck=True,
)


def projPoint(xywh, H):
    points = xywh[:, :2]
    points = points.reshape(-1, 1, 2).astype(np.float32)
    mapped_points = cv2.perspectiveTransform(points, H)  # 转 UTM 坐标系
    mapped_points = mapped_points.reshape(-1, 2)

    # 转经纬度坐标点
    lon, lat = p1(mapped_points[:, 0], mapped_points[:, 1], inverse=True)
    mapped_points = np.vstack([lon, lat]).T
    return mapped_points


def main(H):
    model = YOLO("yolo11n.pt")
    video = "test.mp4"
    cap = cv2.VideoCapture(video)

    is_visual = False  # 是否进行可视化
    frame_id = 0  # 视频帧 id
    with open("all_targets.csv", mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "id", "lon", "lat"])
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_id += 1
            if frame_id % 3 != 0:  # 每3帧取一帧
                continue

            results = model.track(source=frame, persist=True)
            bbx = results[0].boxes
            xywh = bbx.xywh.cpu().numpy()
            cls = list(bbx.cls.cpu().numpy())
            ids = list(bbx.id.cpu().numpy() if bbx.id is not None else None)
            mapped_points = projPoint(xywh, H)

            # 存储识别结果
            for od_id, xy in zip(ids, mapped_points):
                writer.writerow([frame_id, od_id, xy[0], xy[1]])

            if is_visual:
                annotated_frame = results[0].plot()
                cv2.imshow("Detection Result", annotated_frame)
                key = cv2.waitKey(1)
                if key == ord("q"):
                    break  # quit

    cv2.destroyAllWindows()


if __name__ == "__main__":
    # 根据 H_matrix.py 文件计算获得
    H = np.array(
        [
            [4.7850893820477989e02, -5.8755351975478598e03, 6.4838448141444067e05],
            [2.4274073558009218e03, -2.9802526830994153e04, 3.2888311128787268e06],
            [7.3803236496884194e-04, -9.0614219797347118e-03, 1.0000000000000000e00],
        ],
        dtype=np.float32,
    )
    main(H)
