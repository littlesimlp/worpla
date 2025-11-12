import cv2
import numpy as np
from pyproj import Proj  # 用来作坐标转换的函数 UTM-经纬度


def tran2UTM(data_array):
    p1 = Proj(
        proj="utm",
        zone=48,
        ellps="WGS84",
        south=False,
        north=True,
        errcheck=True,
    )
    utm_list = []
    for i in range(len(data_array)):
        lon1, lat1 = p1(data_array[i][0], data_array[i][1])
        utm_list.append([lon1, lat1])
    return np.array(utm_list)


def main():
    # rtk points [rtk1,rtk2]
    _rtk = [
        [106.5345833, 29.7220109],
        [106.5346434, 29.72197556],
        [106.5346947, 29.72194521],
        [106.5347753, 29.72189746],
        [106.5346594, 29.72209176],
        [106.5346914, 29.72207602],
        [106.5347251, 29.72205637],
        [106.5347502, 29.72201252],
        [106.5347794, 29.7219946],
        [106.5348961, 29.72240835],
        [106.5349279, 29.7223953],
        [106.5349574, 29.72236953],
    ]

    # image point [imgx, imgy]
    _image = [
        [620.564460505575, 549.8061245905972],
        [948.030826686148, 537.9403967684567],
        [1200.4522413366906, 531.7995428909146],
        [1511.696572077372, 528.2443116986534],
        [650.3947632854573, 434.6597848587591],
        [768.9949189696653, 430.66206279324007],
        [900.7596081, 431.15874583670336],
        [1078.315904512556, 447.3259385930797],
        [1186.148925844414, 447.8985469195764],
        [617.2844127806981, 290.5479956898547],
        [672.001844267062, 292.1750550474779],
        [740.5281430231618, 296.01225391283225],
    ]

    utm_rtk = tran2UTM(_rtk)
    print(utm_rtk)
    dst_pts = np.array(_image, dtype=np.float32)
    H, status = cv2.findHomography(dst_pts, utm_rtk)

    np.set_printoptions(precision=18, suppress=False)
    print("H matrix:\n", H)


if __name__ == "__main__":
    main()
