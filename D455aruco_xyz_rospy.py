#!/home/min/.pyenv/versions/pyenv_py3810/bin/python3.8

import cv2
import glob
import numpy as np
import math
from cv2 import aruco
import tf

import argparse
import sys
import os
import math

import rospy
import tf
from geometry_msgs.msg import Pose

aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
board = aruco.CharucoBoard_create(7, 5, 0.038, 0.03, aruco_dict)
cam = cv2.VideoCapture(6)         # usb cam
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

aruco_Pose_to_xArm = rospy.Publisher('aruco_Pose_to_xArm', Pose, queue_size=1)

def calibrate_charuco():
    allCorners = []
    allIds = []
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.00001)
    images = glob.glob(f'{os.getcwd()}/D455capture/*jpg')
    #imsize = None
    for im in images:
        print("=> Processing image {0}".format(im))
        frame = cv2.imread(im)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(gray, aruco_dict)

        if len(corners) > 0:
            for corner in corners:
                cv2.cornerSubPix(gray, corner,
                                 winSize=(3, 3),
                                 zeroZone=(-1, -1),
                                 criteria=criteria)
            res2 = cv2.aruco.interpolateCornersCharuco(corners, ids, gray, board)
            if res2[1] is not None and res2[2] is not None and len(res2[1]) > 3:
                allCorners.append(res2[1])
                allIds.append(res2[2])

        imsize = gray.shape

    return allCorners, allIds, imsize


def calibrate_camera(allCorners,allIds,imsize):
    print("CAMERA CALIBRATION")

    cameraMatrixInit = np.array([[392.17007616, 0., 307.67195922],
                                 [0., 392.74213265, 243.35366396],
                                 [0., 0., 1.]])

    distCoeffsInit = np.zeros((5,1))
    flags = (cv2.CALIB_USE_INTRINSIC_GUESS + cv2.CALIB_RATIONAL_MODEL + cv2.CALIB_FIX_ASPECT_RATIO)
    #flags = (cv2.CALIB_RATIONAL_MODEL)
    (ret, camera_matrix, distortion_coefficients0,
     rotation_vectors, translation_vectors,
     stdDeviationsIntrinsics, stdDeviationsExtrinsics,
     perViewErrors) = cv2.aruco.calibrateCameraCharucoExtended(
                      charucoCorners=allCorners,
                      charucoIds=allIds,
                      board=board,
                      imageSize=imsize,
                      cameraMatrix=cameraMatrixInit,
                      distCoeffs=distCoeffsInit,
                      flags=flags,
                      criteria=(cv2.TERM_CRITERIA_EPS & cv2.TERM_CRITERIA_COUNT, 10000, 1e-9))
    print("ret")
    print(ret)
    print("camera matrix")
    print(camera_matrix)
    print("distortion")
    print(distortion_coefficients0)

    return ret, camera_matrix, distortion_coefficients0, rotation_vectors, translation_vectors


def detect_marker(matrix, distortion):
    parameters = cv2.aruco.DetectorParameters_create()
    if cam.isOpened():
        while True:
            _, frame = cam.read()
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            coners, ids, point = cv2.aruco.detectMarkers(gray_frame, aruco_dict, parameters=parameters)
            if np.all(ids != None):

                rvecs, tvecs, _objPoints = cv2.aruco.estimatePoseSingleMarkers(coners, 0.03, matrix, distortion)
                frame = cv2.aruco.drawAxis(frame, matrix, distortion, rvecs[0], tvecs[0], 0.03)
                rvecs_msg = rvecs.tolist()
                tvecs_msg = tvecs.tolist()
                rvecs_msg_x = rvecs_msg[0][0][0]
                rvecs_msg_y = rvecs_msg[0][0][1]
                rvecs_msg_z = rvecs_msg[0][0][2]
                tvecs_msg_x = tvecs_msg[0][0][0]
                tvecs_msg_y = tvecs_msg[0][0][1]
                tvecs_msg_z = tvecs_msg[0][0][2]

                x = tvecs_msg_x
                y = tvecs_msg_y
                z = tvecs_msg_z
                y = -y

                euler_x_to_xArm = np.radians(-180)
                euler_y_to_xArm = np.radians(90)
                euler_z_to_xArm = np.arctan2(x, z)
                q_x, q_y, q_z, q_w = tf.transformations.quaternion_from_euler(euler_x_to_xArm, euler_y_to_xArm,
                                                                              euler_z_to_xArm)

                pose_to_xArm = Pose()
                pose_to_xArm.position.x = x
                pose_to_xArm.position.y = y
                pose_to_xArm.position.z = z
                pose_to_xArm.orientation.x = q_x
                pose_to_xArm.orientation.y = q_y
                pose_to_xArm.orientation.z = q_z
                pose_to_xArm.orientation.w = q_w
                aruco_Pose_to_xArm.publish(pose_to_xArm)
                



                # cv2.putText(frame, f"{round(d_euler_x)}, {round(d_euler_y)}, {round(d_euler_z)}", (0, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                #             (255, 255, 255))
                cv2.putText(frame, f"{euler_x_to_xArm}, {euler_y_to_xArm}, {euler_z_to_xArm}", (0, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))
                # print(rvecs_msg_x, " ", rvecs_msg_y, " ", rvecs_msg_z)

            frame = cv2.aruco.drawDetectedMarkers(frame, coners, ids)
            cv2.imshow('video', frame)
            k = cv2.waitKey(1) & 0xff          #press 'esc'to kill
            if k == 27:
                break
        cam.release()


def main():
    rospy.init_node("D455aruco_xyz_rospy")
    allCorners, allIds, imsize = calibrate_charuco()
    ret, mtx, dist, rvec, tvec = calibrate_camera(allCorners, allIds, imsize)
    detect_marker(mtx, dist)


if __name__ == "__main__":
    main()
