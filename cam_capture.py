import cv2
import os

cap = cv2.VideoCapture(6)                       # 0번 카메라 연결
if cap.isOpened() :
    while True:
        ret, frame = cap.read()                 # 카메라 프레임 읽기
        if ret:
            cv2.imshow('camera',frame)          # 프레임 화면에 표시
            if cv2.waitKey(1) != -1:            # 아무 키나 누르면
                cv2.imwrite(f'{os.getcwd()}/D455capture/D455cal_13.png', frame)
                break
        else:
            print('no frame!')
            break
else:
    print('no camera!')

cap.release()
cv2.destroyAllWindows()