import cv2
import time

cap = cv2.VideoCapture(0)

fps_arr = list(range(1, 6))
frame_num = 1

while True:
    delta = fps_arr[frame_num - 1] - fps_arr[frame_num]
    # self.fps.set(round(5 / delta))
    print(round(5 / delta))
    # print((fps_arr))
    fps_arr[frame_num % 5] = time.time()
    frame_num = (frame_num + 1) % 5

    ret, frame = cap.read()

    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
