# import opencv
import cv2
# import hyperlpr3
import hyperlpr3 as lpr3

# Instantiate object
catcher = lpr3.LicensePlateCatcher()
# load image
#image = cv2.imread("沪AE97033.png")
# print result
# print(catcher(image))


try:
    
    camera = cv2.VideoCapture(0)     # 定义摄像头对象，参数0表示第一个摄像头，默认640x480
    camera.set(3, 320)
    camera.set(4, 320)

    while True:
        ret, frame = camera.read()

        cv2.imshow('frame', frame)
        cher=catcher(frame)
        if(cher):
            print(cher)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except:
    # picam2.stop()
    # picam2.close()
    cv2.destroyAllWindows()
    camera.release() 
