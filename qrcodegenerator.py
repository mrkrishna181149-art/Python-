# Import QRCode from pyqrcode
import pyqrcode
import png
from pyqrcode import QRCode

import cv2

# String which represents the QR code
s = input("Enter Text for QR : ")

# Generate QR code
url = pyqrcode.create(s)

# Create and save the svg file naming "myqr.svg"
url.svg("myqr.svg", scale = 8)

# Create and save the png file naming "myqr.png"
url.png('myqr.png', scale = 6)

img = cv2.imread("./myqr.png")

cv2.imshow("QR Code", img)

cv2.waitKey(0)
cv2.destroyAllWindows()