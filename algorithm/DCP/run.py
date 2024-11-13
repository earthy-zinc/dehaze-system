from .model import *

def dehaze(haze_image_path: str, output_image_path: str, *args, **kwargs):
    hazy_cv2 = cv2.imread(haze_image_path)
    I = hazy_cv2.astype('float64') / 255
    dark = DarkChannel(I, 15)
    A = AtmLight(I, dark)
    te = TransmissionEstimate(I, A, 15)
    t = TransmissionRefine(hazy_cv2, te)
    J = Recover(I, t, A, 0.1)
    cv2.imwrite(output_image_path, J * 255)

