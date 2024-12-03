from io import BytesIO

from PIL import Image

from .model import *

def dehaze(haze_image: BytesIO, *args, **kwargs) -> BytesIO:
    haze = Image.open(haze_image).convert('RGB')
    hazy_cv2 = cv2.cvtColor(np.array(haze), cv2.COLOR_RGB2BGR)

    I = hazy_cv2.astype('float64') / 255
    dark = DarkChannel(I, 15)
    A = AtmLight(I, dark)
    te = TransmissionEstimate(I, A, 15)
    t = TransmissionRefine(hazy_cv2, te)
    J = Recover(I, t, A, 0.1)

    _, buffer = cv2.imencode('.jpg', J * 255)
    return BytesIO(buffer.tobytes())

