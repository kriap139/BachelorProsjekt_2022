import cv2
import numpy as np
from src.Backend.Util import LOG
from src.Backend import Config
from src.Backend.Valve import ValveState
from typing import Dict


def sortBiggestFirst(conts: np.ndarray):
    sc = sorted(conts, key=cv2.contourArea, reverse=True)
    return np.array(sc, dtype=object)


def createValveImages(imgPath: str) -> Dict[ValveState, np.ndarray]:
    img = cv2.imread(imgPath, cv2.IMREAD_UNCHANGED)

    if img is None:
        LOG(f"Failed to load Valve Image from path: {imgPath}")
        return

    # state, Background, Foreground, fillColor, (border, borderWidth) [117, 117, 117, 1]
    # OPEN: [17, 230, 0, 0.8]
    # CLOSED: [224, 34, 10, 1]
    # UNKNOWN: [2, 210, 238, 1]

    colorData = [
        (ValveState.UNKNOWN, [2, 210, 238, 1], [0, 0, 0, 1], None, None),
        (ValveState.OPEN, [0, 230, 17, 0.8], [0, 0, 0, 1], None, None),
        (ValveState.CLOSED, [10, 34, 224, 1], [0, 0, 0, 1], None, None)
    ]
    # (ValveState.OPEN, [], [], None),
    # (ValveState.CLOSED, [], [], None)

    res = {}

    for state, bkg, fg, fill, border in colorData:
        im = img.copy()

        mask_bkg: np.ndarray = im[:, :, 3] == 0
        mask_fg = np.invert(mask_bkg)

        im[mask_bkg] = bkg
        im[mask_fg] = fg

        new_img = cv2.cvtColor(im, cv2.COLOR_BGRA2BGR)
        blur = cv2.blur(new_img, (3, 3))

        if border is not None:
            gray = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
            ret, thresh = cv2.threshold(gray, 127, 255, 0)
            contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(blur, contours, -1, (0, 255, 0), 3)
            blur = cv2.blur(blur, (3, 3))

        if fill is not None:
            img_fill = img.copy()
            img_fill[mask_bkg] = [255, 255, 255, 255]
            img_fill_new = cv2.cvtColor(img_fill, cv2.COLOR_BGRA2BGR)

            gray = cv2.cvtColor(img_fill_new, cv2.COLOR_BGR2GRAY)
            ret, thresh = cv2.threshold(gray, 127, 255, 0)

            contours, hier = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            hier = hier[0]
            inner_contours = [c[0] for i, c in enumerate(zip(contours, hier)) if c[1][3] > 0]

            cv2.drawContours(blur, inner_contours, -1, fill, cv2.FILLED)

        res[state] = blur

        cv2.imshow(f"result {state}", blur)
        cv2.waitKey(0)

    cv2.destroyAllWindows()

    return res


if __name__ == "__main__":
    path = Config.getValveImagePath("images", "valve", fName="Valve.png")
    # images = createValveImages(path)

    # cf = Config.loadAppConfFile()
    # cf['valveImages'] = []

    # for state, img in images.items():
        # p = Config.createAppDataPath("images", "valve", fName=f"Valve_{state.name}.png")
        # cv2.imwrite(p, img)

        # cf['valveImages'].append(("images", "valve", f"Valve_{state.name}.png"))


    # Config.saveAppConfFile(cf)





