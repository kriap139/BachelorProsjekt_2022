


def angleCalcMAR(w, h, angle):
    if w < h:
        angle -= 90
    return angle


def watershedMAR(img: np.ndarray, display: TYDisplay):
    rembg_img = removeBKG(img)
    display("Background Removed", rembg_img)

    # getting the markers by using whatersheld from the rembg_img
    markers = watershed(rembg_img, display)

    # Finding Contours on Markers
    # cv2.RETR_EXTERNAL:Only extracts external contours
    # cv2.RETR_CCOMP: Extracts both internal and external contours organized in a two-level hierarchy
    # cv2.RETR_TREE: Extracts both internal and external contours organized in a  tree graph
    # cv2.RETR_LIST: Extracts all contours without any internal/external relationship
    contours_p, hierarchy_p = cv2.findContours(markers, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    # sort contours left-to-right
    sorted_contours_p = sorted(contours_p, key=lambda ctr: cv2.boundingRect(ctr)[0])

    # filtered Contours
    filter_arr = sorted_contours_p[1]
    for i in range(len(filter_arr)):
        # draw the rquaired contour
        cv2.drawContours(rembg_img, filter_arr, i, (0, 255, 0), 20)
    display("Required contour", rembg_img)

    # draw boundingRect around the rquaired contour
    x_p, y_p, w_p, h_p = cv2.boundingRect(filter_arr)
    cv2.rectangle(rembg_img, (x_p, y_p), (x_p + w_p, y_p + h_p), (255, 0, 0), 2)
    display("Pipe BBox", rembg_img)

    # Draw a rotated min area rectangle around the requaired contour
    minAreaPipe = cv2.minAreaRect(filter_arr)
    box_Pipe = cv2.boxPoints(minAreaPipe)
    box_Pipe = np.int0(box_Pipe)

    output_Pipe = cv2.drawContours(rembg_img, [box_Pipe], -1, (0, 0, 255), 5)
    display("MAR", rembg_img)

    # To find the angle for the pipe according to the x-axis
    (x_p, y_p), (w_p, h_p), ang_P = minAreaPipe

    # calculte the angle for the pipe
    angle_pipe = angleCalcMAR(w_p, h_p, ang_P)

    # colors
    color_Lower = (20, 100, 100)
    color_Upper = (30, 255, 255)

    # blur the orginal image to remove the noise
    blurred = cv2.GaussianBlur(img, (11, 11), 0)
    display("Blurred", blurred)

    # Convert the image to HSV colorspace
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    display("HSV", hsv)

    # Find the colors within the specified boundaries and apply the mask
    mask = cv2.inRange(hsv, color_Lower, color_Upper)
    display("MaksRaw", mask)

    # Deleting noises which are in area of mask
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    display("Maks", mask)

    # Find contours from the mask
    contours_h, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours_h) > 0:
        # get max contour
        c = max(contours_h, key=cv2.contourArea)

        # Draw a rotated min area rectangle around the max contour
        rect = cv2.minAreaRect(c)
        ((x_h, y_h), (w_h, h_h), angle_h) = rect

        # Finding the angle for the handventil
        angle_valve = angleCalcMAR(w_h, h_h, angle_h)

        # box
        box_h = cv2.boxPoints(rect)
        box_h = np.int64(box_h)

        # draw boundingRect around the detected contour for the pipe on the orginal image
        cv2.drawContours(img, [box_Pipe], 0, (0, 0, 255), 3)

        # draw boundingRect around the detected contour for the valve  on the orginal image
        cv2.drawContours(img, [box_h], 0, (255, 0, 0), 3)

        # display the orginal image
        display("OG image", img)

        return angle_pipe, angle_valve