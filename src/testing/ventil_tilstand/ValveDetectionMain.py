#rembg_img = dm.removeBKG(img_original)
    #vs.display("Background Removed", rembg_img)

    # getting the markers by using whatersheld from the rembg_img
    #markers = dm.watershed(rembg_img, vs.display)

    # Finding Contours on Markers
    # cv2.RETR_EXTERNAL:Only extracts external contours
    # cv2.RETR_CCOMP: Extracts both internal and external contours organized in a two-level hierarchy
    # cv2.RETR_TREE: Extracts both internal and external contours organized in a  tree graph
    # cv2.RETR_LIST: Extracts all contours without any internal/external relationship
    #contours_p, hierarchy_p = cv2.findContours(markers, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    # sort contours left-to-right
    #sorted_contours_p = sorted(contours_p, key=lambda ctr: cv2.boundingRect(ctr)[0])

    # filtered Contours
    #filter_arr = sorted_contours_p[1]
    #for i in range(len(filter_arr)):
        # draw the rquaired contour
        #cv2.drawContours(rembg_img, filter_arr, i, (0, 255, 0), 20)
    #vs.display("Required contour", rembg_img)

    # draw boundingRect around the rquaired contour
    #x_p, y_p, w_p, h_p = cv2.boundingRect(filter_arr)

    # test
    #p_p1 = (int(x_p + 0.5 * w_p), y_p)
    #p_p2 = (p_p1[0], int(y_p * 0.5 * h_p))
    #vec_p = np.array((p_p2[0] - p_p1[0], p_p2[1] - p_p1[1]))
    #vec_p = vec_p / np.linalg.norm(vec_p)

    #cv2.arrowedLine(rembg_img, p_p2, p_p1, (218, 165, 32), thickness=6, tipLength=0.01)
    #vs.display("Test1", rembg_img)



    # Draw a rotated min area rectangle around the requaired contour
    #minAreaPipe = cv2.minAreaRect(filter_arr)
    #box_Pipe = cv2.boxPoints(minAreaPipe)
    #box_Pipe = np.int0(box_Pipe)

    #output_Pipe = cv2.drawContours(rembg_img, [box_Pipe], -1, (0, 0, 255), 5)
    #vs.display("MAR", rembg_img)

    # To find the angle for the pipe according to the x-axis
    #(x_p, y_p), (w_p, h_p), ang_P = minAreaPipe

    # calculte the angle for the pipe
    #angle_pipe = dm.angleCalcMAR(w_p, h_p, ang_P)

    # colors
    #color_Lower = (20, 100, 100)
    #color_Upper = (30, 255, 255)

    # blur the orginal image to remove the noise
    #blurred = cv2.GaussianBlur(img_original, (11, 11), 0)
    #vs.display("Blurred", blurred)

    # Convert the image to HSV colorspace
    #hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    #vs.display("HSV", hsv)

    # Find the colors within the specified boundaries and apply the mask
    #mask = cv2.inRange(hsv, color_Lower, color_Upper)
    #vs.display("MaksRaw", mask)

    # Deleting noises which are in area of mask
    #mask = cv2.erode(mask, None, iterations=2)
    #mask = cv2.dilate(mask, None, iterations=2)
    #vs.display("Maks", mask)

    # Find contours from the mask
    #contours_h, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #if len(contours_h) > 0:
        # get max contour
        #c = max(contours_h, key=cv2.contourArea)

        # Draw a rotated min area rectangle around the max contour
        #rect = cv2.minAreaRect(c)
        #((x_h, y_h), (w_h, h_h), angle_h) = rect

        # Finding the angle for the handventil
        #angle_valve = dm.angleCalcMAR(w_h, h_h, angle_h)

        # box
        #box_h = cv2.boxPoints(rect)
        #box_h = np.int64(box_h)

        # moment
        #M = cv2.moments(c)
        #center_h = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))



        #test
        #rows, cols = rembg_img.shape[:2]
        #(vx, vy, x, y) = cv2.fitLine(c, cv2.DIST_L12, 0, 0.01, 0.01)
        #lefty = int((-x * vy / vx) + y)
        #righty = int(((cols - x) * vy / vx) + y)
        #cv2.line(rembg_img, (cols - 1, righty), (0, lefty), (0, 255, 0), 2)
        #vs.display("Test2", rembg_img)

        #vec_v = np.array((vx, vy))
        #ang = np.arccos(np.dot(vec_p, vec_v)) * dm.RAD_TO_DEG
        #print(f"Angle test: {ang}")

        # draw boundingRect around the detected contour for the pipe on the orginal image
        #cv2.drawContours(img_original, [box_Pipe], 0, (0, 0, 255), 3)

        # draw boundingRect around the detected contour for the valve  on the orginal image
        #cv2.drawContours(img_original, [box_h], 0, (255, 0, 0), 3)

        # point in center
        #cv2.circle(img_original, center_h, 5, (255, 0, 255), 1)

        # display the orginal image
        #vs.display("OG image", img_original)