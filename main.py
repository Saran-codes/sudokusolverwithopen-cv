import cv2 as cv2
import numpy as np
from tensorflow.keras.models import load_model


def loadmodel():
    model = load_model('model/my_model.h5')
    return model


cap = cv2.VideoCapture(0)

while True:
    success, self = cap.read()
    # convert the image to gray scale
    gray = cv2.cvtColor(self, cv2.COLOR_BGR2GRAY)
    # apply gaussian blur
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # apply gaussian threshold
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)
    # find all the contours
    contours = cv2.findContours(thresh, cv2.RETR_TREE,
                                cv2.CHAIN_APPROX_SIMPLE)[0]
    cv2.imshow("camera", self)

    # select the largest contour
    max_cnt = max(contours, key=cv2.contourArea)

    print(max_cnt)
    # create black mask
    mask = np.zeros(gray.shape, np.uint8)
    # filling the area inside contour with white pixels
    cv2.drawContours(mask, [max_cnt], 0, 255, -1)
    # filling the rest area with black pixels
    cv2.drawContours(mask, [max_cnt], 0, 0, 2)
    cv2.drawContours(self, contours, -1, (0, 255, 0), 3)

    # create a white mask
    out = 255 * np.ones_like(gray)
    # getting the original image which is marked by white pixels from the black mask.
    out[mask == 255] = gray[mask == 255]
    # return the generated white mask and largest contour
    # print(max_cnt)
    cv2.imshow("Contour Detection", mask)
    #cv2.imshow("dfg", thresh)
    cv2.imshow("vhj", out)
    #cv2.imshow("hdbv", self)

    peri = cv2.arcLength(max_cnt, True)
    # approximates the polygonal curves to detect vertices
    approx = cv2.approxPolyDP(max_cnt, 0.015 * peri, True)
    # flatten the vertices array
    pts = np.squeeze(approx)
    # find width of the puzzle
    box_width = np.max(pts[:, 0]) - np.min(pts[:, 0])
    # find height of the puzzle
    box_height = np.max(pts[:, 1]) - np.min(pts[:, 1])

    # following code is for warp perspective

    sum_pts = pts.sum(axis=1)
    diff_pts = np.diff(pts, axis=1)
    bounding_rect = np.array([pts[np.argmin(sum_pts)],
                              pts[np.argmin(diff_pts)],
                              pts[np.argmax(sum_pts)],
                              pts[np.argmax(diff_pts)]], dtype=np.float32)

    dst = np.array([[0, 0],
                    [box_width - 1, 0],
                    [box_width - 1, box_height - 1],
                    [0, box_height - 1]], dtype=np.float32)

    # transformation matrix
    transform_matrix = cv2.getPerspectiveTransform(bounding_rect, dst)
    # apply the transformation matrix to get the warped sudoku image
    warped_img = cv2.warpPerspective(out, transform_matrix, (box_width, box_height))

    cv2.imshow("fjv", warped_img)

    c_w = cv2.GaussianBlur(warped_img, (5, 5), 0)
    img = cv2.threshold(c_w, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    #cv2.imshow("dg", img)
    img[0:10, :] = 255
    img[:, 0:10] = 255
    img[-10:, :] = 255
    img[:, -10:] = 255

    img = np.bitwise_not(img) / 255

    cv2.imshow("jhb", img)
    # splitting the sudoku grid into small squares to extract digits
    rows = np.array_split(img, 9)
    digits = []
    cols = []
    for r in rows:
        cols.append(cv2.rotate(r, cv2.ROTATE_90_CLOCKWISE))
    for c in cols:
        cut = np.array_split(c, 9)
        for b in cut:
            digits.append(b)

    numbers = []

    for d in digits:
        numbers.append(cv2.rotate(d, cv2.ROTATE_90_COUNTERCLOCKWISE))

    cv2.imshow("0", numbers[0])
    cv2.imshow("1", numbers[1])
    cv2.imshow("2", numbers[2])
    cv2.imshow("3", numbers[3])
    cv2.imshow("4", numbers[4])
    cv2.imshow("5", numbers[5])
    cv2.imshow("6", numbers[6])
    cv2.imshow("7", numbers[7])
    cv2.imshow("8", numbers[8])
    cv2.imshow("9", numbers[9])
    cv2.imshow("10", numbers[10])
    cv2.imshow("11", numbers[11])
    cv2.imshow("12", numbers[12])
    cv2.imshow("13", numbers[13])
    cv2.imshow("14", numbers[14])
    cv2.imshow("15", numbers[15])
    cv2.imshow("16", numbers[16])
    cv2.imshow("17", numbers[17])
    cv2.imshow("18", numbers[18])
    cv2.imshow("19", numbers[19])
    cv2.imshow("20", numbers[20])
    cv2.imshow("21", numbers[21])
    cv2.imshow("22", numbers[22])
    cv2.imshow("23", numbers[23])
    cv2.imshow("24", numbers[24])
    cv2.imshow("25", numbers[25])
    cv2.imshow("26", numbers[26])
    cv2.imshow("27", numbers[27])
    cv2.imshow("28", numbers[28])
    cv2.imshow("29", numbers[29])
    cv2.imshow("30", numbers[30])
    cv2.imshow("31", numbers[31])
    cv2.imshow("32", numbers[32])
    cv2.imshow("33", numbers[33])
    cv2.imshow("34", numbers[34])
    cv2.imshow("35", numbers[35])
    cv2.imshow("36", numbers[36])
    cv2.imshow("37", numbers[37])
    cv2.imshow("38", numbers[38])
    cv2.imshow("39", numbers[39])
    cv2.imshow("40", numbers[40])
    cv2.imshow("41", numbers[41])
    cv2.imshow("42", numbers[42])
    cv2.imshow("43", numbers[43])
    cv2.imshow("44", numbers[44])
    cv2.imshow("45", numbers[45])
    cv2.imshow("46", numbers[46])
    cv2.imshow("47", numbers[47])
    cv2.imshow("48", numbers[48])
    cv2.imshow("49", numbers[49])
    cv2.imshow("50", numbers[50])
    cv2.imshow("51", numbers[51])
    cv2.imshow("52", numbers[52])
    cv2.imshow("53", numbers[53])
    cv2.imshow("54", numbers[54])
    cv2.imshow("55", numbers[55])
    cv2.imshow("56", numbers[56])
    cv2.imshow("57", numbers[57])
    cv2.imshow("58", numbers[58])
    cv2.imshow("59", numbers[59])
    cv2.imshow("60", numbers[60])
    cv2.imshow("61", numbers[61])
    cv2.imshow("62", numbers[62])
    cv2.imshow("63", numbers[63])
    cv2.imshow("64", numbers[64])
    cv2.imshow("65", numbers[66])
    cv2.imshow("67", numbers[67])
    cv2.imshow("68", numbers[68])
    cv2.imshow("69", numbers[69])
    cv2.imshow("70", numbers[70])
    cv2.imshow("71", numbers[71])
    cv2.imshow("72", numbers[72])
    cv2.imshow("73", numbers[73])
    cv2.imshow("74", numbers[74])
    cv2.imshow("75", numbers[75])
    cv2.imshow("76", numbers[76])
    cv2.imshow("77", numbers[77])
    cv2.imshow("78", numbers[78])
    cv2.imshow("79", numbers[79])
    cv2.imshow("80", numbers[80])




    cnn = loadmodel()






    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
