import cv2
import numpy as np
from tensorflow.keras.models import load_model

def preprocess(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)
    return thresh

def sudoku_grid_detector(contours):
    largest_candidate = None
    max_area = 0
    for contour in contours:
        if cv2.contourArea(contour) > 10000:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.015 * peri, True)
            if len(approx) == 4:
                pts = np.squeeze(approx)
                if pts.ndim == 2 and pts.shape[0] == 4 and pts.shape[1] == 2:
                    area = cv2.contourArea(pts)
                    w = np.max(pts[:, 0]) - np.min(pts[:, 0])
                    h = np.max(pts[:, 1]) - np.min(pts[:, 1])
                    aspectratio = w / float(h)
                    if 0.85 < aspectratio < 1.15 and area > max_area:
                        largest_candidate = approx
                        max_area = area
    return largest_candidate

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # Top-left point
    rect[2] = pts[np.argmax(s)]  # Bottom-right point
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # Top-right point
    rect[3] = pts[np.argmax(diff)]  # Bottom-left point
    return rect

def warp(img, pts):
    pts = np.squeeze(pts)
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype="float32")
    transform_matrix = cv2.getPerspectiveTransform(rect, dst)
    warped_img = cv2.warpPerspective(img, transform_matrix, (maxWidth, maxHeight))
    return warped_img

def load_digit_recognition_model():
    return load_model('model/my_digit_recognition_model.h5')

def preprocess_for_model(cell):
    cell = cv2.resize(cell, (28, 28))
    cell = cell.astype("float32") / 255.0
    cell = np.expand_dims(cell, axis=-1)
    return cell

def predict_digit(cell, model):
    processed = preprocess_for_model(cell)
    prediction = model.predict(np.array([processed]))
    return np.argmax(prediction)

def segment_and_store_cells(warped_image):
    grid_size = 9
    cell_height = warped_image.shape[0] // grid_size
    cell_width = warped_image.shape[1] // grid_size
    cells = []
    for i in range(grid_size):
        for j in range(grid_size):
            cell = warped_image[i * cell_height:(i + 1) * cell_height, j * cell_width:(j + 1) * cell_width]
            cells.append(cell)
    return cells

def solve_sudoku(board):
    def is_valid(num, pos):
        # Check row
        for i in range(9):
            if board[pos[0]][i] == num and pos[1] != i:
                return False
        # Check column
        for i in range(9):
            if board[i][pos[1]] == num and pos[0] != i:
                return False
        # Check box
        box_x = pos[1] // 3
        box_y = pos[0] // 3
        for i in range(box_y * 3, box_y * 3 + 3):
            for j in range(box_x * 3, box_x * 3 + 3):
                if board[i][j] == num and (i, j) != (pos[0], pos[1]):
                    return False
        return True

    def solve():
        for i in range(9):
            for j in range(9):
                if board[i][j] == 0:
                    for num in range(1, 10):
                        if is_valid(num, (i, j)):
                            board[i][j] = num
                            if solve():
                                return True
                            board[i][j] = 0
                    return False
        return True
    
    solve()

cap = cv2.VideoCapture(0)
model = load_digit_recognition_model()

while True:
    success, frame = cap.read()
    if not success:
        break

    pp_frame = preprocess(frame)
    contours, _ = cv2.findContours(pp_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    largest_contour = sudoku_grid_detector(contours)
    if largest_contour is None:
        continue
    
    warped_image = warp(frame, largest_contour)
    cells = segment_and_store_cells(warped_image)
    
    sudoku_grid = [[0] * 9 for _ in range(9)]
    for index, cell in enumerate(cells):
        digit = predict_digit(cell, model)
        row, col = divmod(index, 9)
        sudoku_grid[row][col] = digit
    
    solve_sudoku(sudoku_grid)
    

    cv2.imshow("Sudoku Solver AR", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
