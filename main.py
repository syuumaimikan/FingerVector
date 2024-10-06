import cv2
import mediapipe as mp
import numpy as np

# MediaPipe handsの設定
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# ベクトルの方向を計算する関数
def calculate_direction_vector(tip, base):
    return np.array(tip) - np.array(base)

# ベクトルの成分に基づいて方向を分類する関数
def classify_direction(vector):
    x, y, z = vector
    abs_x, abs_y, abs_z = np.abs(x), np.abs(y), np.abs(z)
    if abs_z > abs_x and abs_z > abs_y:
        return "前" if z > 0 else "後ろ"
    elif abs_x > abs_y:
        return "右" if x > 0 else "左"
    else:
        return "下" if y > 0 else "上"

# ピクセル座標に変換する関数
def to_pixel_coordinates(landmarks, width, height):
    return [np.array([lm.x * width, lm.y * height, lm.z * width]) for lm in landmarks]

# 手のポーズと指の方向を検出する関数
def detect_pose_and_direction(image, hands):
    results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if not results.multi_hand_landmarks:
        return image

    height, width, _ = image.shape
    for hand_landmarks in results.multi_hand_landmarks:
        landmarks = hand_landmarks.landmark
        tips = [landmarks[mp_hands.HandLandmark.THUMB_TIP],
                landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP],
                landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP],
                landmarks[mp_hands.HandLandmark.RING_FINGER_TIP],
                landmarks[mp_hands.HandLandmark.PINKY_TIP]]
        bases = [landmarks[mp_hands.HandLandmark.THUMB_MCP],
                 landmarks[mp_hands.HandLandmark.INDEX_FINGER_MCP],
                 landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_MCP],
                 landmarks[mp_hands.HandLandmark.RING_FINGER_MCP],
                 landmarks[mp_hands.HandLandmark.PINKY_MCP]]

        # ピクセル座標に変換
        tips_pixel = to_pixel_coordinates(tips, width, height)
        bases_pixel = to_pixel_coordinates(bases, width, height)

        # 親指のポーズを検出
        thumb_tip_pixel = tips_pixel[0] if len(tips_pixel) > 0 else None
        thumb_base_pixel = bases_pixel[0] if len(bases_pixel) > 0 else None

        if thumb_tip_pixel is not None and thumb_base_pixel is not None:
            thumb_direction_vector = calculate_direction_vector(thumb_tip_pixel, thumb_base_pixel)
            thumb_direction = classify_direction(thumb_direction_vector)

            # 親指が伸びていて、他の指が曲がっている場合
            thumb_extended = np.linalg.norm(thumb_tip_pixel - thumb_base_pixel) > 100
            other_fingers_curl = all(np.linalg.norm(tip - base) < 100 for tip, base in zip(tips_pixel[1:], bases_pixel[1:]))

            if thumb_extended and other_fingers_curl:
                print(f"親指の向き: {thumb_direction}")
                cv2.line(image, tuple(thumb_base_pixel[:2].astype(int)), tuple(thumb_tip_pixel[:2].astype(int)), (0, 255, 0), 2)
                cv2.circle(image, tuple(thumb_tip_pixel[:2].astype(int)), 5, (0, 0, 255), -1)  # 親指の先端をマーク
                continue  # 親指のポーズが検出された場合、他の指の検出をスキップ

        # 人差し指の検出
        index_tip_pixel = tips_pixel[1] if len(tips_pixel) > 1 else None
        index_base_pixel = bases_pixel[1] if len(bases_pixel) > 1 else None

        if index_tip_pixel is not None and index_base_pixel is not None:
            direction_vector = calculate_direction_vector(index_tip_pixel, index_base_pixel)
            direction = classify_direction(direction_vector)
            print(f"人差し指の向き: {direction}")

            cv2.line(image, tuple(index_base_pixel[:2].astype(int)), tuple(index_tip_pixel[:2].astype(int)), (255, 0, 0), 2)
            cv2.circle(image, tuple(index_tip_pixel[:2].astype(int)), 5, (0, 255, 0), -1)  # 先端をマーク

        # その他の指の先端と基部を描画
        for tip, base in zip(tips_pixel, bases_pixel):
            cv2.circle(image, tuple(tip[:2].astype(int)), 5, (255, 0, 0), -1)  # 指先をマーク
            cv2.circle(image, tuple(base[:2].astype(int)), 5, (0, 0, 255), -1)  # 基部をマーク

    return image

# ウェブカメラからのリアルタイム手の検出
def detect_from_webcam():
    cap = cv2.VideoCapture(0)  # ウェブカメラを使用してリアルタイムキャプチャ

    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.7)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = detect_pose_and_direction(frame, hands)

        cv2.imshow('手のポーズと指の方向検出', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    hands.close()

# リアルタイム手の検出を開始
detect_from_webcam()
