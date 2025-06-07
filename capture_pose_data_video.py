import cv2
import mediapipe as mp
import argparse
from utils.feature_extraction import *
import os
from PIL import Image, ImageEnhance
from collections import deque
from config import *
import numpy as np

# Temporarily ignore warning
import warnings
warnings.filterwarnings("ignore")

# Initialize MediaPipe Holistic
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

def extract_temporal_features(hand_results_history, pose_results_history):
    """
    Trích xuất đặc trưng thời gian từ một chuỗi các khung hình tay và pose
    :param hand_results_history: deque chứa lịch sử kết quả tay
    :param pose_results_history: deque chứa lịch sử kết quả pose
    :return: numpy array với shape (timesteps, features)
    """
    temporal_features = []
    if len(hand_results_history) < TIMESTEPS:
        feature_size = (FEATURES_PER_HAND * 4) + 18 + 4
        return np.zeros((TIMESTEPS, feature_size))

    for hand_result, pose_result in zip(hand_results_history, pose_results_history):
        hand_features = extract_normalized_hand_features(mp_hands, hand_result, pose_result)
        pose_features = extract_normalized_pose_result(pose_result)
        frame_features = np.hstack((hand_features, pose_features))
        temporal_features.append(frame_features)

    return np.array(temporal_features)

def tuTheBinhThuong(hand_results, pose_results):
    if not hand_results.multi_hand_landmarks:
        return True
    soTay = 0
    if pose_results.pose_landmarks:
        if pose_results.pose_landmarks.landmark[23].visibility > 0.5 and pose_results.pose_landmarks.landmark[24].visibility > 0.5:
            if pose_results.pose_landmarks.landmark[23].y < pose_results.pose_landmarks.landmark[24].y:
                eo = pose_results.pose_landmarks.landmark[23].y
            else:
                eo = pose_results.pose_landmarks.landmark[24].y
        elif pose_results.pose_landmarks.landmark[23].visibility > 0.5:
            eo = pose_results.pose_landmarks.landmark[23].y
        elif pose_results.pose_landmarks.landmark[24].visibility > 0.5:
            eo = pose_results.pose_landmarks.landmark[24].y
        else:
            eo = 1000
        if eo != 1000:
            for i, hand_landmarks in enumerate(hand_results.multi_hand_landmarks):
                if any(landmark.y > eo for landmark in hand_landmarks.landmark):
                    soTay += 1
    if soTay == 1 and len(hand_results.multi_hand_landmarks) == 1:
        return True
    elif soTay == 2:
        return True
    return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Pose Data Capture")
    parser.add_argument("--pose_name", help="Name of the pose to be save in the data folder", type=str, default="test")
    parser.add_argument("--confidence", help="Confidence of the model", type=float, default=0.6)
    parser.add_argument("--file_path", help="Name of the pose to be save in the data folder", type=str, default="")
    parser.add_argument("--data_list", help="Danh sách dữ liệu", type=str, default="data1")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.file_path)
    if not cap.isOpened():
        print("Không thể mở file video!")
        exit()

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=args.confidence, min_tracking_confidence=args.confidence)
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=args.confidence, min_tracking_confidence=args.confidence)

    hand_history = deque(maxlen=TIMESTEPS)
    pose_history = deque(maxlen=TIMESTEPS)
    temporal_pose_data = []
    dummy_hand_result = type('', (), {})()
    dummy_hand_result.multi_hand_landmarks = None
    dummy_hand_result.multi_handedness = None
    dummy_pose_result = type('', (), {})()
    dummy_pose_result.pose_landmarks = None
    mp_drawing = mp.solutions.drawing_utils
    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

    # Lấy tổng số khung hình
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    processed_frames = 0
    print(f"Total Frames: {frame_count}")

    while cap.isOpened():
        success, image = cap.read()
        if not success or processed_frames >= frame_count:
            print("Hết 1 tư thế")
            break

        processed_frames += 1
        # Bỏ qua 10 khung hình nếu không phải .webm
        current_pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
        if args.file_path.lower().endswith(".webm"):
            new_pos = current_pos  # Không bỏ qua khung hình cho .webm
        else:
            new_pos = current_pos + 3
        if new_pos < frame_count:
            cap.set(cv2.CAP_PROP_POS_FRAMES, new_pos)
        else:
            break

        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        enhancer = ImageEnhance.Sharpness(pil_image)
        image = cv2.cvtColor(np.array(enhancer.enhance(5.0)), cv2.COLOR_RGB2BGR)

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        hand_results = hands.process(image_rgb)
        pose_results = pose.process(image_rgb)

        if tuTheBinhThuong(hand_results, pose_results):
            continue

        hand_history.append(hand_results)
        pose_history.append(pose_results)

        if len(hand_history) == TIMESTEPS:
            try:
                temporal_features = extract_temporal_features(hand_history, pose_history)
                temporal_pose_data.append(temporal_features)
            except Exception as e:
                print(f"Lỗi khi trích xuất đặc trưng thời gian tại khung hình {processed_frames}: {e}")
                continue

        image.flags.writeable = True
        image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=hand_landmarks,
                    connections=mp_hands.HAND_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
                )

        if pose_results.pose_landmarks:
            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=pose_results.pose_landmarks,
                connections=mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=1, circle_radius=1),
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)
            )

        # Resize khung hình
        scale_percent = 50  # Giảm xuống 50% (có thể điều chỉnh)
        new_width = int(image.shape[1] * scale_percent / 100)
        new_height = int(image.shape[0] * scale_percent / 100)
        image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

        display_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        cv2.imshow("Đọc video", display_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    while len(hand_history) < TIMESTEPS:
        hand_history.append(dummy_hand_result)
        pose_history.append(dummy_pose_result)

    if temporal_pose_data:
        temporal_pose_data = np.array(temporal_pose_data)
        save_path = f"{args.data_list}/{args.pose_name}.npy"
        np.save(save_path, temporal_pose_data)
        print(f"Dữ liệu temporal đã được lưu thành công tại {save_path}")
        print(f"Kích thước dữ liệu: {temporal_pose_data.shape}")
    else:
        print("Không có dữ liệu temporal nào được trích xuất.")

    cap.release()
    cv2.destroyAllWindows()