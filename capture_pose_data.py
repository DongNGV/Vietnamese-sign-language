import cv2
import mediapipe as mp
import argparse
import sys
import os
from config import *
import numpy as np
from collections import deque
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.feature_extraction import extract_normalized_hand_features, extract_normalized_pose_result
import glob
from natsort import natsorted

# Temporarily ignore warning
import warnings
warnings.filterwarnings("ignore")

# Initialize MediaPipe Holistic
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Số khung hình liên tiếp cần thu thập cho mỗi mẫu thời gian thực
#TIMESTEPS = 30  # Có thể điều chỉnh tùy thuộc vào ứng dụng

def extract_temporal_features(hand_results_history, pose_results_history):
    """
    Trích xuất đặc trưng thời gian từ một chuỗi các khung hình tay và pose
    :param hand_results_history: deque chứa lịch sử kết quả tay
    :param pose_results_history: deque chứa lịch sử kết quả pose
    :return: numpy array với shape (timesteps, features)
    """
    # Khởi tạo mảng chứa đặc trưng thời gian
    temporal_features = []

    # Đảm bảo có đủ số khung hình
    if len(hand_results_history) < TIMESTEPS:
        feature_size = (FEATURES_PER_HAND * 4) + 18 + 4  # 84 + 18 + 4= 106
        return np.zeros((TIMESTEPS, feature_size))

    # Trích xuất đặc trưng cho từng khung hình trong lịch sử
    for hand_result, pose_result in zip(hand_results_history, pose_results_history):
        hand_features = extract_normalized_hand_features(mp_hands, hand_result, pose_result)
        pose_features = extract_normalized_pose_result(pose_result)
        frame_features = np.hstack((hand_features, pose_features))
        temporal_features.append(frame_features)

    # Chuyển thành mảng numpy với shape (timesteps, features)
    return np.array(temporal_features)

if __name__ == "__main__":
    # Get the pose name from argument
    parser = argparse.ArgumentParser("Pose Data Capture for CNN 3D")
    parser.add_argument("--pose_name", help="Name of the pose to be saved in the data folder",
                        type=str, default="test")
    parser.add_argument("--confidence", help="Confidence of the model",
                        type=float, default=0.2)
    parser.add_argument("--file_path", help="Đường dẫn thư mục chứa các hình ảnh",
                        type=str, default="error")
    args = parser.parse_args()

    # Lấy danh sách các file hình ảnh
    image_files = natsorted(glob.glob(os.path.join(args.file_path, "*.jpg")))

    if not image_files:
        print(f"Không tìm thấy hình ảnh trong thư mục: {args.file_path}")
        sys.exit(1)

    # Khởi tạo MediaPipe Hands và Pose
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=2,
                           min_detection_confidence=args.confidence,
                           min_tracking_confidence=args.confidence)

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=args.confidence,
                        min_tracking_confidence=args.confidence)

    # Khởi tạo deque để lưu trữ lịch sử khung hình
    hand_history = deque(maxlen=TIMESTEPS)
    pose_history = deque(maxlen=TIMESTEPS)

    # Mảng lưu trữ dữ liệu temporal
    temporal_pose_data = []

    # Tạo kết quả giả (zeros) để padding
    dummy_hand_result = type('', (), {})()  # Tạo một đối tượng giả
    dummy_hand_result.multi_hand_landmarks = None
    dummy_hand_result.multi_handedness = None
    dummy_pose_result = type('', (), {})()
    dummy_pose_result.pose_landmarks = None

    # Xử lý từng hình ảnh
    for img_path in image_files[6:len(image_files)-8]:
        image = cv2.imread(img_path)

        if image is not None:
            # Chuyển ảnh sang RGB
            image.flags.writeable = False
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Xử lý tay và pose
            hand_results = hands.process(image_rgb)
            pose_results = pose.process(image_rgb)

            # Lưu vào lịch sử
            hand_history.append(hand_results)
            pose_history.append(pose_results)

            # Khi đủ số khung hình (TIMESTEPS), trích xuất đặc trưng thời gian
            if len(hand_history) == TIMESTEPS:
                try:
                    temporal_features = extract_temporal_features(hand_history, pose_history)
                    temporal_pose_data.append(temporal_features)
                except Exception as e:
                    print(f"Lỗi khi trích xuất đặc trưng thời gian tại ảnh {img_path}: {e}")
                    continue

            """
            # Chuyển lại sang BGR để hiển thị (nếu cần)
            image.flags.writeable = True
            image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

            # Vẽ các điểm mốc tay
            if hand_results.multi_hand_landmarks:
                for hand_landmarks in hand_results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=hand_landmarks,
                        connections=mp_hands.HAND_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                        connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
                    )

            # Vẽ các điểm mốc pose
            if pose_results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=pose_results.pose_landmarks,
                    connections=mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=1, circle_radius=1),
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)
                )

            # Hiển thị hình ảnh
            cv2.imshow("Đọc video", image)

            # Nhấn 'q' để thoát
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            """

        else:
            print(f"Lỗi khi đọc ảnh: {img_path}")

    # Sau khi xử lý hết hình ảnh, nếu không đủ TIMESTEPS khung hình, thêm các giá trị 0
    while len(hand_history) < TIMESTEPS:
        hand_history.append(dummy_hand_result)
        pose_history.append(dummy_pose_result)

    # Trích xuất đặc trưng thời gian cho phần dữ liệu cuối cùng (nếu có dữ liệu)
    if len(hand_history) == TIMESTEPS and len(image_files) > 0:
        try:
            temporal_features = extract_temporal_features(hand_history, pose_history)
            temporal_pose_data.append(temporal_features)
        except Exception as e:
            print(f"Lỗi khi trích xuất đặc trưng thời gian cho phần dữ liệu cuối: {e}")

    # Chuyển dữ liệu thành mảng numpy
    if temporal_pose_data:
        temporal_pose_data = np.array(temporal_pose_data)  # Shape: (samples, timesteps, features)
        # Lưu dữ liệu
        save_path = f"data/{args.pose_name}.npy"
        np.save(save_path, temporal_pose_data)
        print(f"Dữ liệu temporal đã được lưu thành công tại {save_path}")
        print(f"Kích thước dữ liệu: {temporal_pose_data.shape}")
    else:
        print("Không có dữ liệu temporal nào được trích xuất.")

    # Giải phóng tài nguyên
    hands.close()
    pose.close()