import numpy as np
from config import *

def extract_hand_result(mp_hands, hand_results, pose_results):
    """
    Extract features from hand result
    :param mp_hands from mediapipe
    :param hand_results from media_pipe
    :return: features
    """
    # If multi-hand landmark is None -> Return zeros vector
    if hand_results.multi_hand_landmarks is None:
        # 21 x 2 zero array for each hand -> 21 x 4 zero array for two hands
        return np.zeros(FEATURES_PER_HAND * 4)

    # Get the number of hands
    num_hands = len(hand_results.multi_hand_landmarks)
    handedness = hand_results.multi_handedness

    # Handle handedness
    if num_hands == 1:
        # Check which hand
        hand_array = extract_single_hand(mp_hands, hand_results.multi_hand_landmarks[0])
        if handedness[0].classification[0].label == "Right":
            return np.hstack((hand_array.flatten(), np.zeros(FEATURES_PER_HAND * 2)))
        else:
            return np.hstack((np.zeros(FEATURES_PER_HAND * 2), hand_array.flatten()))
    else:
        zeroLeftHand = False
        zeroRightHand = False
        giatrivuot = 1

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
            else: eo = 1000

            if eo != 1000:
                # Kiểm tra và xử lý landmarks của tay
                for i, hand_landmarks in enumerate(hand_results.multi_hand_landmarks):
                    # Kiểm tra nếu tay vượt quá giới hạn
                    if any(landmark.y > eo for landmark in hand_landmarks.landmark):
                        # Loại bỏ landmarks của tay này
                        giatrivuot -= i
                        if handedness[0].classification[0].label == "Right":
                            if i == 0:
                                zeroLeftHand = True
                            else:
                                zeroRightHand = True
                        else:
                            if i == 0:
                                zeroRightHand = True
                            else:
                                zeroLeftHand = True

        if zeroLeftHand == True and zeroRightHand == True:
            return np.zeros(FEATURES_PER_HAND * 4)
        elif zeroLeftHand == True:
            hand_array = extract_single_hand(mp_hands, hand_results.multi_hand_landmarks[giatrivuot])
            return np.hstack((np.zeros(FEATURES_PER_HAND * 2), hand_array.flatten()))
        elif zeroRightHand == True:
            hand_array = extract_single_hand(mp_hands, hand_results.multi_hand_landmarks[giatrivuot])
            return np.hstack((hand_array.flatten(), np.zeros(FEATURES_PER_HAND * 2)))



        # Get the left and right hand
        if handedness[0].classification[0].label == "Right":
            left_hand = hand_results.multi_hand_landmarks[0]
            right_hand = hand_results.multi_hand_landmarks[1]
        else:
            left_hand = hand_results.multi_hand_landmarks[1]
            right_hand = hand_results.multi_hand_landmarks[0]

        # Get left and right hand
        left_hand_array = extract_single_hand(mp_hands, left_hand)
        right_hand_array = extract_single_hand(mp_hands, right_hand)


        return np.hstack((left_hand_array, right_hand_array)).flatten()


def extract_single_hand(mp_hands, hand_landmarks):
    # Create a 2D NumPy array to store the landmarks
    landmarks_array = np.zeros((FEATURES_PER_HAND, 2))

    # Function to safely get landmark coordinates
    def get_landmark(landmark):
        if landmark is None:
            return np.array([0.0, 0.0])
        return np.array([landmark.x, landmark.y])

    # Extract each landmark individually
    landmarks_array[0] = get_landmark(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST])
    landmarks_array[1] = get_landmark(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC])
    landmarks_array[2] = get_landmark(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP])
    landmarks_array[3] = get_landmark(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP])
    landmarks_array[4] = get_landmark(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP])
    landmarks_array[5] = get_landmark(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP])
    landmarks_array[6] = get_landmark(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP])
    landmarks_array[7] = get_landmark(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP])
    landmarks_array[8] = get_landmark(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP])
    landmarks_array[9] = get_landmark(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP])
    landmarks_array[10] = get_landmark(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP])
    landmarks_array[11] = get_landmark(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP])
    landmarks_array[12] = get_landmark(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP])
    landmarks_array[13] = get_landmark(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP])
    landmarks_array[14] = get_landmark(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP])
    landmarks_array[15] = get_landmark(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP])
    landmarks_array[16] = get_landmark(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP])
    landmarks_array[17] = get_landmark(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP])
    landmarks_array[18] = get_landmark(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP])
    landmarks_array[19] = get_landmark(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP])
    landmarks_array[20] = get_landmark(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP])

    return landmarks_array

def extract_features(mp_hands, hand_results, pose_results):
    hand_features = extract_normalized_hand_features(mp_hands, hand_results, pose_results)
    pose_features = extract_normalized_pose_result(pose_results)
    return np.hstack((hand_features, pose_features))



def extract_normalized_pose_result(pose_results):
    """
    Trích xuất và chuẩn hóa các điểm mốc pose quan trọng: 2 điểm miệng, 1 điểm mũi, 2 điểm mắt
    :param pose_results: Kết quả pose từ MediaPipe
    :return: Mảng numpy các đặc trưng pose đã được chuẩn hóa
    """
    selected_indices = [
        0,  # Mũi (nose)
        2,  # Mắt trái (left eye)
        5,  # Mắt phải (right eye)
        9,  # Miệng trái (mouth left)
        10,  # Miệng phải (mouth right)
        11,
        12,
        13,
        14
    ]
    # Số lượng điểm mốc cần lấy
    FEATURES_SELECTED = len(selected_indices)

    # Kiểm tra nếu không có pose landmarks
    if pose_results.pose_landmarks is None:
        return np.zeros(FEATURES_SELECTED * 2)  # Mỗi điểm có 2 tọa độ (x, y)

    # Tạo mảng để lưu trữ landmarks
    landmarks_array = np.zeros((FEATURES_SELECTED, 2))

    # Trích xuất các điểm mốc được chọn
    for i, idx in enumerate(selected_indices):
        landmark = pose_results.pose_landmarks.landmark[idx]
        landmarks_array[i] = [landmark.x, landmark.y]

    # Chuẩn hóa landmarks
    # 1. Tính toán điểm trọng tâm (centroid)
    centroid = np.mean(landmarks_array, axis=0)

    # 2. Dịch chuyển tất cả điểm landmarks về điểm gốc (centroid)
    normalized_landmarks = landmarks_array - centroid

    # 3. Normalize khoảng cách giữa các điểm
    # Tính khoảng cách Euclidean của các điểm so với centroid
    distances = np.linalg.norm(normalized_landmarks, axis=1)

    # Tính scale factor trung bình
    scale_factor = np.mean(distances)

    # Chia các landmarks cho scale factor để chuẩn hóa
    if scale_factor > 0:
        normalized_landmarks /= scale_factor

    # Trả về mảng đã làm phẳng
    return normalized_landmarks.flatten()


def normalize_landmarks(landmarks_array):
    """
    Chuẩn hóa landmarks bằng cách:
    1. Dịch chuyển về centroid
    2. Scale về khoảng cách chuẩn
    3. Duy trì thông tin tương đối của landmarks
    """
    # Kiểm tra đầu vào
    if landmarks_array.size == 0:
        return landmarks_array

    # Tính centroid
    centroid = np.mean(landmarks_array, axis=0)

    # Dịch chuyển landmarks về centroid
    normalized_landmarks = landmarks_array - centroid

    # Tính toán scale factor dựa trên khoảng cách
    distances = np.linalg.norm(normalized_landmarks, axis=1)
    scale_factor = np.mean(distances)

    # Normalize khoảng cách
    if scale_factor > 0:
        normalized_landmarks /= scale_factor

    return normalized_landmarks

def extract_normalized_hand_features(mp_hands, hand_results, pose_results):
    """
    Trích xuất features từ hand landmarks đã được chuẩn hóa, bao gồm vị trí trung tâm của 2 tay so với trung tâm của các pose landmarks được chọn
    :return: numpy array chứa features chuẩn hóa và vị trí trung tâm của 2 tay so với trung tâm pose landmarks được chọn
    """
    # Sử dụng hàm extract_hand_result gốc để lấy landmarks
    hand_features = extract_hand_result(mp_hands, hand_results, pose_results)

    # Reshape lại thành ma trận 2D
    num_landmarks = FEATURES_PER_HAND
    num_hands = 2

    normalized_features = []
    hand_positions = []  # Để lưu vị trí trung tâm của mỗi tay

    # Tính vị trí trung tâm và chuẩn hóa features cho từng tay
    for i in range(num_hands):
        start_idx = i * num_landmarks * 2
        end_idx = start_idx + num_landmarks * 2

        hand_landmarks = hand_features[start_idx:end_idx].reshape((num_landmarks, 2))

        # Chuẩn hóa landmarks của từng bàn tay
        normalized_hand = normalize_landmarks(hand_landmarks)

        # Tính vị trí trung tâm của tay (centroid)
        if np.any(hand_landmarks):  # Kiểm tra nếu tay tồn tại (khác zero)
            hand_center = np.mean(hand_landmarks, axis=0)  # [x, y]
        else:
            hand_center = np.array([0.0, 0.0])  # Nếu không có tay, trả về [0, 0]

        # Thêm vào danh sách
        normalized_features.extend(normalized_hand.flatten())
        hand_positions.extend(hand_center)  # Thêm tọa độ x, y của trung tâm tay

    # Chuyển hand_positions thành mảng numpy
    hand_positions = np.array(hand_positions).reshape((num_hands, 2))

    # Danh sách các điểm pose được chọn (tương tự extract_normalized_pose_result)
    selected_indices = [
        0,  # Mũi (nose)
        2,  # Mắt trái (left eye)
        5,  # Mắt phải (right eye)
        9,  # Miệng trái (mouth left)
        10, # Miệng phải (mouth right)
        11, # Vai trái (left shoulder)
        12, # Vai phải (right shoulder)
        13, # Cùi chỏ trái (left elbow)
        14  # Cùi chỏ phải (right elbow)
    ]

    # Tính vị trí trung tâm và scale factor dựa trên các điểm pose được chọn
    if pose_results.pose_landmarks is not None:
        # Lấy tọa độ của các điểm được chọn
        selected_landmarks = np.array([[pose_results.pose_landmarks.landmark[idx].x,
                                       pose_results.pose_landmarks.landmark[idx].y]
                                      for idx in selected_indices])
        pose_center = np.mean(selected_landmarks, axis=0)  # Trung tâm của các điểm được chọn
        # Tính scale factor dựa trên khoảng cách từ các điểm được chọn đến trung tâm
        pose_distances = np.linalg.norm(selected_landmarks - pose_center, axis=1)
        pose_scale_factor = np.mean(pose_distances) if np.any(pose_distances) else 1.0
    else:
        # Nếu không có pose landmarks, dùng tọa độ mặc định (0, 0) và scale factor mặc định
        pose_center = np.array([0.0, 0.0])
        pose_scale_factor = 1.0

    # Chuẩn hóa vị trí trung tâm của hai tay so với trung tâm các điểm pose được chọn
    if np.any(hand_positions):  # Kiểm tra nếu có ít nhất một tay tồn tại
        # 1. Dịch chuyển vị trí trung tâm của hai tay so với trung tâm pose
        normalized_hand_positions = hand_positions - pose_center

        # 2. Scale lại khoảng cách dựa trên pose_scale_factor
        if pose_scale_factor > 0:
            normalized_hand_positions /= pose_scale_factor
    else:
        # Nếu không có tay nào, trả về zeros
        normalized_hand_positions = np.zeros((num_hands, 2))

    # Kết hợp features chuẩn hóa với vị trí tay đã chuẩn hóa so với trung tâm pose
    final_features = np.hstack((
        np.array(normalized_features),         # Features chuẩn hóa
        normalized_hand_positions.flatten()    # Vị trí trung tâm của 2 tay so với trung tâm pose được chọn
    ))

    return final_features


