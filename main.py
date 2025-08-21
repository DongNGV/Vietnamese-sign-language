import cv2
import mediapipe as mp
import threading
from collections import deque
import tensorflow as tf
import pickle
import numpy as np
import pandas as pd
import os

from utils.feature_extraction import extract_normalized_hand_features, extract_normalized_pose_result
from utils.speak import *
from utils.strings import *
from config import *
from mainTrainUser import FileManager, MainWindow
from utils.strings import ExpressionHandler
from edit_model import ExcelEditorDialog

import sys
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from pygame import mixer


def extract_temporal_features(mp_hands, hand_results_history, pose_results_history):
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


class ProcessingWorker(QObject):
    processed = pyqtSignal(QImage, str, object)
    text_update = pyqtSignal(str, list)
    speak_request = pyqtSignal(str)

    def __init__(self, model, mapping, model_confidence, is_auto_speak=True, show_landmarks=True, parent=None):
        super().__init__(parent)
        self.model = model
        self.mapping = mapping
        self.is_auto_speak = is_auto_speak
        self.show_landmarks = show_landmarks
        self.current_expression = "default"
        self.current_confidence = 0.0
        self.ChuoiKyTu = ""
        self.LastString = ""
        self.historys = []
        self.tgXet = 0
        self.predictions = deque(maxlen=10)
        self.normal_predictions = deque(maxlen=10)
        self.hand_history = deque(maxlen=TIMESTEPS)
        self.pose_history = deque(maxlen=TIMESTEPS)
        self.expression_handler = ExpressionHandler()
        # MediaPipe in worker
        self.mp_hands = mp.solutions.hands
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(max_num_hands=2,
                                         min_detection_confidence=model_confidence,
                                         min_tracking_confidence=model_confidence)
        self.pose = self.mp_pose.Pose(min_detection_confidence=model_confidence,
                                      min_tracking_confidence=model_confidence,
                                      model_complexity=0)
        self._processing = False

    @pyqtSlot(bool)
    def set_show_landmarks(self, value):
        self.show_landmarks = value

    @pyqtSlot(bool)
    def set_auto_speak(self, value):
        self.is_auto_speak = value

    @pyqtSlot()
    def reset_state(self):
        self.current_expression = "default"
        self.current_confidence = 0.0
        self.ChuoiKyTu = ""
        self.LastString = None
        self.historys.clear()
        self.tgXet = 0
        self.predictions.clear()
        self.normal_predictions.clear()
        self.hand_history.clear()
        self.pose_history.clear()

    def tuTheBinhThuong(self, hand_results, pose_results):
        if not hand_results.multi_hand_landmarks:
            return True
        soTay = 0
        if pose_results.pose_landmarks:
            if pose_results.pose_landmarks.landmark[23].visibility > 0.5 and pose_results.pose_landmarks.landmark[
                24].visibility > 0.5:
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

    def predict_with_cnn(self, temporal_features):
        try:
            features = temporal_features[np.newaxis, :, :, np.newaxis, np.newaxis]
            probabilities = self.model.predict(features, verbose=0)
            prediction_idx = np.argmax(probabilities, axis=1)[0]
            confidence = probabilities[0][prediction_idx]
            prediction = self.mapping.get(prediction_idx, "default")
            return {"prediction": prediction, "confidence": confidence}
        except Exception as e:
            print(f"Lỗi trong predict_with_cnn (worker): {e}")
            return {"prediction": "default", "confidence": 0.0}

    @pyqtSlot(object)
    def process_frame(self, image):
        if image is None:
            return
        # Allow dropping frames if one is processing
        if self._processing:
            return
        self._processing = True
        try:
            # Resize ảnh về kích thước nhỏ hơn để tăng tốc xử lý
            target_w = 640
            h0, w0 = image.shape[:2]
            if w0 > target_w:
                scale = target_w / float(w0)
                new_size = (int(w0 * scale), int(h0 * scale))
                image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False
            pose_results = self.pose.process(image_rgb)
            hand_results = self.hands.process(image_rgb)
            image_rgb.flags.writeable = True

            self.hand_history.append(hand_results)
            self.pose_history.append(pose_results)

            expression = "default"
            confidence = 0.0

            if len(self.hand_history) >= TIMESTEPS and len(self.pose_history) >= TIMESTEPS:
                self.tgXet += 1
                if self.tgXet > BO_QUA_KHUNG_HINH:
                    temporal_features = extract_temporal_features(self.mp_hands, self.hand_history, self.pose_history)
                    result = self.predict_with_cnn(temporal_features)
                    confidence = result['confidence']
                    expression = result['prediction']

                    if confidence != 'None' and float(confidence) > GIA_TRI_NHAN_TU_THE:
                        pass
                    else:
                        expression = "none"

                    self.expression_handler.receive(expression)
                    self.tgXet = 0

                    if self.tuTheBinhThuong(hand_results, pose_results):
                        expression = "default"
                        self.expression_handler.receive(expression)

                    if expression != "default":
                        self.predictions.append(expression)
                        majority_expression = max(set(self.predictions), key=self.predictions.count)
                        majority_count = sum(1 for x in self.predictions if x == majority_expression)
                        if majority_expression and majority_count >= GIA_TRI_NHAN_DANG and majority_expression != self.LastString:
                            self.expression_handler.receive(majority_expression)
                            message = self.expression_handler.get_message()
                            self.ChuoiKyTu += message + " "
                            self.LastString = majority_expression
                            self.predictions.clear()

                    self.normal_predictions.append(expression == "default")
                    normal_count = sum(1 for x in self.normal_predictions if x)

                    if normal_count >= 7:
                        if self.ChuoiKyTu != "":
                            curr_text = self.ChuoiKyTu.strip()
                            if curr_text:
                                self.historys.insert(0, curr_text)
                            if self.is_auto_speak:
                                self.speak_request.emit(curr_text)
                        self.ChuoiKyTu = ""
                        self.LastString = None
                        self.predictions.clear()
                        self.normal_predictions.clear()

                    self.current_expression = self.expression_handler.get_message()
                    self.current_confidence = f"{confidence:.5f}"

            # Draw landmarks if requested
            display_image = image_rgb.copy()
            if self.show_landmarks:
                if hand_results and hand_results.multi_hand_landmarks:
                    for hand_landmarks in hand_results.multi_hand_landmarks:
                        self.mp_drawing.draw_landmarks(
                            image=display_image,
                            landmark_list=hand_landmarks,
                            connections=self.mp_hands.HAND_CONNECTIONS,
                            landmark_drawing_spec=self.mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=1),
                            connection_drawing_spec=self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)
                        )

                if pose_results and pose_results.pose_landmarks:
                    self.mp_drawing.draw_landmarks(
                        image=display_image,
                        landmark_list=pose_results.pose_landmarks,
                        connections=self.mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=self.mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=1, circle_radius=1),
                        connection_drawing_spec=self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)
                    )

            display_image = cv2.flip(display_image, 1)
            display_image = cv2.convertScaleAbs(display_image, alpha=1.1, beta=10)

            h, w, ch = display_image.shape
            bytes_per_line = ch * w
            q_image = QImage(display_image.data, w, h, bytes_per_line, QImage.Format_RGB888)

            # Emit results
            self.text_update.emit(self.ChuoiKyTu, list(self.historys))
            self.processed.emit(q_image, self.current_expression, self.current_confidence)

        except Exception as e:
            print(f"Lỗi trong ProcessingWorker.process_frame: {e}")
        finally:
            self._processing = False


class NhanDangNgonNguTiengViet(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Mô hình nhận dạng ngôn ngữ ký hiệu Tiếng Việt")
        self.setGeometry(100, 100, 1200, 800)

        # Thư mục gốc dự án (thư mục chứa file này)
        self.base_dir = os.path.dirname(os.path.abspath(__file__))

        self.model = None
        self.cap = None
        self.timer = None
        self.show_landmarks = True
        self.is_camera_on = False
        self.is_speaking = False
        # Trạng thái âm thanh và các thuộc tính UI/logic
        self.is_auto_speak = True
        self.last_spoken_text = ""
        self.file_manager_window = None
        self.model_input = None
        self.predictions = deque(maxlen=10)
        self.normal_predictions = deque(maxlen=10)
        self.ChuoiKyTu = ""
        self.LastString = None
        self.tgXet = 0
        self.current_expression = "default"
        self.current_confidence = "--"
        self.historys = []
        self.hand_history = deque(maxlen=TIMESTEPS)
        self.pose_history = deque(maxlen=TIMESTEPS)
        self.processed_frames = 0
        self.frame_count = 0
        self.worker_thread = None
        self.worker = None
        # Cấu hình xử lý video: mục tiêu FPS và bước bỏ khung
        self.video_target_fps = 10  # tốc độ xử lý mục tiêu
        self.video_skip_step = 2    # bỏ qua bao nhiêu khung giữa các lần xử lý

        # Tải mô hình mặc định theo MODEL_NAME và ánh xạ lớp
        try:
            model_path = os.path.join(self.base_dir, "models", f"{MODEL_NAME}.keras")
            mapping_path = os.path.join(self.base_dir, "models", f"{MODEL_NAME}_mapping.pkl")
            if os.path.exists(model_path):
                self.model = tf.keras.models.load_model(model_path)
            if os.path.exists(mapping_path):
                with open(mapping_path, "rb") as f:
                    self.mapping = pickle.load(f)
            else:
                self.mapping = {}
        except Exception as e:
            print(f"Lỗi khi tải mô hình: {e}")
            self.mapping = {}

        # Khởi tạo bộ xử lý biểu thức và âm thanh
        self.expression_handler = ExpressionHandler()
        try:
            mixer.init()
        except Exception:
            pass

        # Khởi tạo UI
        self.setup_ui()

    @pyqtSlot(QImage, str, object)
    def on_processed(self, q_image, expression, confidence):
        # Cập nhật UI với ảnh đã xử lý và thông tin dự đoán
        scaled_image = q_image.scaled(
            self.camera_label.size(),
            Qt.KeepAspectRatio,
            Qt.FastTransformation
        )
        self.camera_label.setPixmap(QPixmap.fromImage(scaled_image))
        self.current_expression = expression if expression else "default"
        self.current_confidence = confidence if confidence else "--"
        self.prediction_label.setText(
            f"{'Đang chờ...' if self.current_expression == 'default' else self.current_expression}")
        self.confidence_label.setText(
            f"Độ tin cậy: {self.current_confidence if self.current_confidence != 'None' else '--'}")
        # highlight history item
        for i in range(self.action_list.count()):
            item = self.action_list.item(i)
            if self.current_expression in item.text():
                item.setBackground(QColor(0, 255, 0, 100))
            else:
                item.setBackground(QColor(255, 255, 255))

    @pyqtSlot(str, list)
    def on_text_update(self, text_accum, histories):
        self.ChuoiKyTu = text_accum
        self.text_display.setText(self.ChuoiKyTu)
        # Cập nhật lịch sử nếu thay đổi
        if histories is not None:
            self.historys = histories
            self.action_list.clear()
            for history in self.historys:
                self.action_list.addItem(history)

    @pyqtSlot(str)
    def on_speak_request(self, text):
        # Chỉ nói khi auto_speak đang bật ở GUI
        if self.is_auto_speak and text:
            threading.Thread(target=SpeakText, args=(text,)).start()

    def setup_ui(self):
        toolbar = QToolBar("Main Toolbar")
        self.addToolBar(toolbar)
        collect_action = QAction("Thu Thập Và Huấn Luyện Mô hình", self)
        collect_action.triggered.connect(self.open_file_manager)
        toolbar.addAction(collect_action)
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        splitter = QSplitter(Qt.Horizontal)
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        model_layout = QHBoxLayout()
        model_label = QLabel("Bộ ngôn ngữ:")
        self.model_input = QLineEdit(MODEL_NAME)
        self.model_input.setReadOnly(True)
        self.model_input.setFixedSize(300, 30)
        model_button = QPushButton("chọn bộ ngôn ngữ")
        model_button.clicked.connect(self.select_model)
        model_edit_button = QPushButton("Sửa bộ ngôn ngữ")
        model_edit_button.clicked.connect(self.edit_model)
        model_layout.addWidget(model_label)
        model_layout.addWidget(self.model_input)
        model_layout.addWidget(model_button)
        model_layout.addWidget(model_edit_button)
        model_layout.addStretch()
        left_layout.addLayout(model_layout)
        camera_frame = QFrame()
        camera_frame.setFrameStyle(QFrame.Panel | QFrame.Raised)
        camera_frame.setStyleSheet("""
                QFrame {
                    border: 2px solid #cccccc;
                    border-radius: 10px;
                    background-color: black;
                }
            """)
        camera_layout = QVBoxLayout(camera_frame)
        self.camera_label = QLabel()
        self.camera_label.setFixedSize(800, 600)
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setScaledContents(False)
        camera_layout.addWidget(self.camera_label)
        left_layout.addWidget(camera_frame)
        control_panel = QWidget()
        control_layout = QHBoxLayout(control_panel)
        button_style = """
            QPushButton {
                padding: 10px;
                border-radius: 5px;
                font-weight: bold;
                min-width: 130px;
            }
            QPushButton:checked {
                background-color: #f44336;
            }
            QPushButton[active=true] {
                background-color: #4CAF50;
            }
        """
        self.start_button = QPushButton("Bắt đầu Camera")
        self.start_button.setCheckable(True)
        self.start_button.setStyleSheet(button_style)
        self.start_button.clicked.connect(self.toggle_camera)
        control_layout.addWidget(self.start_button)
        self.landmark_button = QPushButton("Hiển thị Landmarks")
        self.landmark_button.setCheckable(True)
        self.landmark_button.setChecked(True)
        self.landmark_button.setStyleSheet(button_style)
        self.landmark_button.clicked.connect(self.toggle_landmarks)
        control_layout.addWidget(self.landmark_button)
        self.audio_button = QPushButton("Âm thanh: Bật")
        self.audio_button.setCheckable(True)
        self.audio_button.setChecked(True)
        self.audio_button.setStyleSheet(button_style)
        self.audio_button.clicked.connect(self.toggle_audio)
        control_layout.addWidget(self.audio_button)
        self.read_video = QPushButton("Dịch video")
        self.read_video.setChecked(True)
        self.read_video.setStyleSheet(button_style)
        self.read_video.clicked.connect(self.open_video_file)
        control_layout.addWidget(self.read_video)
        left_layout.addWidget(control_panel)
        right_panel = QWidget()
        right_panel.setMinimumWidth(500)
        right_layout = QVBoxLayout(right_panel)
        text_group = QGroupBox("Văn bản nhận dạng")
        text_group.setMaximumHeight(200)
        text_layout = QVBoxLayout()
        self.text_display = QTextEdit()
        self.text_display.setReadOnly(True)
        self.text_display.setMaximumHeight(150)
        self.text_display.setStyleSheet("""
            QTextEdit {
                background-color: white;
                border: 1px solid #cccccc;
                border-radius: 8px;
                padding: 10px;
                font-size: 20px;
                line-height: 1.5;
            }
        """)
        text_layout.addWidget(self.text_display)
        button_layout = QHBoxLayout()
        clear_button = QPushButton("Xóa văn bản")
        clear_button.clicked.connect(self.clear_text)
        button_layout.addWidget(clear_button)
        text_layout.addLayout(button_layout)
        text_group.setLayout(text_layout)
        right_layout.addWidget(text_group)
        info_group = QGroupBox("Thông Tin Nhận Dạng")
        info_layout = QVBoxLayout()
        self.prediction_label = QLabel("Đang chờ...")
        self.prediction_label.setStyleSheet("""
            QLabel {
                font-size: 24px;
                font-weight: bold;
                color: #2196F3;
                padding: 10px;
                background: #f8f9fa;
                border-radius: 5px;
            }
        """)
        info_layout.addWidget(self.prediction_label)
        self.confidence_label = QLabel("Độ tin cậy: --")
        self.confidence_label.setStyleSheet("""
            QLabel {
                font-size: 16px;
                color: #666;
                padding: 5px;
            }
        """)
        info_layout.addWidget(self.confidence_label)
        action_label = QLabel("Danh sách hành động:")
        action_label.setStyleSheet("font-weight: bold; margin-top: 10px;")
        info_layout.addWidget(action_label)
        self.action_list = QListWidget()
        self.action_list.setStyleSheet("""
            QListWidget {
                background-color: white;
                border: 1px solid #cccccc;
                border-radius: 4px;
                padding: 5px;
                font-size: 14px;
            }
            QListWidget::item {
                padding: 5px;
                border-bottom: 1px solid #eee;
            }
            QListWidget::item:selected {
                background-color: #e3f2fd;
                color: #1976D2;
            }
        """)
        self.action_list.setMinimumHeight(200)
        for history in self.historys:
            self.action_list.addItem(history)
        info_layout.addWidget(self.action_list)
        info_group.setLayout(info_layout)
        right_layout.addWidget(info_group)
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        main_layout.addWidget(splitter)
        self.statusBar().showMessage("Sẵn sàng")

    def open_video_file(self):
        if self.is_camera_on:
            self.toggle_camera()

        video_path, _ = QFileDialog.getOpenFileName(
            self, "Chọn file video", "", "Video Files (*.mp4 *.avi *.mov *.mkv *.webm)"
        )
        if not video_path:
            self.statusBar().showMessage("Không có file video nào được chọn")
            return

        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            QMessageBox.critical(self, "Lỗi", "Không thể mở file video!")
            self.statusBar().showMessage("Lỗi khi mở file video")
            return

        # Lấy thông tin video
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.processed_frames = 0
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        print(f"Video FPS: {fps}, Total Frames: {self.frame_count}")
        # Tính bước bỏ khung động dựa trên FPS video và FPS mục tiêu
        if fps and fps > 0:
            # ví dụ: fps=30, target=10 -> skip_step = floor(30/10) - 1 = 2 (đọc 1, bỏ 2)
            self.video_skip_step = max(1, int(fps / self.video_target_fps)) - 1
        else:
            self.video_skip_step = 2

        # Xóa bộ đệm
        self.hand_history.clear()
        self.pose_history.clear()
        self.clear_text()
        self.is_camera_on = False
        self.start_button.setText("Bắt đầu Camera")
        self.statusBar().showMessage(f"Đang xử lý video: {os.path.basename(video_path)}")

        # Đảm bảo có worker để xử lý video (tương tự camera)
        if not self.worker_thread or not self.worker:
            self.worker_thread = QThread()
            self.worker = ProcessingWorker(self.model, self.mapping, MODEL_CONFIDENCE,
                                           is_auto_speak=self.is_auto_speak,
                                           show_landmarks=self.show_landmarks)
            self.worker.moveToThread(self.worker_thread)
            self.worker.processed.connect(self.on_processed)
            self.worker.text_update.connect(self.on_text_update)
            self.worker.speak_request.connect(self.on_speak_request)
            self.worker_thread.start()

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        # Dùng nhịp theo FPS mục tiêu để UI mượt hơn
        interval_ms = int(1000 / (self.video_target_fps if self.video_target_fps > 0 else 10))
        self.timer.start(interval_ms)

    def update_frame(self):
        if self.cap is None:
            return
        # Đọc khung hình hiện tại
        success, image = self.cap.read()

        if self.is_camera_on == False:
            if not success or self.processed_frames >= self.frame_count:
                self.timer.stop()
                self.timer = None
                self.cap.release()
                self.cap = None
                self.camera_label.clear()
                self.statusBar().showMessage("Đã hoàn thành xử lý video")
                return

            # Tăng số khung hình đã xử lý
            self.processed_frames += 1
            # Bỏ qua khung hình
            current_pos = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
            skip = self.video_skip_step if hasattr(self, 'video_skip_step') and self.video_skip_step is not None else 2
            new_pos = current_pos + max(1, int(skip))
            if new_pos < self.frame_count:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, new_pos)
            else:
                # Nếu vượt quá số khung hình, dừng lại
                self.timer.stop()
                self.timer = None
                self.cap.release()
                self.cap = None
                self.camera_label.clear()
                self.statusBar().showMessage("Đã hoàn thành xử lý video")
                return

        try:
            # Gửi frame cho worker xử lý cho cả chế độ camera và video
            if self.worker and image is not None:
                self.worker.process_frame(image)
        except Exception as e:
            print(f"Lỗi trong update_frame: {e}")
            if not self.is_camera_on:
                self.timer.stop()
                self.timer = None
                if self.cap:
                    self.cap.release()
                self.cap = None
                self.camera_label.clear()
                self.statusBar().showMessage("Lỗi khi xử lý video")

    def edit_model(self):
        input_text = self.model_input.text()
        if not input_text:
            QMessageBox.warning(self, "Thông báo", "Vui lòng chọn bộ ngôn ngữ trước!")
            return
        mapping_file_path = os.path.join("./models/", input_text + ".xlsx")
        if not os.path.exists(mapping_file_path):
            QMessageBox.warning(self, "Thông báo", "File .xlsx không tồn tại!")
            return
        dialog = ExcelEditorDialog(mapping_file_path, self)
        if dialog.exec_():
            try:
                df = pd.read_excel(mapping_file_path, engine='openpyxl')
                mapping_dict = dict(zip(df['Key'], df['Value']))
                ExpressionHandler.MAPPING = mapping_dict
                print(ExpressionHandler.MAPPING)
                self.statusBar().showMessage("Đã cập nhật bộ ngôn ngữ thành công")
            except Exception as e:
                QMessageBox.warning(self, "Thông báo", f"Lỗi khi đọc file .xlsx: {str(e)}")

    def select_model(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Chọn bộ dữ liệu")
        dialog.setMinimumSize(400, 300)
        layout = QVBoxLayout(dialog)
        list_widget = QListWidget()
        model_dir = os.path.join(self.base_dir, "models")
        keras_files = [f for f in os.listdir(model_dir) if f.endswith('.keras')] if os.path.exists(model_dir) else []
        if not keras_files:
            list_widget.addItem("Không tìm thấy file .keras")
            list_widget.setEnabled(False)
        else:
            list_widget.addItems([os.path.splitext(f)[0] for f in keras_files])
        layout.addWidget(list_widget)
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        layout.addWidget(button_box)
        if dialog.exec_() == QDialog.Accepted and list_widget.currentItem() and list_widget.isEnabled():
            input_text = list_widget.currentItem().text()
            self.model_input.setText(input_text)
            self.model = tf.keras.models.load_model(os.path.join(self.base_dir, "models", f"{input_text}.keras"))
            with open(os.path.join(self.base_dir, "models", f"{input_text}_mapping.pkl"), "rb") as f:
                self.mapping = pickle.load(f)
            mapping_file_path = os.path.join(self.base_dir, "models", input_text + ".xlsx")
            if not os.path.exists(mapping_file_path):
                QMessageBox.warning(self, title="Thông báo", text="File .xlsx không tồn tại!")
                return
            try:
                df = pd.read_excel(mapping_file_path, engine='openpyxl')
                mapping_dict = dict(zip(df["Key"], df["Value"]))
                ExpressionHandler.MAPPING = mapping_dict
                print(ExpressionHandler.MAPPING)
            except Exception as e:
                QMessageBox.warning(self, title="Thông báo", text=f"Lỗi khi đọc file .xlsx: {str(e)}")
                return

    def open_file_manager(self):
        if self.file_manager_window is None or not self.file_manager_window.isVisible():
            self.file_manager_window = MainWindow()
            self.file_manager_window.show()
        else:
            self.file_manager_window.activateWindow()

    def toggle_camera(self):
        if self.timer is None:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                QMessageBox.critical(self, "Lỗi", "Không thể kết nối camera!")
                self.start_button.setChecked(False)
                return
            # Giảm độ phân giải và giảm độ trễ bộ đệm để mượt hơn
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            # Khởi tạo worker thread
            self.worker_thread = QThread()
            self.worker = ProcessingWorker(self.model, self.mapping, MODEL_CONFIDENCE,
                                           is_auto_speak=self.is_auto_speak,
                                           show_landmarks=self.show_landmarks)
            self.worker.moveToThread(self.worker_thread)
            # Kết nối tín hiệu
            self.worker.processed.connect(self.on_processed)
            self.worker.text_update.connect(self.on_text_update)
            self.worker.speak_request.connect(self.on_speak_request)
            self.worker_thread.start()

            self.timer = QTimer()
            self.timer.timeout.connect(self.update_frame)
            self.timer.start(30)
            self.is_camera_on = True
            self.start_button.setText("Dừng Camera")
        else:
            self.timer.stop()
            self.timer = None
            if self.cap:
                self.cap.release()
            self.cap = None
            self.camera_label.clear()
            self.is_camera_on = False
            self.start_button.setText("Bắt đầu Camera")
            self.statusBar().showMessage("Camera đã dừng")
            # Dừng worker thread
            if self.worker:
                # Ngắt kết nối tín hiệu để tránh callback sau khi đóng
                try:
                    self.worker.processed.disconnect(self.on_processed)
                    self.worker.text_update.disconnect(self.on_text_update)
                    self.worker.speak_request.disconnect(self.on_speak_request)
                except Exception:
                    pass
            if self.worker_thread:
                self.worker_thread.quit()
                self.worker_thread.wait()
            self.worker = None
            self.worker_thread = None

    def toggle_landmarks(self):
        self.show_landmarks = not self.show_landmarks
        self.landmark_button.setText("Landmarks: Bật" if self.show_landmarks else "Landmarks: Tắt")
        self.statusBar().showMessage(f"{'Hiện' if self.show_landmarks else 'Ẩn'} landmarks")
        if self.worker:
            self.worker.set_show_landmarks(self.show_landmarks)

    def toggle_audio(self):
        self.is_auto_speak = not self.is_auto_speak
        btn_text = "Âm thanh: Bật" if self.is_auto_speak else "Âm thanh: Tắt"
        self.audio_button.setText(btn_text)
        self.statusBar().showMessage(
            f"{'Đã bật' if self.is_auto_speak else 'Đã tắt'} chức năng đọc văn bản"
        )
        if self.worker:
            self.worker.set_auto_speak(self.is_auto_speak)

    def clear_text(self):
        self.text_display.clear()
        self.ChuoiKyTu = ""
        self.LastString = None
        self.predictions.clear()
        self.normal_predictions.clear()
        self.last_spoken_text = ""
        self.hand_history.clear()
        self.pose_history.clear()
        if self.worker:
            self.worker.reset_state()

    def speak_text(self):
        if self.is_speaking:
            return
        text = self.text_display.toPlainText().strip()
        if not text:
            return
        if text != self.last_spoken_text:
            self.historys.insert(0, text)
            self.last_spoken_text = text
            self.action_list.clear()
            for history in self.historys:
                self.action_list.addItem(history)
        self.is_speaking = True
        ts = threading.Thread(target=SpeakText, args=(text,))
        ts.start()
        ts.join()
        self.is_speaking = False

    def get_majority_vote(self, predictions):
        if not predictions:
            return None
        return max(set(predictions), key=predictions.count)

    def predict_with_cnn(self, temporal_features):
        try:
            features = temporal_features[np.newaxis, :, :, np.newaxis, np.newaxis]
            probabilities = self.model.predict(features, verbose=0)
            prediction_idx = np.argmax(probabilities, axis=1)[0]
            confidence = probabilities[0][prediction_idx]
            prediction = self.mapping.get(prediction_idx, "default")
            return {"prediction": prediction, "confidence": confidence}
        except Exception as e:
            print(f"Lỗi trong predict_with_cnn: {e}")
            return {"prediction": "default", "confidence": 0.0}

    def tuTheBinhThuong(self, hand_results, pose_results):
        if not hand_results.multi_hand_landmarks:
            return True
        soTay = 0
        if pose_results.pose_landmarks:
            if pose_results.pose_landmarks.landmark[23].visibility > 0.5 and pose_results.pose_landmarks.landmark[
                24].visibility > 0.5:
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

    def closeEvent(self, event):
        if self.cap:
            self.cap.release()
        try:
            mixer.quit()
        except:
            pass
        event.accept()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    app.setStyleSheet("""
        QMainWindow {
            background-color: #f0f0f0;
        }
        QPushButton {
            background-color: #2196F3;
            color: white;
            border: none;
            padding: 8px;
            border-radius: 4px;
            min-width: 100px;
        }
        QPushButton:hover {
            background-color: #1976D2;
        }
        QPushButton:pressed {
            background-color: #0D47A1;
        }
        QGroupBox {
            font-weight: bold;
            border: 1px solid #cccccc;
            border-radius: 6px;
            margin-top: 12px;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 3px 0 3px;
        }
        QListWidget {
            border: 1px solid #cccccc;
            border-radius: 4px;
        }
        QTextEdit {
            background-color: white;
            color: #333333;
            font-size: 16px;
        }
        QPushButton#clear_button {
            background-color: #f44336;
        }
        QPushButton#clear_button:hover {
            background-color: #d32f2f;
        }
    """)
    window = NhanDangNgonNguTiengViet()
    window.show()
    sys.exit(app.exec_())