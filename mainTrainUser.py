import sys
import os
import subprocess
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, \
    QPushButton, QCheckBox, QListWidget, QFileDialog, QListWidgetItem, QStackedWidget, QMessageBox, QToolBar, QAction
from PyQt5.QtCore import QDir, Qt, center
import pandas as pd

class FileItemWidget(QWidget):
    def __init__(self, file_name, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout()
        self.label = QLabel(file_name)
        self.checkbox = QCheckBox()
        self.checkbox.setChecked(True)
        layout.addWidget(self.label)
        layout.addWidget(self.checkbox)
        self.setLayout(layout)
        self.file_name = file_name


class FileManager(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        main_layout = QVBoxLayout(self)

        # Tiêu đề
        title_layout = QHBoxLayout()
        collect_label = QHBoxLayout()
        collect_label = QLabel("THU THẬP DỮ LIỆU")
        collect_label.setAlignment(Qt.AlignCenter)
        collect_label.setStyleSheet("font-size: 30px; font-weight: bold;")
        title_layout.addWidget(collect_label)
        main_layout.addLayout(title_layout)

        # Đường dẫn và nút chọn vị trí
        path_layout = QHBoxLayout()
        path_label = QLabel("Thư mục video:  ")
        self.path_input = QLineEdit("")
        self.path_input.setReadOnly(True)
        path_button = QPushButton("chọn thư mục")
        path_button.clicked.connect(self.select_directory)
        path_layout.addWidget(path_label)
        path_layout.addWidget(self.path_input)
        path_layout.addWidget(path_button)
        main_layout.addLayout(path_layout)

        # Đường dẫn và nút chọn vị trí lưu
        path_layout_save = QHBoxLayout()
        path_label_save = QLabel("Vị trí lưu dữ liệu:")
        self.path_input_save = QLineEdit("")
        self.path_input_save.setReadOnly(True)
        path_button_save = QPushButton("chọn vị trí lưu")
        path_button_save.clicked.connect(self.select_save_directory)
        path_layout_save.addWidget(path_label_save)
        path_layout_save.addWidget(self.path_input_save)
        path_layout_save.addWidget(path_button_save)
        main_layout.addLayout(path_layout_save)

        # Danh sách tệp
        self.file_list = QListWidget()
        main_layout.addWidget(self.file_list)

        # Nút bấm
        button_layout = QHBoxLayout()
        self.button = QPushButton("Xử lý")
        self.button.clicked.connect(self.play_selected_videos)
        button_layout.addStretch()
        button_layout.addWidget(self.button)
        button_layout.addStretch()
        main_layout.addLayout(button_layout)

        # Media player
        self.selected_videos = []

    def select_directory(self):
        dialog = QFileDialog(self)
        dialog.setFileMode(QFileDialog.Directory)
        dialog.setOption(QFileDialog.DontUseNativeDialog, True)
        dialog.setViewMode(QFileDialog.Detail)
        dialog.setFilter(QDir.AllEntries | QDir.NoDotAndDotDot)
        dialog.setDirectory(".\\dic")

        if dialog.exec_():
            selected_dir = dialog.selectedFiles()[0]
            if selected_dir:
                self.path_input.setText(selected_dir)
                self.update_file_list(selected_dir)

    def select_save_directory(self):
        dialog = QFileDialog(self)
        dialog.setFileMode(QFileDialog.Directory)
        dialog.setOption(QFileDialog.DontUseNativeDialog, True)
        dialog.setViewMode(QFileDialog.Detail)
        dialog.setFilter(QDir.AllEntries | QDir.NoDotAndDotDot)
        dialog.setDirectory(".\\data")

        if dialog.exec_():
            selected_dir = dialog.selectedFiles()[0]
            if selected_dir:
                self.path_input_save.setText(selected_dir)

    def update_file_list(self, directory):
        self.file_list.clear()
        video_extensions = (".mp4", ".webm", ".avi", ".mkv", ".mov")

        try:
            for file_name in os.listdir(directory):
                if file_name.lower().endswith(video_extensions):
                    item_widget = FileItemWidget(file_name)
                    list_item = QListWidgetItem()
                    list_item.setSizeHint(item_widget.sizeHint())
                    self.file_list.addItem(list_item)
                    self.file_list.setItemWidget(list_item, item_widget)
        except Exception as e:
            self.file_list.addItem(f"Lỗi: {str(e)}")

    def play_selected_videos(self):
        self.selected_videos = []
        directory = self.path_input.text()

        for i in range(self.file_list.count()):
            item = self.file_list.item(i)
            widget = self.file_list.itemWidget(item)
            if widget and widget.checkbox.isChecked():
                full_path = os.path.join(directory, widget.file_name)
                self.selected_videos.append(full_path)

        if not self.selected_videos:
            return

        self.play_all_video()

    def play_all_video(self):
        for video_path in self.selected_videos:
            file_name = os.path.splitext(os.path.basename(video_path))[0]
            command = [
                "./.venv/Scripts/python.exe",
                "./capture_pose_data_video.py",
                "--pose_name", file_name,
                "--confidence", "0.2",
                "--file_path", video_path,
                "--data_list", self.path_input_save.text()
            ]
            process = subprocess.Popen(command)
            process.wait()

        QMessageBox.information(self, "Thông báo", "Dữ liệu đã được xử lý!")



class FileManager1(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        main_layout = QVBoxLayout(self)

        # Tiêu đề
        title_layout = QHBoxLayout()
        train_label = QHBoxLayout()
        train_label = QLabel("HUẤN LUYỆN MÔ HÌNH")
        train_label.setAlignment(Qt.AlignCenter)
        train_label.setStyleSheet("font-size: 30px; font-weight: bold;")
        title_layout.addWidget(train_label)
        main_layout.addLayout(title_layout)

        # Đường dẫn và nút chọn vị trí
        path_layout = QHBoxLayout()
        path_label = QLabel("Vị trí dữ liệu:")
        self.path_input = QLineEdit("")
        self.path_input.setReadOnly(True)
        path_button = QPushButton("chọn bộ huấn luyện")
        path_button.clicked.connect(self.select_directory)
        path_layout.addWidget(path_label)
        path_layout.addWidget(self.path_input)
        path_layout.addWidget(path_button)
        main_layout.addLayout(path_layout)

        # Đường dẫn và nút chọn vị trí lưu
        path_layout_save = QHBoxLayout()
        path_label_save = QLabel("Tên mô hình:")
        self.name_model = QLineEdit("")
        path_layout_save.addWidget(path_label_save)
        path_layout_save.addWidget(self.name_model)
        main_layout.addLayout(path_layout)
        main_layout.addLayout(path_layout_save)

        # Danh sách tệp
        self.file_list = QListWidget()
        main_layout.addWidget(self.file_list)

        # Nút bấm
        button_layout = QHBoxLayout()
        self.button = QPushButton("Huấn luyện")
        self.button.clicked.connect(self.create_model)
        button_layout.addStretch()
        button_layout.addWidget(self.button)
        button_layout.addStretch()
        main_layout.addLayout(button_layout)

        # Media player
        self.selected_videos = []

    def select_directory(self):
        dialog = QFileDialog(self)
        dialog.setFileMode(QFileDialog.Directory)
        dialog.setOption(QFileDialog.DontUseNativeDialog, True)
        dialog.setViewMode(QFileDialog.Detail)
        dialog.setFilter(QDir.AllEntries | QDir.NoDotAndDotDot)
        dialog.setDirectory(".\\data")

        if dialog.exec_():
            selected_dir = dialog.selectedFiles()[0]
            if selected_dir:
                self.path_input.setText(selected_dir)
                self.update_file_list(selected_dir)

    def select_save_directory(self):
        dialog = QFileDialog(self)
        dialog.setFileMode(QFileDialog.Directory)
        dialog.setOption(QFileDialog.DontUseNativeDialog, True)
        dialog.setViewMode(QFileDialog.Detail)
        dialog.setFilter(QDir.AllEntries | QDir.NoDotAndDotDot)
        dialog.setDirectory(".\\data")

        if dialog.exec_():
            selected_dir = dialog.selectedFiles()[0]
            if selected_dir:
                self.path_input_save.setText(selected_dir)

    def update_file_list(self, directory):
        self.file_list.clear()
        video_extensions = (".npy")

        try:
            for file_name in os.listdir(directory):
                if file_name.lower().endswith(video_extensions):
                    item_widget = FileItemWidget(file_name)
                    list_item = QListWidgetItem()
                    list_item.setSizeHint(item_widget.sizeHint())
                    self.file_list.addItem(list_item)
                    self.file_list.setItemWidget(list_item, item_widget)
        except Exception as e:
            self.file_list.addItem(f"Lỗi: {str(e)}")

    def create_model(self):
        try:
            if self.name_model.text() == "":
                QMessageBox.warning(self, "Thông báo", "Hãy nhập tên mô hình")
                return
            self.selected_videos = []
            directory = self.path_input.text()

            for i in range(self.file_list.count()):
                item = self.file_list.item(i)
                widget = self.file_list.itemWidget(item)
                if widget and widget.checkbox.isChecked():
                    full_path = os.path.join(directory, widget.file_name)
                    self.selected_videos.append(full_path)

            if not self.selected_videos:
                return

            # Lưu danh sách file vào file tạm thời
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as temp_file:
                temp_file.write('\n'.join(self.selected_videos))
                temp_file_path = temp_file.name

            # Tạo lệnh với đường dẫn đến file tạm thời
            command = [
                "./.venv/Scripts/python.exe",
                "./scripts/trainCNN3D.py",
                "--model_name", self.name_model.text(),
                "--data_selected", temp_file_path  # Truyền đường dẫn file tạm thời
            ]
            process = subprocess.Popen(command)
            process.wait()

            # Xóa file tạm thời sau khi sử dụng
            os.remove(temp_file_path)

            # Trích xuất tên file từ self.selected_videos và tạo dictionary Mapping
            Mapping = {}
            Mapping["default"] = "..."
            for video_path in self.selected_videos:
                file_name = os.path.basename(video_path).split('.')[0]
                Mapping[file_name] = file_name

            # Tạo DataFrame từ dictionary
            df = pd.DataFrame(list(Mapping.items()), columns=["Key", "Value"])

            # Ghi vào file Excel
            output_file = os.path.join("./models/", self.name_model.text() + ".xlsx")
            df.to_excel(output_file, index=False, engine='openpyxl')

            QMessageBox.information(self, "Thông báo", f"Tạo mô hình thành công")
        except Exception as e:
            print(e)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Thu thập dữ liệu")
        self.setGeometry(600, 100, 600, 700)

        # Tạo widget chính và layout chính
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Tạo toolbar
        toolbar = QToolBar("Main Toolbar")
        self.addToolBar(toolbar)

        # Tạo action cho FileManager
        collect_action = QAction("Thu Thập Dữ Liệu", self)
        collect_action.triggered.connect(lambda: self.stack.setCurrentIndex(0))
        toolbar.addAction(collect_action)

        # Tạo action cho FileManager1
        train_action = QAction("Huấn Luyện Mô Hình", self)
        train_action.triggered.connect(lambda: self.stack.setCurrentIndex(1))
        toolbar.addAction(train_action)

        # Tạo QStackedWidget để chứa FileManager và FileManager1
        self.stack = QStackedWidget()
        self.file_manager = FileManager()
        self.file_manager1 = FileManager1()
        self.stack.addWidget(self.file_manager)
        self.stack.addWidget(self.file_manager1)
        main_layout.addWidget(self.stack)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())