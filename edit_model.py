import sys
import pandas as pd
import os
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QTableWidget, QTableWidgetItem, QPushButton, QMessageBox
from PyQt5.QtCore import Qt

class ExcelEditorDialog(QDialog):
    def __init__(self, excel_file_path, parent=None):
        super().__init__(parent)
        self.excel_file_path = excel_file_path
        self.setWindowTitle("SỬA BỘ DỮ LIỆU")
        self.setGeometry(300, 300, 600, 700)

        # Tạo layout chính
        layout = QVBoxLayout()

        # Tạo bảng để hiển thị dữ liệu
        self.table = QTableWidget()
        layout.addWidget(self.table)

        # Tạo nút Lưu
        self.save_button = QPushButton("Lưu thay đổi")
        self.save_button.clicked.connect(self.save_changes)
        layout.addWidget(self.save_button)

        self.setLayout(layout)

        # Load dữ liệu từ file Excel
        self.load_excel_data()

    def load_excel_data(self):
        # Đọc file Excel
        try:
            df = pd.read_excel(self.excel_file_path, engine='openpyxl')
            if 'Key' not in df.columns or 'Value' not in df.columns:
                QMessageBox.warning(self, "Thông báo", "File Excel không đúng định dạng! Cần có cột 'Key' và 'Value'.")
                self.close()
                return

            # Thiết lập bảng
            self.table.setRowCount(len(df))
            self.table.setColumnCount(2)
            self.table.setHorizontalHeaderLabels(['Key', 'Value'])
            self.table.setColumnWidth(0, 200)
            self.table.setColumnWidth(1, 350)
            # Điền dữ liệu vào bảng
            for row in range(len(df)):
                # Cột File Name (không cho chỉnh sửa)
                file_name_item = QTableWidgetItem(str(df.iloc[row]['Key']))
                file_name_item.setFlags(file_name_item.flags() & ~Qt.ItemIsEditable)  # Vô hiệu hóa chỉnh sửa
                self.table.setItem(row, 0, file_name_item)

                # Cột Value (cho phép chỉnh sửa)
                value_item = QTableWidgetItem(str(df.iloc[row]['Value']))
                self.table.setItem(row, 1, value_item)

        except Exception as e:
            QMessageBox.warning(self, "Thông báo", f"Lỗi khi đọc file Excel: {str(e)}")
            self.close()

    def save_changes(self):
        # Lấy dữ liệu từ bảng
        data = []
        for row in range(self.table.rowCount()):
            file_name = self.table.item(row, 0).text()
            value = self.table.item(row, 1).text()
            data.append({'Key': file_name, 'Value': value})

        # Tạo DataFrame mới
        df = pd.DataFrame(data)

        # Lưu lại vào file Excel
        try:
            df.to_excel(self.excel_file_path, index=False, engine='openpyxl')
            QMessageBox.information(self, "Thông báo", "Lưu thay đổi thành công!")
            self.accept()  # Đóng hộp thoại
        except Exception as e:
            QMessageBox.warning(self, "Thông báo", f"Lỗi khi lưu file Excel: {str(e)}")

if __name__ == "__main__":
    from PyQt5.QtWidgets import QApplication
    app = QApplication(sys.argv)
    dialog = ExcelEditorDialog("./models/ex1.xlsx")
    dialog.exec_()
    sys.exit(app.exec_())