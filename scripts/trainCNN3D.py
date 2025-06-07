import os
import time
import datetime
import numpy as np
import argparse
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import tensorflow as tf
from tensorflow.keras import layers, models
import ast

# Định nghĩa mô hình CNN 3D
def create_cnn3d_model(input_shape, num_classes):
    """
    Tạo mô hình CNN 3D để nhận dạng cử chỉ thời gian thực
    :param input_shape: (timesteps, features, height, width, channels)
    :param num_classes: Số lượng lớp cử chỉ
    """
    model = models.Sequential([
        layers.Conv3D(64, kernel_size=(3, 3, 1), activation='relu', input_shape=input_shape, padding='same'),
        layers.MaxPooling3D(pool_size=(2, 2, 1)),
        layers.Conv3D(128, kernel_size=(3, 3, 1), activation='relu', padding='same'),
        layers.MaxPooling3D(pool_size=(2, 2, 1)),
        layers.Conv3D(256, kernel_size=(3, 3, 1), activation='relu', padding='same'),
        layers.MaxPooling3D(pool_size=(2, 2, 1)),
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

if __name__ == "__main__":
    timeStart = datetime.datetime.now()
    parser = argparse.ArgumentParser("Huấn luyện mô hình CNN 3D")
    parser.add_argument("--model_name", help="Tên của mô hình", type=str, default="cnn3d_model")
    parser.add_argument("--dir", help="Vị trí của mô hình", type=str, default="models")
    parser.add_argument("--epochs", help="Số epoch huấn luyện", type=int, default=100)
    parser.add_argument("--batch_size", help="Kích thước batch", type=int, default=128)
    parser.add_argument("--data_selected", help="Đường dẫn đến file chứa danh sách dữ liệu", type=str, default="")
    args = parser.parse_args()

    # Đọc danh sách file từ file tạm thời
    try:
        if args.data_selected:
            with open(args.data_selected, 'r') as f:
                data_selected = [line.strip() for line in f.readlines() if line.strip()]
            if not data_selected:
                raise ValueError("File chứa danh sách dữ liệu rỗng.")
        else:
            data_selected = []
    except Exception as e:
        raise ValueError(f"Không thể đọc file danh sách dữ liệu: {args.data_selected}. Lỗi: {e}")

    # Xử lý dữ liệu và tạo mapping
    X, y, mapping = [], [], {}
    FileLoi = []
    valid_classes = set()  # Lưu các lớp hợp lệ
    for current_class_index, pose_file in enumerate(data_selected):
        if not pose_file.endswith(".npy"):
            continue
        file_path = pose_file
        try:
            pose_data = np.load(file_path)
            if pose_data.ndim != 3:
                print(f"Bỏ qua {file_path}: Kỳ vọng mảng 3D, nhận shape {pose_data.shape}")
                continue
            X.append(pose_data)
            y.extend([current_class_index] * pose_data.shape[0])
            mapping[current_class_index] = os.path.basename(pose_file).split(".")[0]
            valid_classes.add(current_class_index)  # Thêm lớp hợp lệ
        except Exception as e:
            print(f"Lỗi khi tải {file_path}: {e}")
            FileLoi.append(pose_file)

    if not X:
        raise ValueError("Không có dữ liệu hợp lệ nào được tải.")

    X = np.vstack(X)
    y = np.array(y)

    # Định dạng lại dữ liệu
    X = X[..., np.newaxis, np.newaxis]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Tạo và huấn luyện mô hình
    input_shape = X.shape[1:]
    num_classes = len(mapping)  # Số lớp dựa trên mapping
    model = create_cnn3d_model(input_shape, num_classes)
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    history = model.fit(
        X_train, y_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_data=(X_test, y_test),
        callbacks=[early_stopping]
    )
    timeEnd = datetime.datetime.now()

    # Đánh giá và lưu kết quả
    os.makedirs(args.dir, exist_ok=True)
    model_path = os.path.join(args.dir, f"{args.model_name}.keras")

    # In classification report
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)

    # Lấy các lớp thực tế trong y_test và đảm bảo khớp với mapping
    unique_classes = np.unique(y_test)
    # Lọc các lớp trong unique_classes có trong valid_classes
    valid_unique_classes = [cls for cls in unique_classes if cls in valid_classes]
    if not valid_unique_classes:
        raise ValueError("Không có lớp hợp lệ nào trong y_test khớp với mapping.")

    # Loại bỏ các mẫu trong X_test và y_test có nhãn không hợp lệ
    valid_mask = np.isin(y_test, valid_unique_classes)  # Tạo mask cho các mẫu hợp lệ
    X_test_filtered = X_test[valid_mask]  # Lọc X_test
    y_test_filtered = y_test[valid_mask]  # Lọc y_test
    y_pred_filtered = y_pred[valid_mask]  # Lọc y_pred tương ứng
    y_pred_classes_filtered = np.argmax(y_pred_filtered, axis=1)  # Tính lại lớp dự đoán

    # In classification report (không sử dụng target_names)
    print("Classification Report:")
    print(classification_report(y_test_filtered, y_pred_classes_filtered))

    model.save(model_path)
    print(f"Đã lưu mô hình vào {model_path}")
    with open(os.path.join(args.dir, f"{args.model_name}_mapping.pkl"), "wb") as f:
        pickle.dump(mapping, f)
    train_loss, train_accuracy = model.evaluate(X_train, y_train, verbose=0)
    test_loss, test_accuracy = model.evaluate(X_test_filtered, y_test_filtered, verbose=0)
    print(
        f"Độ chính xác huấn luyện: {round(train_accuracy * 100, 2)}% - Độ chính xác kiểm tra: {round(test_accuracy * 100, 2)}%")
    with open(os.path.join(args.dir, f"{args.model_name}_results.txt"), "w", encoding="utf-8") as f:
        f.write(f"Số mẫu huấn luyện: {X.shape[0]}\n")
        f.write(f"Số lớp: {num_classes}\n")
        f.write(f"Độ chính xác huấn luyện: {round(train_accuracy * 100, 2)}%\n")
        f.write(f"Độ chính xác kiểm tra: {round(test_accuracy * 100, 2)}%\n")
        f.write(f"Thời gian bắt đầu: {timeStart}\n")
        f.write(f"Thời gian kết thúc: {timeEnd}\n")
        f.write(f"Các file không thể train: {FileLoi}\n")