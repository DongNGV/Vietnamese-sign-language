import pickle
import numpy as np

class ASLClassificationModel:
    @staticmethod
    def load_model(model_path):
        # Load model and mapping from pickle
        with open(model_path, "rb") as file:
            model, mapping = pickle.load(file)

        if model is not None:
            return ASLClassificationModel(model, mapping)

        raise Exception("Model not loaded correctly!")

    def __init__(self, model, mapping):
        self.model = model
        self.mapping = mapping

    def predict(self, feature):
        return self.mapping[self.model.predict(feature.reshape(1, -1)).item()]

    def predict_with_probability(self, feature):
        # Kiểm tra xem mô hình có phương thức predict_proba không
        if hasattr(self.model, 'predict_proba'):
            # Lấy mảng xác suất cho tất cả các lớp
            probabilities = self.model.predict_proba(feature.reshape(1, -1))[0]
            # Lấy dự đoán (chỉ số của lớp có xác suất cao nhất)
            predicted_class_index = np.argmax(probabilities)
            # Lấy xác suất cao nhất
            confidence = probabilities[predicted_class_index]
            # Chuyển đổi chỉ số thành nhãn thông qua mapping
            predicted_label = self.mapping[predicted_class_index]

            return {
                'prediction': predicted_label,
                'confidence': float(confidence)
            }
        else:
            # Nếu mô hình không hỗ trợ predict_proba
            prediction = self.predict(feature)
            return {
                'prediction': prediction,
                'confidence': None
            }

