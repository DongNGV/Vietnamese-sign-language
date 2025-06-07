# Số điểm bàn tay
FEATURES_PER_HAND = 21

# Tên model
MODEL_NAME = "QIPEDC"

# chỉ số nhận dạng tư thế
MODEL_CONFIDENCE = 0.4

# bỏ qua vài khung hình để tăng tốc độ xử lý / không bỏ qua = 0
BO_QUA_KHUNG_HINH = 0

# với số lượng mô hình ít có thể tăng giá trị để bỏ qua các tư thế có độ chính xác thấp
GIA_TRI_NHAN_TU_THE = 0.5

# trong 10 lần đánh giá nếu có 1 đánh giá xuất hiện với số lần bằng số GIA_TRI_NHAN_DANG thì nhận từ đó
GIA_TRI_NHAN_DANG = 7

TIMESTEPS = 15
