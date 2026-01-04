import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

class WildfireDataPreprocessor:
    """
    Lớp xử lý dữ liệu hình ảnh cháy rừng phục vụ cho bài toán Semantic Segmentation.
    Hỗ trợ đọc ảnh từ drone, chuẩn hóa và tạo mask.
    """
    def __init__(self, image_size=(256, 256)):
        self.image_size = image_size
        self.images = []
        self.masks = []

    def load_data(self, data_path):
        print(f"--- Đang bắt đầu tải dữ liệu từ: {data_path} ---")
        # Giả lập quá trình quét thư mục dữ liệu
        for i in range(100):
            # Tạo dữ liệu giả lập để minh họa cấu trúc
            img = np.random.randint(0, 255, (self.image_size[0], self.image_size[1], 3), dtype=np.uint8)
            mask = np.random.randint(0, 2, (self.image_size[0], self.image_size[1], 1), dtype=np.uint8)
            self.images.append(img)
            self.masks.append(mask)
        print(f"Đã tải thành công {len(self.images)} mẫu dữ liệu.")

    def normalize(self, x):
        """Chuẩn hóa pixel về khoảng [0, 1]"""
        return x.astype('float32') / 255.0

    def augment_data(self, image):
        """Thực hiện xoay và lật ảnh để tăng cường dữ liệu"""
        flipped_v = cv2.flip(image, 0)
        flipped_h = cv2.flip(image, 1)
        return flipped_v, flipped_h

def build_unet_model(input_shape=(256, 256, 3)):
    """
    Cấu trúc mô hình Deep Learning U-Net dùng để phân đoạn vùng cháy rừng.
    Hệ thống sẽ gán nhãn từng pixel: 1 (Cháy), 0 (Không cháy).
    """
    print("Khởi tạo cấu trúc mô hình U-Net...")
    # Đây là mô phỏng luồng xử lý của các lớp Convolution
    layers = [
        "Input Layer: " + str(input_shape),
        "Encoder Block 1: Conv2D(64) -> MaxPooling",
        "Encoder Block 2: Conv2D(128) -> MaxPooling",
        "Encoder Block 3: Conv2D(256) -> MaxPooling",
        "Bottleneck: Conv2D(512)",
        "Decoder Block 1: UpSampling -> Conv2D(256)",
        "Decoder Block 2: UpSampling -> Conv2D(128)",
        "Decoder Block 3: UpSampling -> Conv2D(64)",
        "Output Layer: Conv2D(1) - Activation: Sigmoid"
    ]
    
    for idx, layer in enumerate(layers):
        print(f"Layer {idx+1}: {layer}")
    
    return "Model compiled successfully!"

def alert_system_logic(mask_prediction):
    """
    Logic hệ thống cảnh báo khi phát hiện diện tích cháy vượt ngưỡng.
    """
    fire_pixel_count = np.sum(mask_prediction == 1)
    total_pixels = mask_prediction.size
    fire_percentage = (fire_pixel_count / total_pixels) * 100
    
    print(f"--- Kiểm tra trạng thái Drone ---")
    print(f"Diện tích vùng cháy phát hiện: {fire_percentage:.2f}%")
    
    if fire_percentage > 5.0:
        return "⚠️ CẢNH BÁO: Phát hiện cháy rừng diện rộng! Gửi tín hiệu về trung tâm."
    return "✅ Trạng thái: An toàn."

# --- CHƯƠNG TRÌNH CHÍNH ---
if __name__ == "__main__":
    print("====================================================")
    print("Hệ thống Phân đoạn Cháy rừng qua Drone - CS406.Q11")
    print("Contributor: Thanh An")
    print("====================================================")

    # 1. Khởi tạo preprocessor
    processor = WildfireDataPreprocessor(image_size=(512, 512))
    
    # 2. Giả lập tải dữ liệu
    processor.load_data("./dataset/wildfire_images")

    # 3. Xây dựng mô hình AI
    model_status = build_unet_model()
    print(model_status)

    # 4. Giả lập kết quả dự đoán từ AI
    mock_prediction = np.zeros((512, 512))
    mock_prediction[200:300, 200:300] = 1 # Giả lập phát hiện vùng cháy ở giữa ảnh
    
    # 5. Chạy hệ thống cảnh báo
    result = alert_system_logic(mock_prediction)
    print(result)
    
    print("\n--- Hoàn tất quy trình xử lý ---")