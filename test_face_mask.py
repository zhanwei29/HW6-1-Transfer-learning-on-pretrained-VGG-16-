from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import requests
from io import BytesIO

def test_image_url(model):
    """
    測試用戶輸入的圖片 URL，並使用模型進行預測
    """
    try:
        # 用戶輸入圖片 URL
        image_url = input("請輸入圖片 URL: ")

        # 驗證 URL 是否有效
        if not image_url.startswith("http"):
            print("請輸入有效的 URL！")
            return

        # 從 URL 加載圖片
        response = requests.get(image_url)
        img = Image.open(BytesIO(response.content)).resize((224, 224))
        
        # 預處理圖片
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # 模型預測
        prediction = model.predict(img_array)
        class_label = "With Mask" if prediction[0][0] < 0.5 else "Without Mask"
        print(f"Prediction for the image: {class_label}")

    except Exception as e:
        print(f"Error processing image: {e}")

# 加載模型
model = load_model("./face_mask_detector_vgg16.h5")
print("Model loaded successfully!")

# 啟動測試
while True:
    test_image_url(model)
    # 詢問用戶是否繼續
    cont = input("是否繼續測試其他圖片？(y/n): ").lower()
    if cont != 'y':
        print("測試結束，感謝使用！")
        break
