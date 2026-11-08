import os
import shutil
from sklearn.model_selection import train_test_split

# ✅ Đường dẫn gốc chứa dữ liệu sau khi giải nén
base_dir = r"C:\Users\ACER\Desktop\NhanDienVoTraiCay\ThungSongXanh\fer2013"

# ✅ Thư mục đích sau khi chia
output_dir = os.path.join(base_dir, "fer2013_split")
os.makedirs(output_dir, exist_ok=True)

# ✅ Các nhãn cảm xúc (tùy bộ dữ liệu mà có thể khác)
emotions = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

# ✅ Tạo cấu trúc thư mục train/val/test
for split in ["train", "validation", "test"]:
    for emotion in emotions:
        os.makedirs(os.path.join(output_dir, split, emotion), exist_ok=True)

# ✅ Hàm chia dữ liệu
def split_data(emotion_dir, emotion_name):
    images = os.listdir(emotion_dir)
    train, test = train_test_split(images, test_size=0.2, random_state=42)
    val, test = train_test_split(test, test_size=0.5, random_state=42)

    for img in train:
        shutil.copy(os.path.join(emotion_dir, img),
                    os.path.join(output_dir, "train", emotion_name, img))
    for img in val:
        shutil.copy(os.path.join(emotion_dir, img),
                    os.path.join(output_dir, "validation", emotion_name, img))
    for img in test:
        shutil.copy(os.path.join(emotion_dir, img),
                    os.path.join(output_dir, "test", emotion_name, img))

# ✅ Thực hiện chia cho từng cảm xúc
for emotion in emotions:
    emotion_dir = os.path.join(base_dir, emotion)
    if os.path.exists(emotion_dir):
        print(f"➡️ Đang xử lý {emotion} ...")
        split_data(emotion_dir, emotion)
    else:
        print(f"⚠️ Không tìm thấy thư mục {emotion} trong {base_dir}")

print("✅ Hoàn tất chia dữ liệu!")
