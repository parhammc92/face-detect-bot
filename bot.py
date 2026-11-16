"""my bot — تشخیص چهره با OpenCV و مدل Haar Cascade.

مدل (model.xml) را لود می‌کند، چهره‌ها را در عکس (download.webp)
پیدا می‌کند، دورشان مستطیل سبز می‌کشد و نتیجه را در
result.jpg ذخیره می‌کند.
"""

import os
import sys

import cv2

# فایل‌ها همیشه کنار همین اسکریپت هستند (فرقی نمی‌کند از کجا اجرا شود)
BASE = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE, "model.xml")
IMAGE_PATH = os.path.join(BASE, "download.webp")
RESULT_PATH = os.path.join(BASE, "result.jpg")

# ── ۱. لود کردن مدل ───────────────────────────────────────────

face_cascade = cv2.CascadeClassifier(MODEL_PATH)

if face_cascade.empty():
    print("خطا: مدل بارگذاری نشد — model.xml را چک کن.")
    sys.exit(1)

print("مدل با موفقیت لود شد ✅")

# ── ۲. خواندن عکس ─────────────────────────────────────────────

image = cv2.imread(IMAGE_PATH)

if image is None:
    print("خطا: عکس خوانده نشد — download.webp را چک کن.")
    sys.exit(1)

print(f"عکس خوانده شد: {image.shape[1]}×{image.shape[0]} پیکسل")

# ── ۳. تبدیل به سیاه‌وسفید (مدل با خاکستری کار می‌کند) ────────

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# ── ۴. پیدا کردن چهره‌ها ──────────────────────────────────────

faces = face_cascade.detectMultiScale(
    gray,
    scaleFactor=1.1,     # هر مرحله ۱۰٪ کوچک‌تر
    minNeighbors=5,      # هرچه بیشتر، خطای کمتر
    minSize=(60, 60)     # چهره‌های خیلی کوچک نادیده گرفته می‌شوند
)

print(f"چهره پیدا شد: {len(faces)} تا 🎯")

# ── ۵. کشیدن مستطیل دور هر چهره ──────────────────────────────

for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(
        image, "Face", (x, y - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
    )

# ── ۶. ذخیره‌ی نتیجه ─────────────────────────────────────────

cv2.imwrite(RESULT_PATH, image)
print(f"نتیجه ذخیره شد: {RESULT_PATH}")
