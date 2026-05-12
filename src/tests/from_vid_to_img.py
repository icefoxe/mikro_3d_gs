import cv2
import os
import shutil
import numpy as np
import matplotlib.pyplot as plt

video_path = "jerry_lego.mp4"
output_frames = "frames"

os.makedirs(output_frames, exist_ok=True)

cap = cv2.VideoCapture(video_path)

frame_id = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    filename = f"{output_frames}/frame_{frame_id:05d}.jpg"
    cv2.imwrite(filename, frame)

    frame_id += 1

cap.release()
print("Gotowe!")

#zapis do folderu
output_final = "frames_final"

#czyszenie folderu
if (output_final==True):
    for file in os.listdir(output_final):
        file_path = os.path.join(output_final, file)
        if os.path.isfile(file_path):
            os.remove(file_path)
#tworzenie dolferu
else:
    os.makedirs(output_final, exist_ok=True)

#def is_blurry(image, threshold=100):
    #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #score = cv2.Laplacian(gray, cv2.CV_64F).var()
    #return score < threshold, score

image_files = sorted([
    f for f in os.listdir(output_frames)
    if f.lower().endswith(('.png', '.jpg', '.jpeg'))
])

blur_list = []
brightness_list = []
contrast_list = []
valid_data = []

for file in image_files:
    #ściezka do pliku
    path = os.path.join(output_frames, file)
    #wczytanie
    img = cv2.imread(path)

    if img is None:
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.Laplacian(gray, cv2.CV_64F).var()
    brightness = np.mean(img)
    contrast = img.std()

    blur_list.append(blur)
    brightness_list.append(brightness)
    contrast_list.append(contrast)

    valid_data.append((file, img, blur, brightness, contrast))

k = 2  #3

blur_mean = np.mean(blur_list)
blur_std = np.std(blur_list)

bright_mean = np.mean(brightness_list)
bright_std = np.std(brightness_list)

contrast_mean = np.mean(contrast_list)
contrast_std = np.std(contrast_list)

print(f"BLUR:      min={blur_mean - k*blur_std:.2f} | max={blur_mean + k*blur_std:.2f}")
print(f"BRIGHT:    min={bright_mean - k*bright_std:.2f} | max={bright_mean + k*bright_std:.2f}")
print(f"CONTRAST:  min={contrast_mean - k*contrast_std:.2f} | max={contrast_mean + k*contrast_std:.2f}")

saved_count = 0

for file, img, blur, brightness, contrast in valid_data:

    print(f"{file} | blur={blur:.1f}, bright={brightness:.1f}, contrast={contrast:.1f}")

    # 🔹 BLUR (TYLKO DOLNA GRANICA!)
    if blur < blur_mean - k * blur_std:
        print("ODRZUCONO (blur):", file)
        continue

    # 🔹 JASNOŚĆ (2 strony)
    if not (bright_mean - k * bright_std <= brightness <= bright_mean + k * bright_std):
        print("ODRZUCONO (jasność):", file)
        continue

    # 🔹 KONTRAST (2 strony)
    if not (contrast_mean - k * contrast_std <= contrast <= contrast_mean + k * contrast_std):
        print("ODRZUCONO (kontrast):", file)
        continue

    print("ZAPISANO:", file)
    shutil.copy(os.path.join(output_frames, file),
                os.path.join(output_final, file))

    saved_count += 1
print(f"Zapisano {saved_count} zdjęć")

plt.figure()
plt.hist(blur_list, bins=30)
plt.axvline(blur_mean - k*blur_std, color='r')
plt.axvline(blur_mean + k*blur_std, color='r')
plt.title("Blur distribution (2σ)")
plt.show()