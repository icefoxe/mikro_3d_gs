Python 3.10.11


### camera.py
przechwuje parametry kamery, przekształca punkty 3D do układu kamery, rzutuje punkty 3D na piksele obrazu


### rendered.py

Funkcja która dostaje kamerę, środki gaussów, kolory, opacity, rozmycia splatów.
Zwraca  obraz RGB


### colmap_loader.py

Tu sie fajnie robi

**Wczytuje:**
-intricies z camera.txt
-kamery/obrazy z images.txt
-punkty 3D z points3D.txt


**ColmapCameraModel** trzyma jedną kamerę z cameras.txt:
model kamery, rozdzielczość, parametry


**ColmapImage** trzyma jeden wpis z images.txt:
quaternion, translację, camera_id, nazwe zdjęcia


**ColmapPoint3D**

Trzyma jeden punkt z points3D.txt:
pozycję 3D, kolor RGB, błąd

[Kamery](https://colmap.github.io/faq.html)

> SIMPLE_RADIAL (default): A good starting point for most standard cameras. Models a single focal length, principal point, and one radial distortion parameter

> PINHOLE: Use if your images have negligible lens distortion (e.g., already undistorted images or high-quality industrial lenses).

> OPENCV: A good choice for wider-angle lenses with moderate distortion. Models 2 focal lengths, principal point, and 4 distortion parameters (2 radial + 2 tangential).

> SIMPLE_RADIAL_FISHEYE or OPENCV_FISHEYE: Use for fisheye lenses with a field of view significantly larger than 120 degrees.

> FULL_OPENCV: Use only when you have many images sharing intrinsics and need to model complex distortion patterns. With 12 parameters, this model requires a large number of observations to converge reliably.

> As a rule of thumb, use the simplest model that adequately describes your lens. Overly complex models with many parameters can lead to degenerate or overfitted calibration, especially when few images share intrinsics. If in doubt, start with SIMPLE_RADIAL and inspect the reprojection errors in the model statistics.

[Camera Models](https://colmap.github.io/cameras.html)