import os
import cv2
import numpy as np
from tqdm.notebook import tqdm 
import json

class LaserReflexData:
    def __init__(self, path):
        self.path = path
        self.video = LaserFlexInputVideo(os.path.join(path, "input.mp4"));
        self.wrapOptions = LaserFlexWrapOptions(os.path.join(path, "wrapOptions.json"))
        self.processOptions = LaserFlexProcessOptions(os.path.join(path, "processOptions.json"))
        self.lineData = LaserFlexLineData(os.path.join(path, "linescanData.npz"))
        self.lineToImageOptions = LaserFlexLineToImageOptions(os.path.join(path, "lineToImageOptions.json"))
        self.output = LaserFlexOutput(os.path.join(path, "output.npz"))
        
    def load(self):
        objects = [
            self.video,
            self.wrapOptions,
            self.processOptions,
            self.lineData,
            self.lineToImageOptions,
            self.output
        ]

        for obj in objects:
            try:
                obj.load()
            except Exception as e:
                print(f"Fehler beim Laden von {obj.file_path}: {e}")

        return self
        
    def save(self):
        objects = [
            self.video,
            self.wrapOptions,
            self.processOptions,
            self.lineData,
            self.lineToImageOptions,
            self.output
        ]

        for obj in objects:
            try:
                obj.save()
            except Exception as e:
                print(f"Fehler beim Speichern von {obj.file_path}: {e}")

        return self
        
        
    def process_video(self):
        self.processOptions.process_video(self.video, self.wrapOptions, self.lineData)
        
    def line_to_image(self):
        self.lineToImageOptions.process(self.lineData, self.output)

# -------------------------- VIDEO --------------------------
        

class LaserFlexInputVideo:
    def __init__(self, video_path):
        self.video_path = video_path
        self.cap = None

        self.frame_count = 0
        self.fps = 0.0
        self.width = 0
        self.height = 0

    def load(self):
        """Öffnet das Video und prüft, ob es lesbar ist."""

        self.close()

        print("OpenCV Version:", cv2.__version__)
        print("Dateipfad:", os.path.abspath(self.video_path))
        print("Datei existiert:", os.path.isfile(self.video_path))

        if not os.path.isfile(self.video_path):
            raise FileNotFoundError(self.video_path)

        self.cap = cv2.VideoCapture(self.video_path)

        if not self.cap.isOpened():
            self.close()

            print("\nFEHLER: Video konnte nicht geöffnet werden.")
            print("OpenCV Build-Information:")
            print(cv2.getBuildInformation())

            raise IOError("Video konnte nicht geöffnet werden.")

        self.frame_count = int(
            self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
        )
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(
            self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        )
        self.height = int(
            self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        )

        print("Video erfolgreich geöffnet!")
        print("Backend:", self.cap.getBackendName())
        print("FPS:", self.fps)
        print("Frames:", self.frame_count)
        print("Auflösung:", self.width, "x", self.height)

        return self

    def close(self):
        """Schließt das Video und gibt Ressourcen frei."""

        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def getFrameCount(self):
        """Gibt die Anzahl der Frames zurück."""

        self._check_loaded()
        return self.frame_count

    def _check_loaded(self):
        if self.cap is None or not self.cap.isOpened():
            raise RuntimeError(
                "Video ist nicht geladen. Zuerst load() aufrufen."
            )

    def getFrame(self, n):
        """Liest Frame n und gibt ein RGB-NumPy-Array zurück."""

        self._check_loaded()

        if not isinstance(n, (int, np.integer)):
            raise TypeError("Frame-Index muss eine Ganzzahl sein.")

        if n < 0 or n >= self.frame_count:
            raise IndexError(
                f"Frame-Index {n} außerhalb des Bereichs "
                f"0 bis {self.frame_count - 1}."
            )

        # Zu Frame n springen
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, n)

        ret, frame = self.cap.read()

        if not ret:
            raise IOError(f"Frame {n} konnte nicht gelesen werden.")

        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def foreachFrame(self, callback):
        """
        Iteriert über alle Frames.

        callback(rgb_array, frame_index)
        """

        self._check_loaded()

        # Am Anfang beginnen
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        for frame_idx in tqdm(
            range(self.frame_count),
            desc="Verarbeite Frames"
        ):
            ret, frame = self.cap.read()

            if not ret:
                raise IOError(
                    f"Frame {frame_idx} konnte nicht gelesen werden."
                )

            rgb_array = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            callback(rgb_array, frame_idx)

    def __enter__(self):
        return self.load()

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        
        
# -------------------------- WRAP/RECTIFY --------------------------

class LaserFlexWrapOptions:

    def __init__(self, file_path):
        self.file_path = file_path

        self.corner00 = [0, 0]
        self.corner01 = [100, 0]
        self.corner10 = [0, 100]
        self.corner11 = [100, 100]

        self.outputWidth = 400
        self.outputHeight = 400

    def save(self):
        """Speichert die Einstellungen als JSON."""

        data = {
            "corner00": self.corner00,
            "corner01": self.corner01,
            "corner10": self.corner10,
            "corner11": self.corner11,
            "outputWidth": self.outputWidth,
            "outputHeight": self.outputHeight
        }

        # Verzeichnis bei Bedarf erstellen
        directory = os.path.dirname(
            os.path.abspath(self.file_path)
        )
        os.makedirs(directory, exist_ok=True)

        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)

    def load(self):
        """Lädt die Einstellungen aus einer JSON-Datei."""

        if not os.path.isfile(self.file_path):
            raise FileNotFoundError(
                f"Optionsdatei nicht gefunden: {self.file_path}"
            )

        with open(self.file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.corner00 = data["corner00"]
        self.corner01 = data["corner01"]
        self.corner10 = data["corner10"]
        self.corner11 = data["corner11"]

        self.outputWidth = int(data["outputWidth"])
        self.outputHeight = int(data["outputHeight"])

        return self

    def rectify_image(self, u):

        # Quellkoordinaten der vier Ecken
        src = np.array([
            self.corner00,  # oben links
            self.corner01,  # oben rechts
            self.corner10,  # unten links
            self.corner11   # unten rechts
        ], dtype=np.float32)

        # Zielkoordinaten
        dst = np.array([
            [0, 0],
            [self.outputWidth - 1, 0],
            [0, self.outputHeight - 1],
            [self.outputWidth - 1, self.outputHeight - 1]
        ], dtype=np.float32)

        # Homographie berechnen
        H = cv2.getPerspectiveTransform(src, dst)

        # Perspektivische Entzerrung
        u_rectified = cv2.warpPerspective(
            u,
            H,
            (self.outputWidth, self.outputHeight),
            flags=cv2.INTER_LINEAR
        )

        return u_rectified

# -------------------------- PROCESS --------------------------



import os
import json
import numpy as np


class LaserFlexProcessOptions:

    def __init__(self, file_path):
        self.file_path = file_path

        self.angleResolutionX = 20
        self.angleResolutionY = 20

        # Verarbeitungsergebnisse
        self.frame_colors = None
        self.frame_features = None
        self.frame_angles = None

    def save(self):
        """Speichert die Einstellungen als JSON."""

        data = {
            "angleResolutionX": self.angleResolutionX,
            "angleResolutionY": self.angleResolutionY
        }

        directory = os.path.dirname(
            os.path.abspath(self.file_path)
        )
        os.makedirs(directory, exist_ok=True)

        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)

    def load(self):
        """Lädt die Einstellungen aus einer JSON-Datei."""

        if not os.path.isfile(self.file_path):
            raise FileNotFoundError(
                f"Optionsdatei nicht gefunden: {self.file_path}"
            )

        with open(self.file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.angleResolutionX = int(data["angleResolutionX"])
        self.angleResolutionY = int(data["angleResolutionY"])

        return self


    def process_video(
        self,
        laserFlexInputVideo,
        laserFlexWrapOptions,
        output_laserFlexLineData
    ):
        from tqdm.notebook import tqdm

        frame_count = laserFlexInputVideo.getFrameCount()

        resolution_x = int(self.angleResolutionX)
        resolution_y = int(self.angleResolutionY)

        if resolution_x <= 0 or resolution_y <= 0:
            raise ValueError("Die Kachelauflösung muss positiv sein.")

        output_width = int(laserFlexWrapOptions.outputWidth)
        output_height = int(laserFlexWrapOptions.outputHeight)

        if output_width % resolution_x != 0:
            raise ValueError(
                "outputWidth muss durch angleResolutionX teilbar sein."
            )

        if output_height % resolution_y != 0:
            raise ValueError(
                "outputHeight muss durch angleResolutionY teilbar sein."
            )

        tile_width = output_width // resolution_x
        tile_height = output_height // resolution_y

        # LineScanData initialisieren
        line_scan = output_laserFlexLineData

        line_scan.frame_count = frame_count
        line_scan.fps = laserFlexInputVideo.fps

        line_scan.angleResolutionX = resolution_x
        line_scan.angleResolutionY = resolution_y

        line_scan.frame_colors = np.zeros(
            (frame_count, 3),
            dtype=np.float32
        )

        line_scan.frame_features = np.zeros(
            (frame_count, 3),
            dtype=np.float32
        )

        line_scan.frame_angles = np.zeros(
            (frame_count, resolution_x * resolution_y),
            dtype=np.float32
        )

        # Frame-Verarbeitung
        def process_frame(rgb_array, frame_idx):

            u_rect = laserFlexWrapOptions.rectify_image(rgb_array)

            if u_rect.shape[:2] != (output_height, output_width):
                raise ValueError(
                    f"Unerwartete Bildgröße: {u_rect.shape[:2]}"
                )

            # Mittlere RGB-Farbe
            line_scan.frame_colors[frame_idx, :] = np.mean(
                u_rect,
                axis=(0, 1)
            )

            # Drei Features
            u_cut_feature1 = u_rect[:, :output_width // 2]
            u_cut_feature2 = u_rect[:, output_width // 2:]

            u_cut_feature3 = u_rect[
                output_height // 4:
                3 * output_height // 4,
                :
            ]

            line_scan.frame_features[frame_idx, 0] = np.mean(
                u_cut_feature1
            )

            line_scan.frame_features[frame_idx, 1] = np.mean(
                u_cut_feature2
            )

            line_scan.frame_features[frame_idx, 2] = np.mean(
                u_cut_feature3
            )

            # Kacheln
            tiles = u_rect.reshape(
                resolution_y,
                tile_height,
                resolution_x,
                tile_width,
                3
            )

            tile_means = np.mean(
                tiles,
                axis=(1, 3, 4)
            )

            line_scan.frame_angles[frame_idx, :] = (
                tile_means.flatten()
            )

        # Alle Frames verarbeiten
        with tqdm(
            total=frame_count,
            desc="Verarbeite Frames"
        ) as progress:

            def callback(rgb_array, frame_idx):
                process_frame(rgb_array, frame_idx)
                progress.update(1)

            laserFlexInputVideo.foreachFrame(callback)

        return line_scan


# -------------------------- LINE SCAN DATA --------------------------


class LaserFlexLineData:

    def __init__(self, file_path=None):
        self.file_path = file_path

        # Metadaten
        self.frame_count = 0
        self.fps = 0.0

        self.angleResolutionX = 0
        self.angleResolutionY = 0

        # Daten
        self.frame_colors = None
        self.frame_features = None
        self.frame_angles = None

    def save(self, file_path=None):
        """Speichert die LineScan-Daten als komprimierte NPZ-Datei."""

        path = file_path or self.file_path

        if path is None:
            raise ValueError("Kein Dateipfad angegeben.")

        directory = os.path.dirname(os.path.abspath(path))
        os.makedirs(directory, exist_ok=True)

        np.savez_compressed(
            path,
            frame_count=self.frame_count,
            fps=self.fps,
            angleResolutionX=self.angleResolutionX,
            angleResolutionY=self.angleResolutionY,
            frame_colors=self.frame_colors,
            frame_features=self.frame_features,
            frame_angles=self.frame_angles
        )

        self.file_path = path

    def load(self, file_path=None):
        """Lädt LineScan-Daten aus einer NPZ-Datei."""

        path = file_path or self.file_path

        if path is None:
            raise ValueError("Kein Dateipfad angegeben.")

        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"LineScan-Datei nicht gefunden: {path}"
            )

        with np.load(path, allow_pickle=False) as data:
            self.frame_count = int(data["frame_count"])
            self.fps = float(data["fps"])

            self.angleResolutionX = int(data["angleResolutionX"])
            self.angleResolutionY = int(data["angleResolutionY"])

            self.frame_colors = data["frame_colors"]
            self.frame_features = data["frame_features"]
            self.frame_angles = data["frame_angles"]

        self.file_path = path

        return self

    def getFrameCount(self):
        return self.frame_count

    def getFrameColors(self):
        return self.frame_colors

    def getFrameFeatures(self):
        return self.frame_features

    def getFrameAngles(self):
        return self.frame_angles

# -------------------------- LINE SCAN 2 IMAGE --------------------------

    
class LaserFlexLineToImageOptions:

    def __init__(self, file_path):
        self.file_path = file_path

        self.frames_per_line = 400
        self.offset = 400

    def save(self):
        """Speichert die Einstellungen als JSON."""

        data = {
            "frames_per_line": self.frames_per_line,
            "offset": self.offset
        }

        directory = os.path.dirname(
            os.path.abspath(self.file_path)
        )
        os.makedirs(directory, exist_ok=True)

        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)

    def load(self):
        """Lädt die Einstellungen aus einer JSON-Datei."""

        if not os.path.isfile(self.file_path):
            raise FileNotFoundError(
                f"Optionsdatei nicht gefunden: {self.file_path}"
            )

        with open(self.file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.frames_per_line = float(data["frames_per_line"])
        self.offset = float(data["offset"])

        return self

    def reconstruct_rgb(self, laserFlexLineData):
        """
        Schnelle RGB-Vorschau.
        Nur frame_colors werden verwendet.
        """

        return reconstruct_linescan_rgb(
            laserFlexLineData,
            self.frames_per_line,
            self.offset
        )

    def process(
        self,
        laserFlexLineData,
        output_laserFlexOutput
    ):
        """Rekonstruiert alle Linescan-Bilder."""

        from tqdm.notebook import tqdm

        data = laserFlexLineData
        output = output_laserFlexOutput

        if data.frame_colors is None:
            raise ValueError("Keine LineScan-Daten vorhanden.")

        if self.frames_per_line <= 0:
            raise ValueError("frames_per_line muss positiv sein.")

        # Metadaten
        output.frames_per_line = self.frames_per_line
        output.offset = self.offset

        output.angleResolutionX = data.angleResolutionX
        output.angleResolutionY = data.angleResolutionY

        # RGB-Bild
        output.image_colors = reconstruct_linescan_rgb(
            data,
            self.frames_per_line,
            self.offset
        )

        # Drei Features als separate Graustufenbilder
        feature_images = []

        for i in tqdm(range(data.frame_features.shape[1]),
                      desc="Features rekonstruieren"):

            feature = data.frame_features[:, i]

            image = reconstruct_linescan(
                feature,
                self.frames_per_line,
                self.offset
            )

            feature_images.append(image[:, :, 0])

        output.image_features = np.stack(feature_images, axis=0)

        # Alle Winkel-/Kachelpositionen rekonstruieren
        angle_images = []

        for i in tqdm(range(data.frame_angles.shape[1]),
                      desc="Winkelbilder rekonstruieren"):

            angle = data.frame_angles[:, i]

            image = reconstruct_linescan(
                angle,
                self.frames_per_line,
                self.offset
            )

            angle_images.append(image[:, :, 0])

        output.image_angles = np.stack(angle_images, axis=0)

        return output

def reconstruct_linescan(data, frames_per_line=400, offset=0):
    """
    Rekonstruiert ein Linescan-Bild.

    data: (frame_count, channels)
    Rückgabe: (n_lines, width, channels)
    """

    data = np.asarray(data)

    if data.ndim == 1:
        data = data[:, np.newaxis]

    if data.ndim != 2:
        raise ValueError("data muss ein 1D- oder 2D-Array sein.")

    if frames_per_line <= 0:
        raise ValueError("frames_per_line muss positiv sein.")

    n_frames, n_channels = data.shape

    width = max(1, int(round(frames_per_line)))

    if offset < 0 or offset >= n_frames:
        raise ValueError("offset liegt außerhalb des Datenbereichs.")

    n_lines = int(np.ceil(
        (n_frames - offset) / frames_per_line
    ))

    image = np.zeros(
        (n_lines, width, n_channels),
        dtype=np.uint8
    )

    for line in range(n_lines):

        start = int(round(offset + line * frames_per_line))
        end = int(round(offset + (line + 1) * frames_per_line))
        end = min(end, n_frames)

        line_data = data[start:end]

        if len(line_data) == 0:
            continue

        if line % 2 == 1:
            line_data = line_data[::-1]

        old_x = np.linspace(0, 1, len(line_data))
        new_x = np.linspace(0, 1, width)

        for channel in range(n_channels):
            values = np.interp(
                new_x,
                old_x,
                line_data[:, channel]
            )

            image[line, :, channel] = np.clip(
                values, 0, 255
            ).astype(np.uint8)

    return image

def reconstruct_linescan_rgb(line_scan_data,frames_per_line=400,offset=0):
    return reconstruct_linescan(line_scan_data.frame_colors,frames_per_line,offset)




class LaserFlexOutput:

    def __init__(self, file_path=None):
        self.file_path = file_path

        self.image_colors = None
        self.image_features = None
        self.image_angles = None

        self.frames_per_line = 0
        self.offset = 0

        self.angleResolutionX = 0
        self.angleResolutionY = 0

    def save(self, file_path=None):
        """Speichert alle rekonstruierten Bilder."""

        path = file_path or self.file_path

        if path is None:
            raise ValueError("Kein Dateipfad angegeben.")

        directory = os.path.dirname(os.path.abspath(path))
        os.makedirs(directory, exist_ok=True)

        np.savez_compressed(
            path,
            image_colors=self.image_colors,
            image_features=self.image_features,
            image_angles=self.image_angles,
            frames_per_line=self.frames_per_line,
            offset=self.offset,
            angleResolutionX=self.angleResolutionX,
            angleResolutionY=self.angleResolutionY
        )

        self.file_path = path

    def load(self, file_path=None):
        """Lädt rekonstruierte Bilder."""

        path = file_path or self.file_path

        if path is None:
            raise ValueError("Kein Dateipfad angegeben.")

        if not os.path.isfile(path):
            raise FileNotFoundError(path)

        with np.load(path, allow_pickle=False) as data:

            self.image_colors = data["image_colors"]
            self.image_features = data["image_features"]
            self.image_angles = data["image_angles"]

            self.frames_per_line = float(data["frames_per_line"])
            self.offset = float(data["offset"])

            self.angleResolutionX = int(data["angleResolutionX"])
            self.angleResolutionY = int(data["angleResolutionY"])

        self.file_path = path

        return self

    def getImageColors(self):
        return self.image_colors

    def getImageFeatures(self):
        return self.image_features

    def getImageAngles(self):
        return self.image_angles

    def getAngleImage(self, x, y):
        """Gibt das Linescan-Bild einer bestimmten Kachel zurück."""

        if self.image_angles is None:
            raise RuntimeError("Keine Winkelbilder vorhanden.")

        if not (0 <= x < self.angleResolutionX):
            raise IndexError("x außerhalb der Winkelauflösung.")

        if not (0 <= y < self.angleResolutionY):
            raise IndexError("y außerhalb der Winkelauflösung.")

        index = y * self.angleResolutionX + x

        return self.image_angles[index]