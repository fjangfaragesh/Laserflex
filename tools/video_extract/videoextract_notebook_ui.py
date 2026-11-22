import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets

from IPython.display import display


def rectify_image_ui(laserFlexData):
    laserFlexInputVideo = laserFlexData.video
    laserFlexWrapOptions = laserFlexData.wrapOptions
    
    # Eckpunkte initialisieren
    def get_points():
        return np.array([
            getattr(laserFlexWrapOptions, name)
            for name in point_names
        ], dtype=float)
    
    # Eckpunkte und Farben
    point_names = [
        "corner00",
        "corner01",
        "corner10",
        "corner11"
    ]

    point_colors = [
        "black",
        "red",
        "green",
        "yellow"
    ]

    # Aktueller Frame
    frame_count = laserFlexInputVideo.getFrameCount()

    frame_slider = widgets.IntSlider(
        value=0,
        min=0,
        max=frame_count - 1,
        step=1,
        description="Frame:",
        continuous_update=False,
        layout=widgets.Layout(width="500px")
    )

    save_button = widgets.Button(
        description="Einstellungen speichern",
        button_style="success",
        icon="save"
    )

    status = widgets.Label(value="")

    # Matplotlib Figure
    fig, (ax_original, ax_rectified) = plt.subplots(
        1, 2,
        figsize=(14, 7)
    )

    fig.tight_layout()

    # Ersten Frame laden
    rgb = laserFlexInputVideo.getFrame(0)

    image_original = ax_original.imshow(rgb)
    ax_original.set_title("Originalbild")
    ax_original.axis("on")

    # Entzerrtes Bild
    rectified = laserFlexWrapOptions.rectify_image(rgb)

    image_rectified = ax_rectified.imshow(rectified)
    ax_rectified.set_title("Entzerrtes Bild")
    ax_rectified.axis("on")

    # Eckpunkte als Scatter-Plot
    initial_points = get_points()

    points_artist = ax_original.scatter(
        initial_points[:, 0],
        initial_points[:, 1],
        s=100,
        c=point_colors,
        edgecolors="white",
        linewidths=1.5,
        picker=True
    )



    def update_points():
        points = get_points()

        points_artist.set_offsets(points)

        fig.canvas.draw_idle()

    update_points()

    # Live-Entzerrung aktualisieren
    def update_rectified():
        points = get_points()

        # Punkte in den Optionen speichern
        for i, name in enumerate(point_names):
            setattr(
                laserFlexWrapOptions,
                name,
                points[i].astype(int).tolist()
            )

        rectified = laserFlexWrapOptions.rectify_image(rgb)

        image_rectified.set_data(rectified)

        ax_rectified.set_title(
            f"Entzerrt: {rectified.shape[1]} x {rectified.shape[0]}"
        )

        fig.canvas.draw_idle()

    # Frame aktualisieren
    def on_frame_change(change):
        nonlocal rgb

        rgb = laserFlexInputVideo.getFrame(change["new"])

        image_original.set_data(rgb)

        ax_original.set_title(
            f"Originalbild - Frame {change['new']}"
        )

        update_rectified()

    frame_slider.observe(
        on_frame_change,
        names="value"
    )

   # Drag & Drop der Eckpunkte
    selected_point = None

    def on_mouse_press(event):
        nonlocal selected_point

        if event.inaxes != ax_original:
            return

        if event.button != 1:
            return

        if event.xdata is None or event.ydata is None:
            return

        points = get_points()

        distances = np.linalg.norm(
            points - np.array([event.xdata, event.ydata]),
            axis=1
        )

        nearest = np.argmin(distances)

        # Auswahlradius in Bildpixeln
        if distances[nearest] < 30:
            selected_point = nearest


    def on_mouse_move(event):
        if selected_point is None:
            return

        if event.inaxes != ax_original:
            return

        if event.xdata is None or event.ydata is None:
            return

        x = int(round(event.xdata))
        y = int(round(event.ydata))

        x = np.clip(x, 0, rgb.shape[1] - 1)
        y = np.clip(y, 0, rgb.shape[0] - 1)

        points = get_points()
        points[selected_point] = [x, y]

        # Punktposition aktualisieren
        points_artist.set_offsets(points)

        # Optionen aktualisieren
        name = point_names[selected_point]
        setattr(
            laserFlexWrapOptions,
            name,
            [x, y]
        )

        # Entzerrung aktualisieren
        rectified = laserFlexWrapOptions.rectify_image(rgb)
        image_rectified.set_data(rectified)

        fig.canvas.draw_idle()


    def on_mouse_release(event):
        nonlocal selected_point
        selected_point = None


    fig.canvas.mpl_connect(
        "button_press_event",
        on_mouse_press
    )

    fig.canvas.mpl_connect(
        "motion_notify_event",
        on_mouse_move
    )

    fig.canvas.mpl_connect(
        "button_release_event",
        on_mouse_release
    )

    def on_mouse_move(event):
        if selected_point is None:
            return

        if event.inaxes != ax_original:
            return

        if event.xdata is None or event.ydata is None:
            return

        x = int(round(event.xdata))
        y = int(round(event.ydata))

        # Bildgrenzen beachten
        x = np.clip(x, 0, rgb.shape[1] - 1)
        y = np.clip(y, 0, rgb.shape[0] - 1)

        points = get_points()
        points[selected_point] = [x, y]

        update_points()
        update_rectified()

    def on_mouse_release(event):
        nonlocal selected_point
        selected_point = None

    fig.canvas.mpl_connect(
        "button_press_event",
        on_mouse_press
    )

    fig.canvas.mpl_connect(
        "motion_notify_event",
        on_mouse_move
    )

    fig.canvas.mpl_connect(
        "button_release_event",
        on_mouse_release
    )

    # Speichern
    def on_save_clicked(button):
        try:
            update_rectified()
            laserFlexWrapOptions.save()

            status.value = (
                f"Gespeichert: {laserFlexWrapOptions.file_path}"
            )

        except Exception as e:
            status.value = f"Fehler beim Speichern: {e}"

    save_button.on_click(on_save_clicked)

    # Oberfläche anzeigen
    controls = widgets.HBox([
        frame_slider,
        save_button
    ])

    ui = widgets.VBox([
        controls,
        status
    ])

    display(ui)
    plt.show()

    return ui














import ipywidgets as widgets
import matplotlib.pyplot as plt
from IPython.display import display, clear_output

def line2image_ui(laserFlexData):
    laserFlexLineScanData = laserFlexData.lineData
    laserFlexLineToImageOptions = laserFlexData.lineToImageOptions
    
    frames_slider = widgets.FloatSlider(
        value=float(laserFlexLineToImageOptions.frames_per_line),
        min=10,
        max=1000,
        step=0.01,
        description="Frames/Line:",
        continuous_update=True,
        readout_format=".2f",
        style={"description_width": "initial"},
        layout=widgets.Layout(width="600px")
    )

    offset_slider = widgets.FloatSlider(
        value=float(laserFlexLineToImageOptions.offset),
        min=0,
        max=1000,
        step=0.01,
        description="Offset:",
        continuous_update=True,
        readout_format=".2f",
        style={"description_width": "initial"},
        layout=widgets.Layout(width="600px")
    )

    save_button = widgets.Button(
        description="Einstellungen speichern",
        button_style="success",
        icon="save"
    )

    status = widgets.HTML()

    # Figure und Vorschau
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis("off")

    laserFlexLineToImageOptions.frames_per_line = frames_slider.value
    laserFlexLineToImageOptions.offset = offset_slider.value

    preview = laserFlexLineToImageOptions.reconstruct_rgb(
        laserFlexLineScanData
    )

    image = ax.imshow(preview, aspect="auto")
    ax.set_aspect("equal", adjustable="box")

    fig.tight_layout()

    # Vorschau aktualisieren
    def update_preview(change=None):

        laserFlexLineToImageOptions.frames_per_line = float(frames_slider.value)
        laserFlexLineToImageOptions.offset = float(offset_slider.value)

        preview = laserFlexLineToImageOptions.reconstruct_rgb(
            laserFlexLineScanData
        )

        image.set_data(preview)

        height, width = preview.shape[:2]

        # Bildgrenzen aktualisieren
        image.set_extent((-0.5, width - 0.5, height - 0.5, -0.5))

        # Echtes Seitenverhältnis beibehalten
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(-0.5, width - 0.5)
        ax.set_ylim(height - 0.5, -0.5)

        ax.set_title(
            f"RGB-Vorschau: "
            f"{frames_slider.value:.2f} Frames/Line, "
            f"Offset {offset_slider.value:.2f}"
        )

        fig.canvas.draw_idle()

    frames_slider.observe(update_preview, names="value")
    offset_slider.observe(update_preview, names="value")

    # Speichern
    def save_settings(button):

        laserFlexLineToImageOptions.frames_per_line = float(frames_slider.value)
        laserFlexLineToImageOptions.offset = float(offset_slider.value)

        laserFlexLineToImageOptions.save()

        status.value = (
            "<span style='color:green'>Einstellungen gespeichert.</span>"
        )

    save_button.on_click(save_settings)

    ui = widgets.VBox([
        frames_slider,
        offset_slider,
        save_button,
        status
    ])

    display(ui)

    
    
    
    
def show_output_ui(laserFlexData):
    laserFlexOutput = laserFlexData.output

    angles = laserFlexOutput.getImageAngles()

    if angles is None:
        raise ValueError("Keine Winkelbilder vorhanden.")

    res_x = laserFlexOutput.angleResolutionX
    res_y = laserFlexOutput.angleResolutionY

    height, width = angles.shape[1:3]

    if angles.shape[0] != res_x * res_y:
        raise ValueError("Winkelauflösung passt nicht zu den Daten.")

    colors = ["red", "lime", "blue"]
    channel_names = ["Rot", "Grün", "Blau"]

    # Startpositionen
    positions = np.array([
        [width * 0.25, height * 0.25],
        [width * 0.50, height * 0.50],
        [width * 0.75, height * 0.75]
    ], dtype=float)

    angle_positions = np.array([
        [res_x * 0.25, res_y * 0.25],
        [res_x * 0.50, res_y * 0.50],
        [res_x * 0.75, res_y * 0.75]
    ], dtype=float)

    positions[:, 0] = np.clip(positions[:, 0], 0, width - 1)
    positions[:, 1] = np.clip(positions[:, 1], 0, height - 1)

    angle_positions[:, 0] = np.clip(angle_positions[:, 0], 0, res_x - 1)
    angle_positions[:, 1] = np.clip(angle_positions[:, 1], 0, res_y - 1)

    # Kanalzustände
    channel_enabled = [True, True, True]

    # RGB-Bilder
    def make_rgb_image(data):
        return np.clip(data, 0, 255).astype(np.uint8)

    def get_xy_image():

        channels = []

        for i, (x, y) in enumerate(angle_positions):

            if channel_enabled[i]:
                channels.append(
                    laserFlexOutput.getAngleImage(
                        int(round(x)),
                        int(round(y))
                    )
                )
            else:
                channels.append(
                    np.zeros((height, width), dtype=np.uint8)
                )

        return make_rgb_image(np.stack(channels, axis=-1))

    def get_angle_image():

        channels = []

        for i, (x, y) in enumerate(positions):

            if channel_enabled[i]:
                px = int(round(x))
                py = int(round(y))

                channel = angles[:, py, px].reshape(res_y, res_x)
                channels.append(channel)
            else:
                channels.append(
                    np.zeros((res_y, res_x), dtype=np.uint8)
                )

        return make_rgb_image(np.stack(channels, axis=-1))

    # Figure
    fig, (ax_xy, ax_angle) = plt.subplots(
        1, 2,
        figsize=(15, 7)
    )

    fig.subplots_adjust(wspace=0.25)

    image_xy = ax_xy.imshow(get_xy_image(), origin="upper")
    image_angle = ax_angle.imshow(get_angle_image(), origin="upper")

    ax_xy.set_title("X-Y-Raum")
    ax_xy.set_xlabel("X")
    ax_xy.set_ylabel("Y")
    ax_xy.set_aspect("equal")

    ax_angle.set_title("Winkelraum")
    ax_angle.set_xlabel("Winkel X")
    ax_angle.set_ylabel("Winkel Y")
    ax_angle.set_aspect("equal")

    # Fadenkreuze: kurze Linien
    crosshair_size = 8

    def create_crosshairs(ax, points):

        artists = []

        for i, (x, y) in enumerate(points):

            color = colors[i]

            horizontal, = ax.plot(
                [x - crosshair_size, x + crosshair_size],
                [y, y],
                color=color,
                linewidth=1.2,
                alpha=0.85
            )

            vertical, = ax.plot(
                [x, x],
                [y - crosshair_size, y + crosshair_size],
                color=color,
                linewidth=1.2,
                alpha=0.85
            )

            center = ax.scatter(
                [x], [y],
                s=12,
                color=color,
                edgecolors="white",
                linewidths=0.4,
                zorder=10
            )

            artists.append((horizontal, vertical, center))

        return artists

    xy_crosshairs = create_crosshairs(ax_xy, positions)
    angle_crosshairs = create_crosshairs(ax_angle, angle_positions)

    # Fadenkreuze aktualisieren
    def update_crosshairs(artists, points):

        for i, (horizontal, vertical, center) in enumerate(artists):

            x, y = points[i]

            horizontal.set_data(
                [x - crosshair_size, x + crosshair_size],
                [y, y]
            )

            vertical.set_data(
                [x, x],
                [y - crosshair_size, y + crosshair_size]
            )

            center.set_offsets([[x, y]])

            # Deaktivierte Kanäle vollständig ausblenden
            visible = channel_enabled[i]

            horizontal.set_visible(visible)
            vertical.set_visible(visible)
            center.set_visible(visible)

    # Aktualisierung
    def update_images():

        image_xy.set_data(get_xy_image())
        image_angle.set_data(get_angle_image())

        update_crosshairs(xy_crosshairs, positions)
        update_crosshairs(angle_crosshairs, angle_positions)

        fig.canvas.draw_idle()

    # Kanal-Checkboxen
    channel_checkboxes = []

    def on_channel_change(index):

        def callback(change):
            channel_enabled[index] = change["new"]
            update_images()

        return callback

    for i in range(3):

        checkbox = widgets.Checkbox(
            value=True,
            description=channel_names[i],
            indent=False,
            style={"description_width": "initial"}
        )

        checkbox.observe(on_channel_change(i), names="value")
        channel_checkboxes.append(checkbox)

    channel_controls = widgets.HBox(channel_checkboxes)

    # Drag & Drop
    selected = None

    def find_nearest_crosshair(event, ax, points):

        for i, (x, y) in enumerate(points):

            if not channel_enabled[i]:
                continue

            screen_pos = ax.transData.transform((x, y))

            distance = np.linalg.norm(
                screen_pos - np.array([event.x, event.y])
            )

            if distance < 18:
                return i

        return None

    def on_mouse_press(event):

        nonlocal selected

        if event.button != 1:
            return

        if event.inaxes == ax_xy:

            index = find_nearest_crosshair(event, ax_xy, positions)

            if index is not None:
                selected = ("xy", index)

        elif event.inaxes == ax_angle:

            index = find_nearest_crosshair(
                event, ax_angle, angle_positions
            )

            if index is not None:
                selected = ("angle", index)

    def on_mouse_move(event):

        if selected is None:
            return

        if event.xdata is None or event.ydata is None:
            return

        panel, index = selected

        if panel == "xy":

            if event.inaxes != ax_xy:
                return

            positions[index] = [
                np.clip(event.xdata, 0, width - 1),
                np.clip(event.ydata, 0, height - 1)
            ]

        else:

            if event.inaxes != ax_angle:
                return

            angle_positions[index] = [
                np.clip(event.xdata, 0, res_x - 1),
                np.clip(event.ydata, 0, res_y - 1)
            ]

        update_images()

    def on_mouse_release(event):

        nonlocal selected
        selected = None

    fig.canvas.mpl_connect("button_press_event", on_mouse_press)
    fig.canvas.mpl_connect("motion_notify_event", on_mouse_move)
    fig.canvas.mpl_connect("button_release_event", on_mouse_release)

    # UI
    ui = widgets.VBox([
        widgets.HTML("<b>RGB-Kanäle</b>"),
        channel_controls
    ])

    display(ui)
    plt.show()

    return ui