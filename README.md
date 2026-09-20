# ray-march-test

Two small CPU ray marching experiments written in Odin.
Everything renders into a 120x120 pixel buffer; Raylib only opens the window and shows the buffer.

## 1. Smiley (`main.odin`)

A yellow smiley face inside a box with six colored walls, drawn with signed distance functions.
Drag with the left mouse button to orbit the camera around the face.

```sh
odin build . -o:speed
./ray-march-test
```

## 2. Parallax (`parallax/`)

The same kind of scene, but the screen is treated as a window into a shadow box.
Moving your head (or the mouse) shifts the view through the window, so the box walls and a ball floating in front of the glass move with real parallax.

```sh
odin build parallax -o:speed
./parallax.bin
```

Move the mouse to move your head. Space toggles an automatic sway.

Optional webcam head tracking, from the `parallax/` directory:

```sh
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python tracker.py
```

Run it alongside the renderer and your eye position drives the view. Press C to re-center.
