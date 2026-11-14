"""
Thermal reader for GY-MCU90640 (MLX90640) through the ESP32 serial bridge.
Runs in a background thread so it never slows the face-recognition loop.
"""
import struct
import threading
import time

import cv2
import numpy as np
import serial

W, H = 32, 24
FRAME_LEN = 1544


class ThermalReader:
    def __init__(self, port, baud=921600, mirror=False, flip=False,
                 scale=1.0, dx=0.0, dy=0.0, temp_offset=0.0):
        """
        mirror / flip : flip the thermal image so it matches the webcam
        scale         : webcam FOV / thermal FOV (e.g. 70 / 55 = 1.27). 1.0 = same FOV
        dx, dy        : shift of the thermal image, as a fraction of the frame (-0.1 .. 0.1)
        temp_offset   : added to every reading (calibrate against a real thermometer)
        """
        self.mirror, self.flip = mirror, flip
        self.scale, self.dx, self.dy = scale, dx, dy
        self.temp_offset = temp_offset

        self.ser = serial.Serial(port, baud, timeout=0.05)
        self._lock = threading.Lock()
        self._temps = None
        self._ta = None
        self._stamp = 0.0
        self._running = False
        self._thread = threading.Thread(target=self._loop, daemon=True)

    # ---------- lifecycle ----------
    def start(self):
        time.sleep(2)                      # ESP32 resets when the port opens
        self.ser.reset_input_buffer()
        self.ser.write(bytes([0xA5, 0x35, 0x02, 0xDC]))   # automatic output on
        self._running = True
        self._thread.start()
        return self

    def stop(self):
        self._running = False
        time.sleep(0.2)
        self.ser.close()

    # ---------- background reader ----------
    @staticmethod
    def _extract(buf):
        latest = None
        while True:
            i = buf.find(b"\x5a\x5a")
            if i < 0:
                del buf[:-1]
                break
            if i > 0:
                del buf[:i]
            if len(buf) < FRAME_LEN:
                break
            frame = bytes(buf[:FRAME_LEN])
            chk = struct.unpack("<H", frame[-2:])[0]
            px = np.frombuffer(frame[4:4 + W * H * 2], dtype="<i2")
            sane = px.min() > -4000 and px.max() < 30000
            if (sum(frame[:-2]) & 0xFFFF) == chk or sane:
                latest = frame
                del buf[:FRAME_LEN]
            else:
                del buf[:2]
        return latest

    def _loop(self):
        buf = bytearray()
        while self._running:
            try:
                buf += self.ser.read(self.ser.in_waiting or 1)
            except (serial.SerialException, OSError):
                break
            frame = self._extract(buf)
            if frame:
                temps = np.frombuffer(frame[4:4 + W * H * 2], dtype="<i2")
                temps = temps.astype(np.float32).reshape(H, W) / 100.0
                ta = struct.unpack("<h", frame[4 + W * H * 2:6 + W * H * 2])[0] / 100.0
                with self._lock:
                    self._temps, self._ta, self._stamp = temps, ta, time.time()

    # ---------- public API ----------
    def latest(self, max_age=1.5):
        """Newest thermal image (24x32, aligned to the webcam) or None if stale."""
        with self._lock:
            if self._temps is None or time.time() - self._stamp > max_age:
                return None
            t = self._temps.copy()
        if self.mirror:
            t = t[:, ::-1]
        if self.flip:
            t = t[::-1, :]
        return t

    def _to_grid(self, px, py, fw, fh):
        u = (px / fw - 0.5) * self.scale + 0.5 + self.dx
        v = (py / fh - 0.5) * self.scale + 0.5 + self.dy
        return u * W, v * H

    def face_temperature(self, box, frame_w, frame_h):
        """Forehead temperature (C) for a webcam face box (x, y, w, h), or None."""
        t = self.latest()
        if t is None:
            return None
        x, y, w, h = box
        # forehead area: upper part of the face box, middle 50% of its width
        c0, r0 = self._to_grid(x + 0.25 * w, y + 0.05 * h, frame_w, frame_h)
        c1, r1 = self._to_grid(x + 0.75 * w, y + 0.40 * h, frame_w, frame_h)
        ci0 = int(np.clip(np.floor(c0), 0, W - 1))
        ci1 = int(np.clip(np.ceil(c1), ci0 + 1, W))
        ri0 = int(np.clip(np.floor(r0), 0, H - 1))
        ri1 = int(np.clip(np.ceil(r1), ri0 + 1, H))
        region = t[ri0:ri1, ci0:ci1]
        if region.size == 0:
            return None
        return float(region.max()) + self.temp_offset

    def render_panel(self, size):
        """Colormapped thermal image, same size as the webcam frame."""
        fw, fh = size
        t = self.latest()
        if t is None:
            panel = np.zeros((fh, fw, 3), dtype=np.uint8)
            cv2.putText(panel, "No thermal data", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            return panel

        lo, hi = float(t.min()), float(t.max())
        gray = ((t - lo) / max(hi - lo, 0.5) * 255).clip(0, 255).astype(np.uint8)
        big = cv2.resize(gray, (fw, fh), interpolation=cv2.INTER_CUBIC)
        panel = cv2.applyColorMap(big, cv2.COLORMAP_INFERNO)
        cv2.putText(panel, f"{lo:.1f} - {hi:.1f} C", (10, fh - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return panel

    def draw_face_box(self, panel, box, fw, fh, color=(0, 255, 0), text=""):
        """Draw the forehead sampling area on the thermal panel."""
        x, y, w, h = box
        # same forehead region that face_temperature() samples
        c0, r0 = self._to_grid(x + 0.25 * w, y + 0.05 * h, fw, fh)
        c1, r1 = self._to_grid(x + 0.75 * w, y + 0.40 * h, fw, fh)
        # grid coords -> panel pixels
        p0 = (int(c0 / W * fw), int(r0 / H * fh))
        p1 = (int(c1 / W * fw), int(r1 / H * fh))
        cv2.rectangle(panel, p0, p1, color, 2)
        if text:
            cv2.putText(panel, text, (p0[0], max(20, p0[1] - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    def overlay(self, frame, alpha=0.45):
        """Blend the thermal image on top of the webcam frame (for alignment)."""
        t = self.latest()
        if t is None:
            return frame
        fh, fw = frame.shape[:2]
        lo, hi = float(t.min()), float(t.max())
        gray = ((t - lo) / max(hi - lo, 0.5) * 255).clip(0, 255).astype(np.uint8)
        big = cv2.resize(gray, (fw, fh), interpolation=cv2.INTER_CUBIC)
        color = cv2.applyColorMap(big, cv2.COLORMAP_INFERNO)

        # apply scale / shift so the overlay matches the sampling geometry
        s = self.scale
        tx = (0.5 - 0.5 * s + self.dx) * fw
        ty = (0.5 - 0.5 * s + self.dy) * fh
        m = np.float32([[s, 0, tx], [0, s, ty]])   # webcam px -> thermal px
        color = cv2.warpAffine(color, m, (fw, fh),
                               flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        return cv2.addWeighted(color, alpha, frame, 1 - alpha, 0)