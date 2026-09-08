"""Generate SYNTHETIC CNIC / non-CNIC test images. No real personal data.

Test CNIC numbers are obviously-fake sequential digits:
  A = 35201-1234567-1
  B = 42101-7654321-9

Run:  python tests/fixtures/_generate.py
"""
from __future__ import annotations

import os
import random

from PIL import Image, ImageDraw, ImageFilter, ImageFont

HERE = os.path.dirname(__file__)
CNIC_A = "35201-1234567-1"
CNIC_B = "42101-7654321-9"


def _font(size: int) -> ImageFont.FreeTypeFont:
    for p in (
        r"C:\Windows\Fonts\arialbd.ttf",
        r"C:\Windows\Fonts\arial.ttf",
        r"C:\Windows\Fonts\segoeui.ttf",
    ):
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def _save(img: Image.Image, name: str) -> None:
    path = os.path.join(HERE, name)
    img.save(path)
    print(f"  wrote {name}  ({img.size[0]}x{img.size[1]})")


def make_cnic(number: str, name: str) -> Image.Image:
    """A rough National Identity Card mock — green header band, field labels.
    The identity number sits well inside the margins with a large font so a
    clean capture is fully legible (a real clear photo would be)."""
    w, h = 1300, 820
    img = Image.new("RGB", (w, h), "#f4f6f4")
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, w, 110], fill="#1f6f3f")
    d.text((36, 32), "PAKISTAN", font=_font(46), fill="white")
    d.text((460, 44), "National Identity Card", font=_font(30), fill="white")

    d.text((50, 170), "Name", font=_font(26), fill="#555")
    d.text((50, 206), name, font=_font(38), fill="#111")

    d.text((50, 300), "Father Name", font=_font(26), fill="#555")
    d.text((50, 336), "Test Father", font=_font(38), fill="#111")

    d.text((50, 452), "Identity Number", font=_font(30), fill="#555")
    d.text((50, 496), number, font=_font(72), fill="#0a0a0a")   # big, far from edges

    d.text((50, 650), "Date of Birth", font=_font(24), fill="#555")
    d.text((50, 682), "01.01.1990", font=_font(30), fill="#111")
    d.text((760, 650), "Date of Issue", font=_font(24), fill="#555")
    d.text((760, 682), "01.01.2018", font=_font(30), fill="#111")

    d.rectangle([1010, 150, 1240, 440], outline="#999", width=2)
    d.text((1075, 280), "PHOTO", font=_font(26), fill="#999")
    return img


def make_random_photo() -> Image.Image:
    w, h = 900, 600
    img = Image.new("RGB", (w, h))
    px = img.load()
    for y in range(h):
        for x in range(w):
            px[x, y] = (
                (x * 3 + y) % 256,
                (y * 5 + random.randint(0, 40)) % 256,
                (x + y * 2) % 256,
            )
    return img.filter(ImageFilter.GaussianBlur(1.5))


def make_selfie() -> Image.Image:
    w, h = 720, 900
    img = Image.new("RGB", (w, h), "#8fb3d9")
    d = ImageDraw.Draw(img)
    d.ellipse([210, 230, 510, 620], fill="#e8c9a0")          # face
    d.ellipse([120, 470, 600, 1100], fill="#3b4a63")          # shoulders
    d.ellipse([280, 360, 330, 410], fill="#3a2a1a")           # eyes
    d.ellipse([390, 360, 440, 410], fill="#3a2a1a")
    d.arc([300, 430, 420, 540], start=20, end=160, fill="#7a4a2a", width=8)
    return img


def make_passport() -> Image.Image:
    w, h = 1000, 680
    img = Image.new("RGB", (w, h), "#e7e2d4")
    d = ImageDraw.Draw(img)
    d.text((40, 30), "PASSPORT", font=_font(44), fill="#22323f")
    d.text((40, 110), "Type  P     Country Code  PAK", font=_font(26), fill="#333")
    d.text((40, 160), "Passport No.  AB1234567", font=_font(30), fill="#111")
    d.text((40, 220), "Surname  DOE", font=_font(26), fill="#333")
    d.text((40, 260), "Given Names  JOHN", font=_font(26), fill="#333")
    d.text((40, 320), "Nationality  PAKISTANI", font=_font(26), fill="#333")
    d.text((40, 360), "Date of birth  01 JAN 1990", font=_font(26), fill="#333")
    # MRZ (no 13-digit CNIC anywhere)
    d.rectangle([0, h - 150, w, h], fill="#f6f4ec")
    d.text((30, h - 130), "P<PAKDOE<<JOHN<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<", font=_font(28), fill="#111")
    d.text((30, h - 80), "AB1234567PAK9001011M2801019<<<<<<<<<<<<<<04", font=_font(28), fill="#111")
    return img


def make_license() -> Image.Image:
    w, h = 1000, 640
    img = Image.new("RGB", (w, h), "#eef3ec")
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, w, 80], fill="#3a5a8c")
    d.text((28, 22), "DRIVING LICENCE", font=_font(36), fill="white")
    d.text((40, 130), "Licence No.  DLX-99-2021-004567", font=_font(32), fill="#111")
    d.text((40, 200), "Name  JOHN DOE", font=_font(28), fill="#333")
    d.text((40, 250), "Valid Till  31-12-2026", font=_font(28), fill="#333")
    d.text((40, 300), "Vehicle Class  LTV / MCYCLE", font=_font(28), fill="#333")
    d.text((40, 360), "Blood Group  O+", font=_font(28), fill="#333")
    return img


def main() -> None:
    print("Generating synthetic fixtures in", HERE)
    a = make_cnic(CNIC_A, "Test Person A")
    b = make_cnic(CNIC_B, "Test Person B")
    _save(a, "cnic_valid_A.png")
    _save(b, "cnic_valid_B.png")
    _save(make_random_photo(), "random_photo.png")
    _save(make_selfie(), "selfie.png")
    _save(make_passport(), "passport.png")
    _save(make_license(), "license.png")
    # r>=14 is past PaddleOCR's recognition threshold for this card (swept);
    # r<=11 stays legible and would (correctly) match. Use a genuinely
    # unreadable blur so this fixture tests the "cannot detect" path.
    _save(a.filter(ImageFilter.GaussianBlur(16)), "cnic_blurry.png")
    _save(a.filter(ImageFilter.GaussianBlur(10)), "cnic_mild_blur.png")
    _save(a.rotate(90, expand=True), "cnic_rotated.png")
    # illegible: extreme blur + heavy pixel noise so no text survives
    import numpy as np

    base = a.resize((260, 164)).resize(a.size).filter(ImageFilter.GaussianBlur(22))
    arr = np.asarray(base).astype("int16")
    noise = np.random.randint(-90, 90, arr.shape, dtype="int16")
    illegible = Image.fromarray(np.clip(arr + noise, 0, 255).astype("uint8"))
    _save(illegible, "cnic_illegible.png")
    print("done.")


if __name__ == "__main__":
    main()
