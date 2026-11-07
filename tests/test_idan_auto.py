"""Run: python -m unittest discover -s tests -v"""

import sqlite3
import tempfile
import unittest
import xml.dom.minidom
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from idan_auto.analyze import analyze
from idan_auto.catalog import iter_catalog_edits, parse_lua_settings
from idan_auto.model import StyleModel, blend, feature_vector, targets_from_settings
from idan_auto.rules import DEFAULT_CONFIG, clamp_and_round, compute_settings
from idan_auto.xmp import build_xmp, write_sidecar


def showroom(color, gain=1.0, cast=(1.0, 1.0, 1.0), size=(900, 600)):
    w, h = size
    img = Image.new("RGB", size, (200, 200, 198))
    d = ImageDraw.Draw(img)
    for y in range(int(h * 0.62), h):  # floor gradient
        g = int(150 + 40 * (y - h * 0.62) / (h * 0.38))
        d.line([(0, y), (w, y)], fill=(g, g, g - 2))
    d.rounded_rectangle([w * 0.18, h * 0.38, w * 0.82, h * 0.72], radius=40, fill=color)
    d.rounded_rectangle([w * 0.32, h * 0.26, w * 0.66, h * 0.42], radius=30, fill=color)
    d.polygon([(w * 0.36, h * 0.29), (w * 0.62, h * 0.29), (w * 0.60, h * 0.39), (w * 0.38, h * 0.39)],
              fill=(40, 45, 50))
    for cx in (0.30, 0.70):
        d.ellipse([w * cx - 55, h * 0.64, w * cx + 55, h * 0.64 + 110], fill=(20, 20, 20))
    a = np.asarray(img, dtype=np.float32) / 255
    a = np.clip(a * gain * np.array(cast), 0, 1)
    return Image.fromarray((a * 255).astype(np.uint8))


class Analysis(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.dir = Path(cls.tmp.name)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def run_case(self, name, **kw):
        p = self.dir / f"{name}.jpg"
        showroom(**kw).save(p, quality=95)
        a = analyze(p, mask_method="heuristic")
        s, meta = compute_settings(a, DEFAULT_CONFIG)
        return a, clamp_and_round(s), meta

    def test_black_car(self):
        a, s, _ = self.run_case("black", color=(18, 18, 20))
        self.assertEqual(a["car_tone"], "black")
        self.assertGreaterEqual(s["Shadows2012"], 40)
        self.assertGreater(s["Blacks2012"], DEFAULT_CONFIG["base"]["Blacks2012"])

    def test_white_car(self):
        a, s, _ = self.run_case("white", color=(240, 240, 238))
        self.assertEqual(a["car_tone"], "white")
        self.assertLessEqual(s["Highlights2012"], -70)
        self.assertLess(s["Whites2012"], 0)

    def test_red_car(self):
        a, s, _ = self.run_case("red", color=(190, 20, 25))
        self.assertEqual(a["car_color"], "red")
        self.assertLess(s["SaturationAdjustmentRed"], 0)

    def test_blue_car(self):
        a, _, _ = self.run_case("blue", color=(20, 50, 160))
        self.assertEqual(a["car_color"], "blue")

    def test_warm_hall_light_is_corrected(self):
        a, s, meta = self.run_case("warm", color=(235, 235, 235), cast=(1.0, 0.85, 0.6))
        self.assertGreater(a["cast_rb"], 0.2)
        self.assertTrue(a["artificial_light"])
        self.assertLess(s["IncrementalTemperature"], 0)  # cooler

    def test_exposure_normalisation(self):
        _, dark, _ = self.run_case("silver_dark", color=(150, 150, 152), gain=0.5)
        _, bright, _ = self.run_case("silver_bright", color=(150, 150, 152), gain=1.0)
        self.assertGreater(dark["Exposure2012"], bright["Exposure2012"] + 0.6)


class RawPath(unittest.TestCase):
    def fake_analysis(self, **kw):
        a = dict(raw=True, as_shot_kelvin=5500.0, car_tone="white", car_color="white",
                 frame_std=0.2, clip_hi=0.0, clip_lo=0.0, low_light=False, evening=True,
                 artificial_light=True, iso=3200, car_lin=0.2, car_clip_hi=0.0,
                 cast_rb=0.4, cast_g=0.0, wb_ref="car-paint", car_L_norm=0.8, car_sat=0.05,
                 car_chroma_frac=0.0, car_hue=None, frame_mean=0.4)
        a.update(kw)
        return a

    def test_absolute_temperature_and_noise(self):
        s, _ = compute_settings(self.fake_analysis(), DEFAULT_CONFIG)
        s = clamp_and_round(s)
        self.assertLess(s["Temperature"], 5500)
        self.assertGreater(s["LuminanceSmoothing"], 30)

    def test_sidecar_written_and_valid(self):
        with tempfile.TemporaryDirectory() as td:
            raw = Path(td) / "IMG_0001.CR3"
            raw.write_bytes(b"x")
            s = clamp_and_round(compute_settings(self.fake_analysis(), DEFAULT_CONFIG)[0])
            xmp, status = write_sidecar(raw, s)
            self.assertEqual(xmp.name, "IMG_0001.xmp")
            self.assertEqual(status, "written")
            doc = xml.dom.minidom.parse(str(xmp))
            desc = doc.getElementsByTagName("rdf:Description")[0]
            self.assertEqual(desc.getAttribute("crs:WhiteBalance"), "Custom")
            self.assertTrue(desc.getAttribute("crs:Exposure2012").startswith(("+", "-")))
            _, status = write_sidecar(raw, s)
            self.assertIn("skipped", status)
            _, status = write_sidecar(raw, s, force=True)
            self.assertIn("backup", status)
            self.assertTrue((Path(td) / "IMG_0001.xmp.bak").exists())

    def test_build_xmp_parses(self):
        xml.dom.minidom.parseString(build_xmp({"Exposure2012": 0.45, "Shadows2012": 38}))


LUA = 's = { AutoLateralCA = 1, Exposure2012 = 0.45, Shadows2012 = 38, Highlights2012 = -62, ' \
      'Temperature = 5100, Tint = 4, WhiteBalance = "Custom", Clarity2012 = 10, ' \
      'ToneCurvePV2012 = { 0, 0, 255, 255, }, MaskGroupBasedCorrections = { { Exposure2012 = 9, }, }, }'


class Catalog(unittest.TestCase):
    def test_parse_lua(self):
        d = parse_lua_settings(LUA)
        self.assertEqual(d["Exposure2012"], 0.45)
        self.assertEqual(d["WhiteBalance"], "Custom")
        self.assertEqual(d["Highlights2012"], -62)

    def test_catalog_and_model(self):
        with tempfile.TemporaryDirectory() as td:
            cat = Path(td) / "Test.lrcat"
            con = sqlite3.connect(cat)
            con.executescript("""
              CREATE TABLE AgLibraryRootFolder(id_local INTEGER, absolutePath TEXT);
              CREATE TABLE AgLibraryFolder(id_local INTEGER, rootFolder INTEGER, pathFromRoot TEXT);
              CREATE TABLE AgLibraryFile(id_local INTEGER, folder INTEGER, baseName TEXT, extension TEXT);
              CREATE TABLE Adobe_images(id_local INTEGER, rootFile INTEGER, captureTime TEXT, masterImage INTEGER);
              CREATE TABLE Adobe_imageDevelopSettings(image INTEGER, text TEXT);
              INSERT INTO AgLibraryRootFolder VALUES (1, '/Volumes/Photos/');
              INSERT INTO AgLibraryFolder VALUES (1, 1, '2026/car1/');
              INSERT INTO AgLibraryFile VALUES (1, 1, 'IMG_1', 'CR3');
              INSERT INTO AgLibraryFile VALUES (2, 1, 'IMG_1', 'CR3');
              INSERT INTO Adobe_images VALUES (1, 1, '2026-01-01', NULL);
              INSERT INTO Adobe_images VALUES (2, 2, '2026-01-01', 1);
            """)
            con.execute("INSERT INTO Adobe_imageDevelopSettings VALUES (1, ?)", (LUA,))
            con.execute("INSERT INTO Adobe_imageDevelopSettings VALUES (2, ?)", (LUA,))
            con.commit(); con.close()
            rows = list(iter_catalog_edits(cat, [("/Volumes/Photos/", "/data/")]))
            self.assertEqual(len(rows), 1)  # virtual copy skipped
            self.assertEqual(rows[0]["path"], "/data/2026/car1/IMG_1.CR3")

            a = RawPath().fake_analysis()
            t = targets_from_settings(rows[0]["settings"], a)
            self.assertIsNotNone(t)
            X = [feature_vector(a)] * 40
            m = StyleModel(X, [t] * 40)
            path = Path(td) / "m.json"
            m.save(path)
            m2 = StyleModel.load(path)
            out, info = blend({"Shadows2012": 0, "Exposure2012": 0}, a, m2)
            self.assertGreater(info["model_weight"], 0.4)
            self.assertGreater(out["Shadows2012"], 15)
            self.assertIn("Temperature", out)

    def test_unedited_ignored(self):
        self.assertIsNone(targets_from_settings({"Exposure2012": 0}, {}))


if __name__ == "__main__":
    unittest.main()
