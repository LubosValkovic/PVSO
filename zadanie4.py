import argparse
import json
import re
import shutil
import statistics
import time
from pathlib import Path

import cv2
import numpy as np

try:
    from ximea import xiapi
except ImportError:
    xiapi = None

try:
    from PIL import Image
except ImportError:
    Image = None

try:
    from pillow_heif import register_heif_opener
except ImportError:
    register_heif_opener = None

if register_heif_opener is not None and Image is not None:
    register_heif_opener()

IMAGE_EXTENSIONS = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff", "*.heic", "*.HEIC")


def load_json(path):
    path = Path(path)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def parse_training_metrics(log_path, iteration):
    log_path = Path(log_path)
    if not log_path.exists():
        return {}

    pattern = re.compile(rf"\[ITER {iteration}\] Evaluating (test|train): L1 ([0-9.]+) PSNR ([0-9.]+)")
    metrics = {}

    for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = pattern.search(line)
        if not match:
            continue

        split_name = match.group(1)
        metrics[f"{split_name}_l1"] = float(match.group(2))
        metrics[f"{split_name}_psnr"] = float(match.group(3))

    return metrics


def read_registered_camera_count(cameras_json_path):
    payload = load_json(cameras_json_path)
    if isinstance(payload, list):
        return len(payload)
    return None


def read_ply_vertex_count(ply_path):
    ply_path = Path(ply_path)
    if not ply_path.exists():
        return None

    with ply_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if line.startswith("element vertex "):
                return int(line.split()[-1])
            if line.strip() == "end_header":
                break

    return None


def format_psnr(value):
    if value is None:
        return "DOPLN"
    return f"{value:.2f} dB"


def format_optional_int(value):
    if value is None:
        return "DOPLN"
    return str(value)


def build_self_scene_table_note(summary, registered_cameras, blur_warning_threshold):
    notes = []

    if summary["sharpness_mean"] < blur_warning_threshold:
        notes.append("malo ostrych zaberov")

    if registered_cameras is not None and summary["count"] > 0:
        registration_ratio = registered_cameras / summary["count"]
        if registration_ratio < 0.5:
            notes.append(f"zaregistrovalo sa len {registered_cameras} / {summary['count']} kamier")
        elif registration_ratio < 0.8:
            notes.append(f"zaregistrovalo sa {registered_cameras} / {summary['count']} kamier")

    if not notes:
        return "bez zjavnych technickych problemov"

    return ", ".join(notes)


def build_self_scene_commentary(original_summary, training_summary, registered_cameras, psnr_7000, psnr_30000):
    comments = []

    if original_summary["count"] != training_summary["count"]:
        comments.append(
            f"Kuracia znizila dataset z `{original_summary['count']}` na `{training_summary['count']}` snimok, "
            "ale odstranila len cast problemovych zaberov."
        )

    if registered_cameras is not None and training_summary["count"] > 0:
        registration_ratio = registered_cameras / training_summary["count"]
        if registration_ratio < 0.5:
            comments.append(
                f"Zo `{training_summary['count']}` kuratorovanych fotografii sa zaregistrovalo len "
                f"`{registered_cameras}`, co ukazuje na slabu konzistenciu alebo nedostatok spolahlivej textury."
            )
        elif registration_ratio < 0.8:
            comments.append(
                f"Zo `{training_summary['count']}` fotografii sa zaregistrovalo `{registered_cameras}`, "
                "takze rekonstrukcia je pouzitelna, ale cast pohladov sa do sparse modelu nedostala."
            )
        else:
            comments.append(
                f"COLMAP zaregistroval vacsinu datasetu (`{registered_cameras}` z `{training_summary['count']}` fotiek), "
                "co je pre vlastnu scenu slusny zaklad pre trening."
            )

    if training_summary["sharpness_mean"] < 8.0:
        comments.append(
            f"Aj kuratorovany set ma nizku priemernu ostrost `{training_summary['sharpness_mean']:.2f}`, "
            "takze COLMAP aj 3DGS pracovali s rozmazanymi vstupmi."
        )

    if psnr_7000 is not None and psnr_30000 is not None:
        if psnr_30000 < psnr_7000:
            comments.append(
                f"Testovaci PSNR klesol z `{psnr_7000:.2f} dB` na `{psnr_30000:.2f} dB`, "
                "co naznacuje preucenie na malom alebo nekvalitnom train sete."
            )
        else:
            comments.append(
                f"Testovaci PSNR narastol z `{psnr_7000:.2f} dB` na `{psnr_30000:.2f} dB`, "
                "takze dalsie iteracie pomohli dotiahnut kvalitu rekonstrukcie."
            )

    if not comments:
        comments.append("Vlastna scena neukazuje zjavny technicky problem, ale stale ju treba porovnat s referencnym datasetom.")

    return comments


def collect_image_files(source_dir):
    files = []
    seen = set()
    for extension in IMAGE_EXTENSIONS:
        for path in source_dir.glob(extension):
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            files.append(path)
    return sorted(files)


def read_image(path):
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".heic":
        if Image is None or register_heif_opener is None:
            raise RuntimeError(
                "HEIC support is not available. Install 'pillow' and 'pillow-heif' to read .heic files."
            )
        image = Image.open(path).convert("RGB")
        return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    return cv2.imread(str(path), cv2.IMREAD_COLOR)


def bgr_from_ximea(image):
    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    if image.ndim == 3 and image.shape[2] == 3:
        return image
    return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)


def to_gray(image):
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def compute_sharpness(image):
    gray = to_gray(image)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def compute_brightness(image):
    gray = to_gray(image)
    return float(gray.mean())


def resize_to_max_side(image, max_side):
    h, w = image.shape[:2]
    longest = max(h, w)

    if longest <= max_side:
        return image

    scale = max_side / longest
    new_size = (int(w * scale), int(h * scale))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)


def save_image(image, output_dir, index, max_side, quality):
    image = resize_to_max_side(image, max_side)
    path = output_dir / f"image_{index:04d}.jpg"
    cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return path


def evaluate_capture_quality(sharpness, min_sharpness):
    if sharpness >= max(min_sharpness, 20.0):
        return "dobra"
    if sharpness >= max(min_sharpness, 8.0):
        return "prijatelna"
    return "slaba"


def draw_help(preview, saved_count, output_dir, max_side, auto_enabled, auto_interval, sharpness, min_sharpness):
    auto_text = "zapnute" if auto_enabled else "vypnute"
    quality = evaluate_capture_quality(sharpness, min_sharpness)
    text_lines = [
        f"Ulozene snimky: {saved_count}",
        f"Vystup: {output_dir}",
        f"Max strana: {max_side}px",
        f"Ostrost: {sharpness:.2f} ({quality})",
        f"Minimalna ostrost: {min_sharpness:.2f}",
        "SPACE / s = ulozit snimku",
        "f = ulozit aj rozmazanu snimku",
        f"a = auto snimanie ({auto_text}, {auto_interval}s)",
        "q = koniec",
    ]

    y = 24
    for text in text_lines:
        cv2.putText(
            preview,
            text,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        y += 24


def capture_dataset(args):
    if xiapi is None:
        raise RuntimeError("Ximea Python SDK nie je dostupne. Prikaz capture vyzaduje modul 'ximea'.")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    cam = xiapi.Camera()
    img = xiapi.Image()
    acquisition_started = False

    print("Opening first Ximea camera...")
    cam.open_device()

    try:
        cam.set_exposure(args.exposure)
        cam.set_param("imgdataformat", "XI_RGB32")
        cam.set_param("auto_wb", 1)

        print(f"Exposure was set to {cam.get_exposure()} us")
        print("Starting data acquisition...")
        cam.start_acquisition()
        acquisition_started = True

        saved_count = len(list(output_dir.glob("image_*.jpg")))
        next_index = saved_count + 1
        auto_enabled = False
        last_auto_save = 0.0

        while True:
            cam.get_image(img)
            frame = bgr_from_ximea(img.get_image_data_numpy())
            sharpness = compute_sharpness(frame)

            preview = resize_to_max_side(frame, args.preview_size)
            draw_help(
                preview,
                saved_count,
                output_dir,
                args.max_side,
                auto_enabled,
                args.auto_interval,
                sharpness,
                args.min_sharpness,
            )
            cv2.imshow("Zadanie 4 - Ximea dataset capture", preview)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            if key == ord("a"):
                auto_enabled = not auto_enabled
                last_auto_save = 0.0
                print(f"Auto capture {'enabled' if auto_enabled else 'disabled'}")

            should_auto_save = auto_enabled and time.time() - last_auto_save >= args.auto_interval
            is_manual_save = key in (ord("s"), ord(" "))
            is_forced_save = key == ord("f")
            can_save = sharpness >= args.min_sharpness or is_forced_save

            if (is_manual_save or should_auto_save or is_forced_save) and not can_save:
                print(
                    f"Skipped blurry frame: sharpness={sharpness:.2f}, "
                    f"required>={args.min_sharpness:.2f}. Press 'f' to force save."
                )

            if (is_manual_save or should_auto_save or is_forced_save) and can_save:
                path = save_image(frame, output_dir, next_index, args.max_side, args.quality)
                print(f"Saved {path} (sharpness={sharpness:.2f})")
                saved_count += 1
                next_index += 1
                last_auto_save = time.time()

    finally:
        if acquisition_started:
            cam.stop_acquisition()
        cam.close_device()
        cv2.destroyAllWindows()
        print("Camera closed.")


def downscale_dataset(args):
    source_dir = Path(args.source)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = collect_image_files(source_dir)
    if not files:
        print(f"No images found in {source_dir}")
        return

    for index, path in enumerate(sorted(files), start=1):
        image = read_image(path)
        if image is None:
            print(f"Skipped unreadable file: {path}")
            continue

        output_path = save_image(image, output_dir, index, args.max_side, args.quality)
        print(f"{path.name} -> {output_path.name}")

    print(f"Done. Downscaled images are in {output_dir}")


def inspect_dataset_dir(source_dir, blur_warning_threshold):
    files = collect_image_files(source_dir)
    if not files:
        raise ValueError(f"No images found in {source_dir}")

    images = []
    for path in files:
        image = read_image(path)
        if image is None:
            continue

        h, w = image.shape[:2]
        images.append(
            {
                "name": path.name,
                "path": path,
                "width": int(w),
                "height": int(h),
                "sharpness": compute_sharpness(image),
                "brightness": compute_brightness(image),
            }
        )

    if not images:
        raise ValueError(f"Images in {source_dir} could not be read.")

    sharpness_values = [item["sharpness"] for item in images]
    brightness_values = [item["brightness"] for item in images]
    widths = [item["width"] for item in images]
    heights = [item["height"] for item in images]

    blur_count = sum(value < blur_warning_threshold for value in sharpness_values)
    long_side = max(widths[0], heights[0])

    return {
        "source_dir": source_dir,
        "count": len(images),
        "width": widths[0],
        "height": heights[0],
        "long_side": long_side,
        "sharpness_mean": statistics.fmean(sharpness_values),
        "sharpness_median": statistics.median(sharpness_values),
        "sharpness_min": min(sharpness_values),
        "sharpness_max": max(sharpness_values),
        "brightness_mean": statistics.fmean(brightness_values),
        "blur_count": blur_count,
        "blur_ratio": blur_count / len(images),
        "images": images,
    }


def build_quality_warnings(summary, blur_warning_threshold):
    warnings = []

    if summary["count"] < 60:
        warnings.append(
            f"Dataset ma len {summary['count']} snimok. Zadanie odporuca aspon 60-120 fotografii."
        )
    if summary["long_side"] < 600:
        warnings.append(
            f"Dlhsia strana je len {summary['long_side']} px. Pre 3DGS sa typicky oplati 600-2000 px."
        )
    if summary["long_side"] > 2000:
        warnings.append(
            f"Dlhsia strana je {summary['long_side']} px. Pred COLMAP je vhodnejsi downscale na 600-2000 px."
        )
    if summary["blur_ratio"] >= 0.25:
        warnings.append(
            f"Rozmazanych snimok je {summary['blur_count']} / {summary['count']} "
            f"(ostrost < {blur_warning_threshold:.2f})."
        )
    if summary["sharpness_mean"] < blur_warning_threshold:
        warnings.append(
            f"Priemerna ostrost {summary['sharpness_mean']:.2f} je nizka. Zaber vyzera slabo zaostreny alebo rozhybany."
        )

    if not warnings:
        warnings.append("Dataset neobsahuje zjavne technicke problemy podla jednoduchych heuristik.")

    return warnings


def write_contact_sheet(summary, output_path, columns, thumb_width):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = (summary["count"] + columns - 1) // columns
    label_height = 24
    gap = 10
    thumb_height = int(round(summary["height"] * (thumb_width / summary["width"])))

    canvas_width = columns * thumb_width + (columns + 1) * gap
    canvas_height = rows * (thumb_height + label_height) + (rows + 1) * gap
    canvas = 255 * cv2.UMat(canvas_height, canvas_width, cv2.CV_8UC3).get()

    for index, item in enumerate(summary["images"]):
        image = cv2.imread(str(item["path"]), cv2.IMREAD_COLOR)
        if image is None:
            continue

        row = index // columns
        col = index % columns
        x = gap + col * (thumb_width + gap)
        y = gap + row * (thumb_height + label_height + gap)

        thumb = cv2.resize(image, (thumb_width, thumb_height), interpolation=cv2.INTER_AREA)
        canvas[y : y + thumb_height, x : x + thumb_width] = thumb

        label = f"{item['name']} | S={item['sharpness']:.1f}"
        cv2.putText(
            canvas,
            label,
            (x, y + thumb_height + 17),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (20, 20, 20),
            1,
            cv2.LINE_AA,
        )

    cv2.imwrite(str(output_path), canvas, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return output_path


def curate_dataset(args):
    source_dir = Path(args.source)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = inspect_dataset_dir(source_dir, args.blur_warning_threshold)
    images = summary["images"]
    target_count = min(args.target_count, len(images))

    if target_count <= 0:
        raise ValueError("target_count must be greater than zero.")

    selected = []
    used_names = set()

    for bucket_index in range(target_count):
        start = int(bucket_index * len(images) / target_count)
        end = int((bucket_index + 1) * len(images) / target_count)
        bucket = images[start:end]
        if not bucket:
            continue

        bucket = sorted(bucket, key=lambda item: item["sharpness"], reverse=True)
        chosen = None
        for candidate in bucket:
            if candidate["sharpness"] >= args.min_sharpness and candidate["name"] not in used_names:
                chosen = candidate
                break

        if chosen is None:
            for candidate in bucket:
                if candidate["name"] not in used_names:
                    chosen = candidate
                    break

        if chosen is None:
            continue

        used_names.add(chosen["name"])
        selected.append(chosen)

    selected = sorted(selected, key=lambda item: item["name"])

    for index, item in enumerate(selected, start=1):
        destination = output_dir / f"image_{index:04d}{item['path'].suffix.lower()}"
        shutil.copy2(item["path"], destination)

    manifest = {
        "source_dir": str(source_dir),
        "output_dir": str(output_dir),
        "requested_count": args.target_count,
        "selected_count": len(selected),
        "min_sharpness": args.min_sharpness,
        "blur_warning_threshold": args.blur_warning_threshold,
        "sharpness_mean": round(statistics.fmean(item["sharpness"] for item in selected), 4) if selected else 0.0,
        "sharpness_min": round(min(item["sharpness"] for item in selected), 4) if selected else 0.0,
        "sharpness_max": round(max(item["sharpness"] for item in selected), 4) if selected else 0.0,
        "selected_files": [
            {
                "source_name": item["name"],
                "sharpness": round(item["sharpness"], 4),
                "brightness": round(item["brightness"], 4),
            }
            for item in selected
        ],
    }

    manifest_path = output_dir.parent / "selection_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(json.dumps(manifest, indent=2))
    print(f"\nCurated dataset saved to {output_dir}")


def inspect_dataset(args):
    source_dir = Path(args.source)
    summary = inspect_dataset_dir(source_dir, args.blur_warning_threshold)
    warnings = build_quality_warnings(summary, args.blur_warning_threshold)

    payload = {
        "source_dir": str(source_dir),
        "count": summary["count"],
        "width": summary["width"],
        "height": summary["height"],
        "long_side": summary["long_side"],
        "sharpness_mean": round(summary["sharpness_mean"], 4),
        "sharpness_median": round(summary["sharpness_median"], 4),
        "sharpness_min": round(summary["sharpness_min"], 4),
        "sharpness_max": round(summary["sharpness_max"], 4),
        "brightness_mean": round(summary["brightness_mean"], 4),
        "blur_count": summary["blur_count"],
        "blur_ratio": round(summary["blur_ratio"], 4),
        "warnings": warnings,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(json.dumps(payload, indent=2))
    print(f"\nSummary saved to {output_path}")

    if args.contact_sheet:
        sheet_path = Path(args.contact_sheet)
        write_contact_sheet(summary, sheet_path, args.columns, args.thumb_width)
        print(f"Contact sheet saved to {sheet_path}")


def generate_report(args):
    source_dir = Path(args.source)
    original_summary = inspect_dataset_dir(source_dir, args.blur_warning_threshold)

    training_source_dir = Path(args.training_source) if args.training_source else source_dir
    training_summary = inspect_dataset_dir(training_source_dir, args.blur_warning_threshold)
    warnings = build_quality_warnings(training_summary, args.blur_warning_threshold)

    original_contact_sheet = None
    if args.contact_sheet:
        original_contact_sheet = Path(args.contact_sheet)
        write_contact_sheet(original_summary, original_contact_sheet, args.columns, args.thumb_width)

    curated_contact_sheet = None
    if args.curated_contact_sheet:
        curated_contact_sheet = Path(args.curated_contact_sheet)
        write_contact_sheet(training_summary, curated_contact_sheet, args.columns, args.thumb_width)

    metrics_7000 = parse_training_metrics(args.train_log_7000, 7000)
    metrics_30000 = parse_training_metrics(args.train_log_30000, 30000)
    test_psnr_7000 = metrics_7000.get("test_psnr")
    test_psnr_30000 = metrics_30000.get("test_psnr")

    registered_cameras = args.registered_cameras
    if registered_cameras is None and args.model_dir:
        registered_cameras = read_registered_camera_count(Path(args.model_dir) / "cameras.json")

    sparse_points = read_ply_vertex_count(args.sparse_ply) if args.sparse_ply else None
    selection_manifest = load_json(args.selection_manifest) if args.selection_manifest else None

    report_path = Path(args.output)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    own_scene_note = build_self_scene_table_note(training_summary, registered_cameras, args.blur_warning_threshold)
    commentary = build_self_scene_commentary(
        original_summary,
        training_summary,
        registered_cameras,
        test_psnr_7000,
        test_psnr_30000,
    )

    sparse_parts = []
    if registered_cameras is not None:
        sparse_parts.append(f"`{registered_cameras}` registrovanych kamier")
    if sparse_points is not None:
        sparse_parts.append(f"`{sparse_points}` 3D bodov")
    if args.reprojection_error is not None:
        sparse_parts.append(f"priemerna reprojekcna chyba `{args.reprojection_error:.2f} px`")

    report_lines = [
        "# Zadanie 4 - 3D Gaussian Splatting",
        "",
        "## 1. Referencny dataset",
        f"- Nazov datasetu: {args.reference_name}",
        f"- Pocet snimok: {format_optional_int(args.reference_image_count)}",
        f"- PSNR po 7 000 iteraciach: {format_psnr(args.reference_psnr_7000)}",
        f"- PSNR po 30 000 iteraciach: {format_psnr(args.reference_psnr_30000)}",
        f"- Screenshoty: {args.reference_screenshots}",
        "",
        "## 2. Vlastna scena",
        f"- Popis sceny: {args.scene_description}",
        f"- Povodny dataset: `{source_dir}` (`{original_summary['count']}` snimok, typicke rozlisenie "
        f"`{original_summary['width']} x {original_summary['height']}`)",
        f"- Dataset pre COLMAP a trening: `{training_source_dir}` (`{training_summary['count']}` snimok)",
        f"- Priemerna ostrost datasetu pre trening: `{training_summary['sharpness_mean']:.2f}`",
        f"- Priemerna jasova uroven datasetu pre trening: `{training_summary['brightness_mean']:.2f}`",
        (
            "- Sparse rekonstrukcia COLMAP: " + ", ".join(sparse_parts)
            if sparse_parts
            else "- Sparse rekonstrukcia COLMAP: DOPLN"
        ),
        f"- PSNR po 7 000 iteraciach: `{format_psnr(test_psnr_7000)}`",
        f"- PSNR po 30 000 iteraciach: `{format_psnr(test_psnr_30000)}`",
        (
            f"- Treningovy vystup: `{Path(args.model_dir)}`"
            if args.model_dir
            else "- Treningovy vystup: DOPLN"
        ),
        (
            f"- Test rendery modelu: `{Path(args.model_dir) / 'test' / 'ours_30000'}`"
            if args.model_dir
            else "- Test rendery modelu: DOPLN"
        ),
        f"- Screenshoty: {args.self_screenshots}",
        "",
        "### Poznamky k datasetu",
    ]

    for warning in warnings:
        report_lines.append(f"- {warning}")

    if selection_manifest:
        report_lines.append(
            f"- Kuracia vybrala `{selection_manifest['selected_count']}` zaberov z povodnych "
            f"`{original_summary['count']}` fotografii."
        )

    if original_contact_sheet or curated_contact_sheet or args.comparison_image:
        report_lines.extend(
            [
                "",
                "### Kontaktne tabule a porovnania",
            ]
        )

    if original_contact_sheet:
        report_lines.append(f"- Kontaktna tabula povodneho datasetu: `{original_contact_sheet}`")
    if curated_contact_sheet:
        report_lines.append(f"- Kontaktna tabula datasetu pre trening: `{curated_contact_sheet}`")
    if args.comparison_image:
        report_lines.append(f"- Vizualne porovnanie `GT vs render`: `{Path(args.comparison_image)}`")

    report_lines.extend(
        [
            "",
            "## 3. Porovnanie vysledkov",
            "| Scena | PSNR 7 000 | PSNR 30 000 | Poznamka |",
            "| --- | --- | --- | --- |",
            f"| Referencny dataset | {format_psnr(args.reference_psnr_7000)} | "
            f"{format_psnr(args.reference_psnr_30000)} | {args.reference_note} |",
            f"| Vlastna scena | {format_psnr(test_psnr_7000)} | {format_psnr(test_psnr_30000)} | {own_scene_note} |",
            "",
            "### Vizualne porovnanie",
            f"- Referencny dataset: {args.reference_visual_note}",
            (
                f"- Vlastna scena: sumarne porovnanie piatich testovacich pohladov je v "
                f"`{Path(args.comparison_image)}`."
                if args.comparison_image
                else "- Vlastna scena: DOPLN rendery z rovnakych uhlov pohladu."
            ),
            "",
            "### Komentar",
        ]
    )

    for comment in commentary:
        report_lines.append(f"- {comment}")

    report_lines.extend(
        [
            "",
            "## 4. Zaver",
            "- Pre dalsi pokus by bolo vhodne pouzit kratsi expozicny cas, pevne zaostrenie a pomalsi pohyb kamery.",
            "- Pomohlo by aj texturovanejsie pozadie alebo objekt s vyraznejsou strukturou, aby COLMAP nasiel viac spolahlivych bodov.",
            "- Referencny dataset este treba doplnit, aby bolo mozne uzavriet tabulku PSNR a obrazove porovnanie oboch scen.",
        ]
    )

    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(f"Report saved to {report_path}")


def reset_workspace(args):
    targets = [
        Path("data/vlastna_scena_curated"),
        Path("data/vlastna_scena_v2"),
        Path("output/c0285fd5-6"),
        Path("output/interactive_viewer"),
        Path("output/vlastna_scena_contact_sheet.jpg"),
        Path("output/vlastna_scena_curated_contact_sheet.jpg"),
        Path("output/vlastna_scena_curated_summary.json"),
        Path("output/vlastna_scena_curated_train30000.log"),
        Path("output/vlastna_scena_curated_train7000.log"),
        Path("output/vlastna_scena_summary.json"),
        Path("output/vlastna_scena_test_comparison.jpg"),
        Path("output/zadanie4_report.md"),
        Path("gsplat_viewer"),
        Path("make_pointcloud_viewer.py"),
    ]

    for target in targets:
        if not target.exists():
            continue

        if target.is_dir():
            shutil.rmtree(target)
            print(f"Removed directory: {target}")
        else:
            target.unlink()
            print(f"Removed file: {target}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Zadanie 4: snimanie a priprava fotografii pre 3D Gaussian Splatting."
    )
    subparsers = parser.add_subparsers(dest="command")

    capture = subparsers.add_parser("capture", help="Snimanie vlastnej sceny kamerou Ximea.")
    capture.add_argument("--output", default="data/vlastna_scena/input")
    capture.add_argument("--max-side", type=int, default=1200)
    capture.add_argument("--preview-size", type=int, default=900)
    capture.add_argument("--quality", type=int, default=95)
    capture.add_argument("--exposure", type=int, default=30000)
    capture.add_argument("--auto-interval", type=float, default=1.0)
    capture.add_argument("--min-sharpness", type=float, default=0.0)
    capture.set_defaults(func=capture_dataset)

    downscale = subparsers.add_parser("downscale", help="Downscale existujucich fotografii.")
    downscale.add_argument("--source", required=True)
    downscale.add_argument("--output", default="data/vlastna_scena/input")
    downscale.add_argument("--max-side", type=int, default=1200)
    downscale.add_argument("--quality", type=int, default=95)
    downscale.set_defaults(func=downscale_dataset)

    curate = subparsers.add_parser(
        "curate",
        help="Vyberie casovo rozlozene a najostrejsie snimky do noveho datasetu.",
    )
    curate.add_argument("--source", default="data/vlastna_scena/input")
    curate.add_argument("--output", default="data/vlastna_scena_curated/input")
    curate.add_argument("--target-count", type=int, default=80)
    curate.add_argument("--min-sharpness", type=float, default=4.0)
    curate.add_argument("--blur-warning-threshold", type=float, default=8.0)
    curate.set_defaults(func=curate_dataset)

    inspect = subparsers.add_parser("inspect", help="Skontroluje dataset a vypise technicke statistiky.")
    inspect.add_argument("--source", default="data/vlastna_scena/input")
    inspect.add_argument("--output", default="output/vlastna_scena_summary.json")
    inspect.add_argument("--blur-warning-threshold", type=float, default=8.0)
    inspect.add_argument("--contact-sheet", default="output/vlastna_scena_contact_sheet.jpg")
    inspect.add_argument("--columns", type=int, default=4)
    inspect.add_argument("--thumb-width", type=int, default=260)
    inspect.set_defaults(func=inspect_dataset)

    report = subparsers.add_parser("report", help="Vygeneruje Markdown report pre odovzdanie.")
    report.add_argument("--source", default="data/vlastna_scena/input")
    report.add_argument("--training-source", default="data/vlastna_scena_curated/input")
    report.add_argument("--output", default="output/zadanie4_report.md")
    report.add_argument("--blur-warning-threshold", type=float, default=8.0)
    report.add_argument("--contact-sheet", default="output/vlastna_scena_contact_sheet.jpg")
    report.add_argument("--curated-contact-sheet", default="output/vlastna_scena_curated_contact_sheet.jpg")
    report.add_argument("--columns", type=int, default=4)
    report.add_argument("--thumb-width", type=int, default=260)
    report.add_argument(
        "--scene-description",
        default="stolovy objekt `VELO` na svetlej doske, snimany kamerou Ximea okolo objektu z viacerych uhlov",
    )
    report.add_argument("--selection-manifest", default="data/vlastna_scena_curated/selection_manifest.json")
    report.add_argument("--train-log-7000", default="output/vlastna_scena_curated_train7000.log")
    report.add_argument("--train-log-30000", default="output/vlastna_scena_curated_train30000.log")
    report.add_argument("--model-dir", default="output/c0285fd5-6")
    report.add_argument("--sparse-ply", default="data/vlastna_scena_curated/sparse/0/points3D.ply")
    report.add_argument("--comparison-image", default="output/vlastna_scena_test_comparison.jpg")
    report.add_argument("--registered-cameras", type=int)
    report.add_argument("--reprojection-error", type=float)
    report.add_argument("--reference-name", default="DOPLN")
    report.add_argument("--reference-image-count", type=int)
    report.add_argument("--reference-psnr-7000", type=float)
    report.add_argument("--reference-psnr-30000", type=float)
    report.add_argument("--reference-note", default="DOPLN")
    report.add_argument("--reference-visual-note", default="DOPLN rendery z rovnakych uhlov pohladu.")
    report.add_argument("--reference-screenshots", default="DOPLN (`COLMAP GUI`, trening, viewer)")
    report.add_argument("--self-screenshots", default="dopln rucne (`COLMAP GUI`, trening, viewer)")
    report.set_defaults(func=generate_report)

    reset = subparsers.add_parser("reset-workspace", help="Zmaze stare generovane artefakty pre novu pipeline.")
    reset.set_defaults(func=reset_workspace)

    args = parser.parse_args()

    if args.command is None:
        args = parser.parse_args(["capture"])

    return args


def main():
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
