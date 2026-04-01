import os
import shutil
import random

random.seed(42)

# ── Split GOEC-4 into train/valid/test ────────────────────────────────────────
goec_images = os.listdir("GOEC-4/train/images")
goec_labels = os.listdir("GOEC-4/train/labels")

random.shuffle(goec_images)

total = len(goec_images)
train_end = int(0.7 * total)
valid_end = int(0.9 * total)

goec_splits = {
    "train": goec_images[:train_end],
    "valid": goec_images[train_end:valid_end],
    "test":  goec_images[valid_end:]
}

# ── Balance limit per split ───────────────────────────────────────────────────
limits = {
    "train": min(len(goec_splits["train"]), 7500, 8468),
    "valid": min(len(goec_splits["valid"]), 1500, 2314),
    "test":  min(len(goec_splits["test"]),  1000, 1149)
}

# ── Create output folders ─────────────────────────────────────────────────────
for split in ["train", "valid", "test"]:
    os.makedirs(f"combined/{split}/images", exist_ok=True)
    os.makedirs(f"combined/{split}/labels", exist_ok=True)

# ── Copy function ─────────────────────────────────────────────────────────────
def copy_files(dataset, split, filenames, limit, prefix):
    filenames = random.sample(filenames, min(limit, len(filenames)))
    for fname in filenames:
        name = os.path.splitext(fname)[0]
        src_img = f"{dataset}/{split}/images/{fname}"
        src_lbl = f"{dataset}/{split}/labels/{name}.txt"
        dst_img = f"combined/{split}/images/{prefix}_{fname}"
        dst_lbl = f"combined/{split}/labels/{prefix}_{name}.txt"
        if os.path.exists(src_img):
            shutil.copy(src_img, dst_img)
        if os.path.exists(src_lbl):
            shutil.copy(src_lbl, dst_lbl)

# ── Copy Cars ─────────────────────────────────────────────────────────────────
for split in ["train", "valid", "test"]:
    imgs = os.listdir(f"car-detection-1/{split}/images")
    copy_files("car-detection-1", split, imgs, limits[split], "car")

# ── Copy Fire and Smoke ───────────────────────────────────────────────────────
for split in ["train", "valid", "test"]:
    imgs = os.listdir(f"Fire-and-Smoke-7/{split}/images")
    copy_files("Fire-and-Smoke-7", split, imgs, limits[split], "fire")

# ── Copy GOEC-4 ───────────────────────────────────────────────────────────────
for split in ["train", "valid", "test"]:
    imgs = goec_splits[split]
    for fname in random.sample(imgs, min(limits[split], len(imgs))):
        name = os.path.splitext(fname)[0]
        src_img = f"GOEC-4/train/images/{fname}"
        src_lbl = f"GOEC-4/train/labels/{name}.txt"
        dst_img = f"combined/{split}/images/goec_{fname}"
        dst_lbl = f"combined/{split}/labels/goec_{name}.txt"
        if os.path.exists(src_img):
            shutil.copy(src_img, dst_img)
        if os.path.exists(src_lbl):
            shutil.copy(src_lbl, dst_lbl)

print("Done! Combined dataset created.")
for split in ["train", "valid", "test"]:
    count = len(os.listdir(f"combined/{split}/images"))
    print(f"{split}: {count} images")
