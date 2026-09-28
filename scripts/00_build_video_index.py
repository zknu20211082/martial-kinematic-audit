"""Build the martial-arts video index: HMDB51-MA (official 3 splits) and UCF101-MA (official group splits).

Paper: Sec. 1.1 and Table 1 (HMDB51-MA / UCF101-MA subsets). Output: <work_dir>/video_index.csv, used by every
video step (03, 04, 20, 22, 30, 40)."""
import pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # make `mkaudit` importable from a clone

import glob, os, re
import pandas as pd
from mkaudit.config import DATA, DD, ensure_dirs

HMDB_MA = ["punch", "kick", "hit", "sword", "sword_exercise", "draw_sword", "fencing"]
UCF_MA = ["Punch", "BoxingPunchingBag", "BoxingSpeedBag", "Fencing", "TaiChi", "SumoWrestling", "Nunchucks"]
HROOT = DATA / "hmdb51" / "hmdb51_org"; HSPLIT = DATA / "hmdb51" / "test_train_splits" / "testTrainMulti_7030_splits"
UROOT = DATA / "UCF101" / "UCF-101"


def main():
    ensure_dirs(DD)
    rows = []
    for cls in HMDB_MA:
        split = {}
        for s in (1, 2, 3):
            for line in open(HSPLIT / f"{cls}_test_split{s}.txt", encoding="utf-8", errors="ignore"):
                parts = line.split()
                if len(parts) >= 2:
                    split.setdefault(parts[0], {})[s] = int(parts[1])
        for f in sorted(glob.glob(str(HROOT / cls / "*.avi"))):
            name = os.path.basename(f)
            sp = split.get(name, {})
            rows.append({"dataset": "HMDB51-MA", "cls": cls, "path": f, "clip": name,
                         "split1": sp.get(1, 0), "split2": sp.get(2, 0), "split3": sp.get(3, 0), "group": name.rsplit("_", 1)[0]})
    for cls in UCF_MA:
        for f in sorted(glob.glob(str(UROOT / cls / "*.avi"))):
            name = os.path.basename(f)
            g = int(re.search(r"_g(\d\d)_", name).group(1))
            # official UCF101 splits: test groups 1-7 / 8-14 / 15-21; 1=train 2=test
            rows.append({"dataset": "UCF101-MA", "cls": cls, "path": f, "clip": name,
                         "split1": 2 if 1 <= g <= 7 else 1, "split2": 2 if 8 <= g <= 14 else 1, "split3": 2 if 15 <= g <= 21 else 1,
                         "group": f"g{g:02d}"})
    df = pd.DataFrame(rows)
    df.to_csv(DD / "video_index.csv", index=False)
    print(df.groupby(["dataset", "cls"]).size())
    for s in (1, 2, 3):
        print(f"split{s}", df.groupby(["dataset", f"split{s}"]).size().to_dict())


if __name__ == "__main__":
    main()
