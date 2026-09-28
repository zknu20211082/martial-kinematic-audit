"""Video LLM under audit (Qwen2.5-VL-7B-Instruct, paper Sec. 1.5): frame sampling, the zero-shot / evidence prompt
template and JSON parsing of the answers."""
import json
import re

import numpy as np

from .config import HF_CACHE

QWEN_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
QWEN_REVISION_USED = "cc594898137f460bfe9f0759e9844b3ce807cfb5"  # Hub revision (snapshot) used for the paper
N_FRAMES = 8
HMDB_CLASSES = ["punch", "kick", "hit", "sword", "sword_exercise", "draw_sword", "fencing"]
UCF_CLASSES = ["Punch", "BoxingPunchingBag", "BoxingSpeedBag", "Fencing", "TaiChi", "SumoWrestling", "Nunchucks"]
KARATE_CLASSES = ["Gyaku-Zuki", "Mae-Geri", "Mawashi-Geri-gedan", "Mawashi-Geri-jodan", "Ushiro-Mawashi-Geri"]
FIELDS = ["technique", "n_people", "striking_limb", "peak_foot_height", "body_turn", "knee_extended_at_peak"]


def read_frames(path, n=N_FRAMES):
    import av
    fr = []
    with av.open(path) as c:
        for f in c.decode(video=0):
            fr.append(f.to_ndarray(format="rgb24"))
    idx = np.linspace(0, len(fr) - 1, n).round().astype(int)
    return np.stack([fr[i] for i in idx])


def prompt_text(classes, evidence=None):
    p = ("You are analysing a short martial-arts video. Answer ONLY with a JSON object with these keys:\n"
         f'"technique": one of {classes};\n'
         '"n_people": integer number of people visible;\n'
         '"striking_limb": the limb that moves fastest / delivers the main strike, one of ["left_arm","right_arm","left_leg","right_leg","none"] (the performer\'s own left/right);\n'
         '"peak_foot_height": highest point reached by a foot, one of ["no_kick","below_hip","hip_to_shoulder","above_shoulder"];\n'
         '"body_turn": "yes" if the performer rotates the body by 90 degrees or more, else "no";\n'
         '"knee_extended_at_peak": "yes" if the striking leg\'s knee is nearly straight (>150 degrees) at the peak of the strike, "no" otherwise, "n/a" if the strike is with the arm;\n'
         '"explanation": one sentence citing the visible body motion that justifies the technique label.\n')
    if evidence:
        p += ("Kinematic evidence measured from the performer's tracked body joints (trust it over your visual impression):\n"
              + "\n".join(f"- {k}: {v}" for k, v in evidence.items()) + "\n")
    p += "Return the JSON only."
    return p


def parse_json(s):
    m = re.search(r"\{.*\}", s, re.S)
    if not m:
        return {}
    txt = m.group(0)
    try:
        return json.loads(txt)
    except Exception:
        try:
            return json.loads(re.sub(r",\s*}", "}", txt.replace("'", '"')))
        except Exception:
            return {}


def load_qwen(model_id=QWEN_ID, cache_dir=None, revision=None):
    """Load Qwen2.5-VL (bf16, on the GPU) and its processor. ``model_id`` may also be a local snapshot directory."""
    import torch
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    cache_dir = str(cache_dir or HF_CACHE)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, dtype=torch.bfloat16, device_map="cuda",
                                                               cache_dir=cache_dir, revision=revision).eval()
    proc = AutoProcessor.from_pretrained(model_id, cache_dir=cache_dir, revision=revision)
    return model, proc
