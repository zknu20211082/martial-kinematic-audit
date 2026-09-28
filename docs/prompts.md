# Prompts, answer schema and parsing

Everything on this page is produced by the code in `mkaudit/vlm.py` (prompt template, frame sampling, JSON parsing),
`scripts/30_vlm_audit.py` (generation loop) and `mkaudit/verifier.py` (scoring). The prompt texts below were printed
with `mkaudit.vlm.prompt_text`; they are reproduced verbatim.

## Generation setup

- Model: `Qwen/Qwen2.5-VL-7B-Instruct` (Hub revision `cc594898137f460bfe9f0759e9844b3ce807cfb5`), bfloat16, one GPU.
- Input video: 8 frames sampled uniformly over the clip (`np.linspace(0, n_frames - 1, 8)`), passed to the processor as
  one video.
- Conversation: one user turn whose content is `[{"type": "video"}, {"type": "text", "text": <prompt>}]`, formatted with
  the model's chat template (`add_generation_prompt=True`); the template inserts its default system message
  "You are a helpful assistant.".
- Decoding: greedy (`do_sample=False`), `max_new_tokens=220`; all other generation settings are the defaults of the
  model's `generation_config.json` (it sets `repetition_penalty=1.05`).
- Conditions: `zeroshot` (prompt without evidence) and `grounded` (the same prompt with a block of kinematic evidence
  inserted before the last line). The evidence prompt is only issued when evidence exists; in each in-the-wild set one
  clip had no stably tracked performer, hence 202/255 evidence generations against 203/256 zero-shot generations.

## Zero-shot prompt

Example: `data/synthetic_benchmark/videos/synth_0000_karate.mp4` (ground truth: Gyaku-Zuki, right arm, no kick).

```text
You are analysing a short martial-arts video. Answer ONLY with a JSON object with these keys:
"technique": one of ['Gyaku-Zuki', 'Mae-Geri', 'Mawashi-Geri-gedan', 'Mawashi-Geri-jodan', 'Ushiro-Mawashi-Geri'];
"n_people": integer number of people visible;
"striking_limb": the limb that moves fastest / delivers the main strike, one of ["left_arm","right_arm","left_leg","right_leg","none"] (the performer's own left/right);
"peak_foot_height": highest point reached by a foot, one of ["no_kick","below_hip","hip_to_shoulder","above_shoulder"];
"body_turn": "yes" if the performer rotates the body by 90 degrees or more, else "no";
"knee_extended_at_peak": "yes" if the striking leg's knee is nearly straight (>150 degrees) at the peak of the strike, "no" otherwise, "n/a" if the strike is with the arm;
"explanation": one sentence citing the visible body motion that justifies the technique label.
Return the JSON only.
```

## Evidence prompt

Same video. The evidence block is inserted between the field list and the last line:

```text
You are analysing a short martial-arts video. Answer ONLY with a JSON object with these keys:
"technique": one of ['Gyaku-Zuki', 'Mae-Geri', 'Mawashi-Geri-gedan', 'Mawashi-Geri-jodan', 'Ushiro-Mawashi-Geri'];
"n_people": integer number of people visible;
"striking_limb": the limb that moves fastest / delivers the main strike, one of ["left_arm","right_arm","left_leg","right_leg","none"] (the performer's own left/right);
"peak_foot_height": highest point reached by a foot, one of ["no_kick","below_hip","hip_to_shoulder","above_shoulder"];
"body_turn": "yes" if the performer rotates the body by 90 degrees or more, else "no";
"knee_extended_at_peak": "yes" if the striking leg's knee is nearly straight (>150 degrees) at the peak of the strike, "no" otherwise, "n/a" if the strike is with the arm;
"explanation": one sentence citing the visible body motion that justifies the technique label.
Kinematic evidence measured from the performer's tracked body joints (trust it over your visual impression):
- number of tracked people: 1
- fastest-moving limb (performer's own side): right_arm
- highest foot position: no kick
- peak limb speed (body-heights per second): 3.2
Return the JSON only.
```

### Evidence for the other test sets

Synthetic videos take the evidence from the ground-truth table (`synth_videos.csv`): number of tracked people is
always 1, the fastest limb is `striking_limb`, the foot position is `peak_foot_height` with underscores replaced by
spaces, and the speed is `peak_speed` with one decimal. Body turn and knee extension are never given as evidence.

In-the-wild clips take it from the cached RTMPose keypoints (`mkaudit.verifier.kcv_from_pose`); the same values serve
as pseudo ground truth. Example (HMDB51 `A_Beautiful_Mind_4_punch_h_nm_np1_ba_med_11.avi`, class `punch`):

```text
- number of tracked people: 2+
- fastest-moving limb (performer's own side): left_arm
- highest foot position: below hip or no kick
- peak limb speeds (body-heights per second): left_arm=6.8, right_arm=5.3, left_leg=1.9, right_leg=2.1
```

The people count is `1` or `2+` (at most two persons are tracked), the fastest limb (95th percentile of the
normalised keypoint speed) is `unverifiable` when the second-fastest limb is within 15% of it, the foot position is
one of `above shoulder`, `hip to shoulder`, `below hip or no kick` or `unverifiable`, and the speed line is omitted
when the limb is unverifiable.

### Technique lists

| Test set | List inserted after `"technique": one of` |
|---|---|
| synthetic karate | `['Gyaku-Zuki', 'Mae-Geri', 'Mawashi-Geri-gedan', 'Mawashi-Geri-jodan', 'Ushiro-Mawashi-Geri']` |
| synthetic Taichi | `['Beginning position (Wuji)', 'Bring sky and earth together', 'Brush knee and twist step', 'Canalize energy', 'Drive the monkey away', 'Fair lady works shuttles', 'Golden rooster stands on one leg', "Grasp the bird's tail", 'Kick with heel', 'Move hands like clouds', 'Open and close lotus flower', "Part the wild horse's mane", 'Tree posture (Taiji)']` |
| HMDB51-MA | `['punch', 'kick', 'hit', 'sword', 'sword_exercise', 'draw_sword', 'fencing']` |
| UCF101-MA | `['Punch', 'BoxingPunchingBag', 'BoxingSpeedBag', 'Fencing', 'TaiChi', 'SumoWrestling', 'Nunchucks']` |

## Answer schema

```json
{
  "technique": "<one entry of the technique list>",
  "n_people": 1,
  "striking_limb": "left_arm | right_arm | left_leg | right_leg | none",
  "peak_foot_height": "no_kick | below_hip | hip_to_shoulder | above_shoulder",
  "body_turn": "yes | no",
  "knee_extended_at_peak": "yes | no | n/a",
  "explanation": "<one sentence>"
}
```

Stored answers for the example video (`results/e3_vlm_synth.csv`, column `raw`, newlines replaced by spaces):

````text
zeroshot: ```json {   "technique": "Mawashi-Geri-gedan",   "n_people": 1,   "striking_limb": "right_leg",   "peak_foot_height": "above_shoulder",   "body_turn": "yes",   "knee_extended_at_peak": "yes",   "explanation": "The performer executes a spinning back kick with the right leg extended and rotated, reaching above shoulder height." } ```
grounded: ```json {   "technique": "Gyaku-Zuki",   "n_people": 1,   "striking_limb": "right_arm",   "peak_foot_height": "no_kick",   "body_turn": "no",   "knee_extended_at_peak": "n/a",   "explanation": "The performer's right arm extends forward in a punching motion, indicating a Gyaku-Zuki." } ```
````

## Parsing and scoring

1. At generation time (`mkaudit.vlm.parse_json`) the span from the first `{` to the last `}` is parsed with
   `json.loads`; if that fails, single quotes are replaced by double quotes and trailing commas before `}` are removed
   and parsing is retried; if it still fails the answer is treated as empty.
2. For the paper, every field is re-extracted directly from the stored raw answer (`mkaudit.verifier.recover`: a
   regular expression for `"<field>": "<string>"`, an integer, `null`, `true` or `false`), which does not depend on
   the rest of the JSON being well formed. A field that cannot be found becomes an empty answer, which counts as a
   failed claim whenever the ground truth is verifiable.
3. Each claim is compared with the ground truth (`mkaudit.verifier.match`) and marked `pass`, `fail` or
   `unverifiable` (no ground truth): `n_people` uses the first integer of the answer and a truth of `2+` accepts any
   integer of at least 2; `technique` is compared case-insensitively after removing spaces, hyphens and underscores; a
   pose-derived foot truth of `below_hip_or_no_kick` accepts `below_hip` and `no_kick`; the other fields require an
   exact match after lower-casing. For `knee_extended_at_peak` the truth is `n/a` when the true striking limb is an
   arm (as the prompt instructs). `striking_limb_type` (arm / leg) is scored in addition and is not part of KF.
4. KF = passes / (passes + fails) and HR = fails / all claims, pooled over the five kinematic fields (`n_people`,
   `striking_limb`, `peak_foot_height`, `body_turn`, `knee_extended_at_peak`) of all videos of a test set.

There were 0 parse failures in the 1316 generations: every answer contained a JSON object with a non-empty
`technique` field (`results/revision_checks.json`, `A5_counts`: `raw_without_json = 0` and
`parse_fail_technique_empty = 0` for every test set and condition).
