#!/usr/bin/env python3
"""The prompts that turn a plain VQA pool into a dual-world (CWAD) dataset.

``build_dual_world_data.py`` only assembles images that already exist: it pairs
each original with the counterfactual edit of the same scene. What decides
whether a pair is a real dual-world pair -- same question, answer flips from the
ground truth to exactly one distractor, everything else pixel-familiar -- is the
text below. Anyone reproducing the dataset needs these, so they live next to the
code that reads their output rather than in a paper appendix.

Stage order, and what each stage consumes from the previous one:

  1. EDIT_INSTRUCTION_PROMPT  (image + question + options + GT)
                            -> {"chosen_option", "chosen_text", "edit_scope",
                                "edit_instruction", "editability"}
     Rows whose answer is not among the options, and rows where the planner
     returns ``chosen_option = null``, are dropped: no edit is requested for
     them, so they never reach the builder.
  2. The instruction stage 1 emits is then handed to the image-editing API
     (Qwen-Image-3.0), which returns a square 1024x1024 counterfactual. The
     paper reproduces the stage-1 prompt box at this stage as well: the planning
     prompt and the editing request are one and the same instruction stream, and
     ``edit_instruction`` has to carry the keep-unchanged list by itself.
  3. QC_FAITHFULNESS_PROMPT (original + edited + instruction + scope) -> score
     A pair is kept only if the answer flips from the ground truth to the chosen
     option and the question plus the non-critical scene context survive.
     Rejected pairs go back to stage 1 for a rewritten instruction.
     QC_APPLIED_PROMPT / QC_ANSWER_PROMPT are the two binary probes this tree's
     QC step uses to make that judgement: did the edit land, and which option is
     true in the edited image.
  4. EDIT_QUESTION_TYPE_PROMPT labels the surviving pairs; the builder does not
     read these labels, they are for reporting.
  5. scripts/build_dual_world_data.py -- the parquet the trainer consumes.

Run this file to print every template with its placeholders.
"""
from __future__ import annotations

import json
import re
import string
from typing import Any


# Stage 1 -- planner. Sent with the original image. max_tokens=1024,
# temperature=0.4; the temperature is what lets the planner reject honestly
# instead of forcing an edit for every question.
EDIT_INSTRUCTION_PROMPT = """You are preparing a counterfactual image for a visual-editing API (Qwen-Image-3.0).
The image shown is the ORIGINAL. Below is a 4-option VQA question about it and its ground truth (GT).

Question: {question}
Options: {options}
GT: {gt}. {gt_text}

Task: Pick ONE distractor option (not the GT) that can become TRUE in the image via a SINGLE LOCAL edit,
and write the English editing instruction for it.

Hard criteria for the chosen option:
1. The edit touches ONE object or ONE attribute only; everything else must stay pixel-familiar.
2. After the edit the chosen option is UNAMBIGUOUSLY true and the GT UNAMBIGUOUSLY false for this
   question. Reject any option that would leave two plausible answers.
3. Reject options that require:
   - Tiny or invisible details that cannot be clearly seen in the image.
   - A different camera viewpoint or major scene restructuring.
4. The instruction must (a) name what is CURRENTLY in the image and where it is, (b) say what to
   change it into, keeping the same pose/size/position, (c) pin the material and style to the original,
   (d) explicitly list what to keep unchanged (other parts, background, framing, lighting).

NOTE: Text editing (signs, labels, logos, jerseys, scoreboards, banners, license plates, book covers, etc.)
is NOW ALLOWED. Qwen-Image-3.0 can handle text modifications well.

Example (different image, for format only):
{few_shot}

Answer with ONE JSON object and nothing else:
{{"chosen_option": "<letter or null>", "chosen_text": "<option text>", "edit_scope": "<the object/attribute touched>", "edit_instruction": "<English instruction, 2-4 sentences>", "editability": <1-5>}}"""


# One example is enough to pin the format, and it must be a scene the planner has
# not been shown, so the few-shot is always from a different image.
EDIT_INSTRUCTION_FEW_SHOT = {
    "question": "Which design is present on each arm among the following choices: cat, eagles, dog, or bat? Options: A. cat B. eagles C. dog D. bat",
    "ground_truth": "B",
    "chosen_option": "A",
    "chosen_text": "cat",
    "edit_instruction": "Replace the two cast-iron armrests of the bench, which are currently shaped like eagles with spread wings, with armrests shaped like cats in the same pose. Keep the cats as black cast iron with the same material, size, and position as the original eagles. Keep everything else exactly the same: the red wooden slats of the seat and backrest, the surrounding green plants, the camera angle, framing, and lighting.",
    "edit_scope": "armrests",
}


# Stage 2 -- the editing request is the instruction stage 1 produced, applied to
# the original image; the paper reprints EDIT_INSTRUCTION_PROMPT here. World B is
# returned as a square image, which is why the world-B prompt budget in
# config/best.env is the larger one.


# Stage 3 -- quality control. Scored continuously so a threshold can be moved
# without re-running the judge; the pass/fail list it produces is what
# build_dual_world_data.py --good-edit-ids-file consumes.
QC_FAITHFULNESS_PROMPT = """You are given TWO images of the same scene. The FIRST image is the ORIGINAL. \
The SECOND image is the EDITED version produced by an image-editing model.

Editing instruction given to the model: "{edit_instruction}"

What the edit targets: "{edit_scope}"

Score how faithfully the SECOND image executes the instruction, on a continuous scale from 0.0 to 1.0:
- 1.0: the described change is fully and unambiguously realized in the edited image, AND the rest \
of the image (background, other objects, composition, lighting) stays consistent with the original.
- 0.75: the target change is clearly realized with minor imperfections; rest well preserved.
- 0.5: the edit attempted the change but result is ambiguous, partial, or the wrong object changed.
- 0.25: barely related to the instruction, or large unintended changes elsewhere.
- 0.0: no change at all, or the image is corrupted/unrelated.

Answer with ONE JSON object and nothing else:
{{"score": <float 0.0-1.0>, "reason": "<one short sentence>"}}"""


# The binary form of the same check, asked at temperature 0.0.
QC_APPLIED_PROMPT = (
    "You are given TWO images of the same scene. The FIRST image is the ORIGINAL. "
    "The SECOND image is the EDITED version.\n\n"
    'Editing instruction: "{edit_instruction}"\n\n'
    'What the edit targets: "{edit_scope}"\n\n'
    "Compare the two images. Has the edit described above been successfully applied "
    "in the SECOND (edited) image compared to the FIRST (original)? That is, does the "
    "edited image show the changed object/attribute as described in the instruction "
    "(e.g. the new color, shape, text, or object is now present, or the removed object "
    "is now gone), while the rest of the image stays consistent with the original?\n\n"
    "Answer with ONLY one word: yes or no. No explanation."
)


# Asked when the yes/no check fails: the answer may have flipped to a different
# distractor than the one the instruction targeted, which is still an unusable
# pair, so the letter is checked rather than assumed.
QC_ANSWER_PROMPT = (
    "Look at this image and answer the question by choosing exactly ONE option.\n\n"
    "Question: {question}\n"
    "Options: {options}\n\n"
    "Answer with ONLY the single letter (A, B, C, or D) of the correct option. "
    "Output just the letter, no explanation."
)


# Stage 4 -- pure text classification of the verified pairs, no image is sent.
EDIT_QUESTION_TYPE_PROMPT = """Classify the following image editing task and question into the categories below.

**Edit instruction:** {edit_instruction}
**Edit scope:** {edit_scope}
**Chosen answer:** {chosen_text}
**Question:** {question}

Output JSON with exactly these fields:
- "edit_type": one of ["count", "color", "shape", "presence/absence", "spatial relation", "common-sense/knowledge"]
- "question_type": one of ["perception", "counting", "spatial reasoning/knowledge", "others"]

Edit type definitions:
- count: editing the number/quantity of objects (e.g., "change 3 dogs to 5 dogs")
- color: editing the color of something (e.g., "change red to blue")
- shape: editing the shape/form/design of something (e.g., "change eagle shape to bat shape")
- presence/absence: adding or removing an object entirely (e.g., "add a cat", "remove the person")
- spatial relation: editing position/orientation/posture (e.g., "change lying to standing")
- common-sense/knowledge: editing based on world knowledge (e.g., "change to African setting")

Question type definitions:
- perception: asking about visual attributes (color, shape, texture, design)
- counting: asking about number/quantity
- spatial reasoning/knowledge: asking about position, action, or requiring world knowledge
- others: anything else

Output only the JSON, no explanation."""


def parse_mcq(content: str) -> tuple[str | None, dict[str, str] | None]:
    """Recover the question and its options from a dataset row's prompt text.

    The rows the builder consumes store the question as one user message with an
    ``<image>`` line and the options written inline (``A. cat B. eagles ...``),
    which is also the shape scripts/verify.py checks for. This splits it back into
    the fields stage 1 asks for.
    """
    body = content.split("\n", 1)[1]
    body = body.split("Answer with the option")[0].strip()
    labels = list(re.finditer(r"(?:^|(?<=\s))([A-E])\.\s+", body))
    if len(labels) < 2:
        return None, None
    question = body[: labels[0].start()].strip()
    options: dict[str, str] = {}
    for i, match in enumerate(labels):
        end = labels[i + 1].start() if i + 1 < len(labels) else len(body)
        options[match.group(1)] = body[match.end():end].strip().rstrip(",;").strip()
    return question, options


def format_options(options: dict[str, str]) -> str:
    return "  ".join(f"{letter}. {text}" for letter, text in options.items())


def edit_instruction_prompt(question: str, options: dict[str, str], gt: str) -> str:
    gt = gt.strip().upper()
    if gt not in options:
        raise ValueError(f"ground truth {gt!r} is not one of the options {sorted(options)}")
    return EDIT_INSTRUCTION_PROMPT.format(
        question=question,
        options=format_options(options),
        gt=gt,
        gt_text=f"correct option text: {options[gt]}",
        few_shot=json.dumps(EDIT_INSTRUCTION_FEW_SHOT, ensure_ascii=False, indent=1),
    )


def qc_faithfulness_prompt(edit_instruction: str, edit_scope: str) -> str:
    return QC_FAITHFULNESS_PROMPT.format(
        edit_instruction=edit_instruction, edit_scope=edit_scope
    )


def qc_applied_prompt(edit_instruction: str, edit_scope: str) -> str:
    return QC_APPLIED_PROMPT.format(
        edit_instruction=edit_instruction, edit_scope=edit_scope
    )


def qc_answer_prompt(question: str, options: dict[str, str]) -> str:
    return QC_ANSWER_PROMPT.format(question=question, options=format_options(options))


def edit_question_type_prompt(
    edit_instruction: str, edit_scope: str, chosen_text: str, question: str
) -> str:
    return EDIT_QUESTION_TYPE_PROMPT.format(
        edit_instruction=edit_instruction,
        edit_scope=edit_scope,
        chosen_text=chosen_text,
        question=question,
    )


def extract_json(raw: str) -> dict[str, Any] | None:
    """Read the one JSON object a stage asked for, or None.

    A None from stage 1 is a legitimate answer -- it means the planner found no
    distractor a single local edit could make true, and the row is dropped.
    """
    start, end = raw.find("{"), raw.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        return json.loads(raw[start:end + 1])
    except json.JSONDecodeError:
        return None


TEMPLATES = [
    ("stage 1  edit instruction generation", "EDIT_INSTRUCTION_PROMPT", EDIT_INSTRUCTION_PROMPT),
    ("stage 1  few-shot example", "EDIT_INSTRUCTION_FEW_SHOT", json.dumps(EDIT_INSTRUCTION_FEW_SHOT, indent=1)),
    ("stage 2  image editing request", "EDIT_INSTRUCTION_PROMPT", EDIT_INSTRUCTION_PROMPT),
    ("stage 3  quality control: faithfulness score", "QC_FAITHFULNESS_PROMPT", QC_FAITHFULNESS_PROMPT),
    ("stage 3  quality control: applied yes/no", "QC_APPLIED_PROMPT", QC_APPLIED_PROMPT),
    ("stage 3  quality control: answer letter", "QC_ANSWER_PROMPT", QC_ANSWER_PROMPT),
    ("stage 4  edit / question type labels", "EDIT_QUESTION_TYPE_PROMPT", EDIT_QUESTION_TYPE_PROMPT),
]


def main() -> None:
    for stage, name, text in TEMPLATES:
        print(f"\n{'=' * 72}\n{name}  --  {stage}\n{'=' * 72}")
        print(text)
        placeholders = sorted(
            {
                field
                for _, field, _, _ in string.Formatter().parse(text)
                if field
            }
        )
        if placeholders:
            print(f"\n[placeholders] {', '.join(placeholders)}")


if __name__ == "__main__":
    main()
