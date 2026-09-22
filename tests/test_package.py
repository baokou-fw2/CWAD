#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.dont_write_bytecode = True
GENERATED_TEST_BYTECODE = (
    Path(__cached__).resolve() if globals().get("__cached__") else None
)

# scripts/ carries the data-preparation tools; a tree that has not received them
# yet must still be checkable, so those cases are gated rather than deleted.
# Everything else in this suite runs unconditionally.
SCRIPTS_READY = (PACKAGE_ROOT / "scripts").is_dir()

# The only picture this repository ships: the method overview the README links.
ALLOWED_DOCUMENTATION_MEDIA = {"assets/CWAD_framework.png"}


def import_data_libraries():
    import pyarrow as pa
    import pyarrow.parquet as pq
    from PIL import Image

    return pa, pq, Image


def is_generated_test_cache(path: Path) -> bool:
    if GENERATED_TEST_BYTECODE is None:
        return False
    resolved = path.resolve()
    if resolved == GENERATED_TEST_BYTECODE:
        return True
    if path.is_dir() and resolved == GENERATED_TEST_BYTECODE.parent:
        return all(
            child.resolve() == GENERATED_TEST_BYTECODE for child in path.iterdir()
        )
    return False


def import_script(name: str):
    path = PACKAGE_ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class ReproductionPackageTests(unittest.TestCase):
    def test_public_package_contains_no_data_media_weights_or_archives(self):
        forbidden_endings = (
            ".arrow",
            ".avif",
            ".bin",
            ".bmp",
            ".ckpt",
            ".csv",
            ".gif",
            ".heic",
            ".jpeg",
            ".jpg",
            ".jsonl",
            ".npy",
            ".npz",
            ".parquet",
            ".pyc",
            ".pyo",
            ".pth",
            ".pt",
            ".png",
            ".safetensors",
            ".tif",
            ".tiff",
            ".webp",
            ".tar",
            ".tar.gz",
            ".tar.zst",
            ".tgz",
            ".zip",
            ".7z",
        )
        forbidden_magic = (
            b"\x89PNG\r\n\x1a\n",
            b"\xff\xd8\xff",
            b"GIF87a",
            b"GIF89a",
            b"BM",
            b"PK\x03\x04",
            b"PK\x05\x06",
            b"PK\x07\x08",
            b"\x1f\x8b",
            b"\x28\xb5\x2f\xfd",
            b"7z\xbc\xaf\x27\x1c",
            b"Rar!\x1a\x07",
        )
        violations = []
        for path in PACKAGE_ROOT.rglob("*"):
            if is_generated_test_cache(path):
                continue
            if path.is_dir() and path.name in {
                "__pycache__",
                ".mypy_cache",
                ".pytest_cache",
                ".ruff_cache",
            }:
                violations.append(str(path.relative_to(PACKAGE_ROOT)))
                continue
            if path.is_symlink():
                violations.append(str(path.relative_to(PACKAGE_ROOT)))
                continue
            if not path.is_file():
                continue
            relative = path.relative_to(PACKAGE_ROOT)
            if relative.as_posix() in ALLOWED_DOCUMENTATION_MEDIA:
                continue
            if relative.parts[0] in {".runtime", "outputs"}:
                continue
            name = path.name.lower()
            with path.open("rb") as handle:
                head = handle.read(512)
            if name.endswith(forbidden_endings):
                violations.append(str(relative))
            elif any(head.startswith(magic) for magic in forbidden_magic):
                violations.append(str(relative))
            elif head.startswith(b"RIFF") and head[8:12] == b"WEBP":
                violations.append(str(relative))
            elif len(head) >= 262 and head[257:262] == b"ustar":
                violations.append(str(relative))
        self.assertEqual(violations, [])

    def test_public_package_contains_no_internal_identity_or_credentials(self):
        forbidden_markers = [
            "/" + "mnt" + "/" + "bn" + "/",
            "/" + "home" + "/" + "tiger" + "/",
            "/" + "Users" + "/" + "byted" + "ance" + "/",
            "wyc" + "." + "wyc",
            "wang" + "yuchen",
            "hub." + "byted" + ".org",
            "byted" + "ance",
            "byte" + "intl",
            "cloud" + "native",
            "ies_" + "content_algorithm",
            "mlx " + "worker",
            "mer" + "lin" + "-cli",
            "huggingface" + "-proxy",
        ]
        credential_patterns = [
            re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
            re.compile(r"\bASIA[0-9A-Z]{16}\b"),
            re.compile(r"\bhf_[A-Za-z0-9]{20,}\b"),
            re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
            re.compile(
                r"-----BEGIN (?:RSA |OPENSSH |EC |DSA )?PRIVATE KEY-----"
            ),
        ]
        violations = []
        for path in PACKAGE_ROOT.rglob("*"):
            if is_generated_test_cache(path):
                continue
            if not path.is_file():
                continue
            relative_path = path.relative_to(PACKAGE_ROOT)
            relative = relative_path.as_posix()
            text = path.read_text(encoding="utf-8", errors="ignore")
            lowered = text.lower()
            if relative_path.parts[0] != "verl":
                for marker in forbidden_markers:
                    if marker.lower() in lowered:
                        violations.append(f"{relative}: {marker}")
            for pattern in credential_patterns:
                if pattern.search(text):
                    violations.append(f"{relative}: {pattern.pattern}")
        self.assertEqual(violations, [])

    def test_expected_configuration(self):
        text = (PACKAGE_ROOT / "config" / "best.env").read_text(encoding="utf-8")
        expected = {
            # The reported Qwen3.5-4B setup: 4B student (hidden 2560) on a global
            # batch of 48 for one epoch. config/best.env, the recipe launchers and
            # this test have to agree, so a change to the recipe fails here rather
            # than quietly diverging from the paper's table.
            "CWAD_MODEL_HIDDEN_SIZE=2560",
            "CWAD_STUDENT_MODEL_HIDDEN_SIZE=2560",
            "CWAD_TEACHER_MODEL_HIDDEN_SIZE=4096",
            "CWAD_GPUS=8",
            "CWAD_PROMPTS_PER_RANK=6",
            "CWAD_DUAL_WORLD_PROMPTS_PER_RANK=6",
            "CWAD_LEARNING_RATE=2e-6",
            "CWAD_WEIGHT_DECAY=1e-2",
            "CWAD_LR_SCHEDULER=constant",
            "CWAD_WARMUP_STEPS=10",
            "CWAD_FREEZE_VISION_TOWER=False",
            "CWAD_CROSS_MAX_PROMPT_LENGTH=8192",
            "CWAD_CROSS_MAX_RESPONSE_LENGTH=1024",
            "CWAD_DUAL_WORLD_MAX_PROMPT_LENGTH=8192",
            "CWAD_DUAL_WORLD_MAX_RESPONSE_LENGTH=1024",
            "CWAD_CROSS_ROLLOUT_TEMPERATURE=2.0",
            "CWAD_ROLLOUT_TOP_P=1.0",
            "CWAD_EXPECTED_ROWS=",
            "CWAD_DATA_PARQUET=",
            "CWAD_TOTAL_STEPS=49",
            "CWAD_TOPK=100",
            "CWAD_ALPHA=1.0",
            "CWAD_TEACHER_UPDATE_RATE=0.05",
            "CWAD_DUAL_WORLD=1",
            "CWAD_LAMBDA=1.0",
            "CWAD_TAU=1.0",
            "CWAD_MAX_IMAGE_PIXELS=1048576",
        }
        for line in expected:
            self.assertIn(line, text)
        stale = [
            line
            for line in text.splitlines()
            if line.startswith("RP_OPSD_")
        ]
        self.assertEqual(stale, [])

    @staticmethod
    def _assignments(text):
        assignments = {}
        for match in re.finditer(r"^[ \t]*(CWAD_[A-Z0-9_]+)=(.*)$", text, re.MULTILINE):
            # The recipe launchers indent each RECIPE entry and annotate it with
            # a trailing comment.
            assignments[match.group(1)] = re.split(r"\s+#", match.group(2), 1)[0].strip()
        return assignments

    def test_recipe_launchers_agree_with_the_configuration(self):
        # The launcher wins over config/best.env, so a value the two disagree on
        # is silently decided by the launcher. Both are meant to state the same
        # reported recipe, so pin that.
        configured = self._assignments(
            (PACKAGE_ROOT / "config" / "best.env").read_text(encoding="utf-8")
        )
        for recipe in ("run_cwad.sh", "run_cwad_hint.sh"):
            source = (PACKAGE_ROOT / "scripts" / recipe).read_text(encoding="utf-8")
            launcher = self._assignments(source)
            self.assertTrue(launcher, f"{recipe} states no recipe values")
            conflicts = {
                key: (configured[key], value)
                for key, value in launcher.items()
                if key in configured and configured[key] != value
            }
            self.assertEqual(conflicts, {}, f"{recipe} contradicts config/best.env")

    def test_embedded_verl_drops_every_inherited_brand_name(self):
        # The training branch is selected by policy_loss.loss_mode, the loss by
        # distillation_objective, and the world-B pixel cap travels from
        # config/best.env into verl through an environment variable. All three
        # names are part of the public recipe, so pin them.
        brand = re.compile(
            r"(?i)\b(vopd|mopd|opsd|sdpo|opd|rp[-_]opsd|vision[-_]opd)\b"
        )
        sources = [
            path
            for path in (PACKAGE_ROOT / "verl").rglob("*")
            if path.suffix in {".py", ".yaml"} and path.is_file()
        ]
        self.assertTrue(sources)
        violations = []
        for path in sources:
            text = path.read_text(encoding="utf-8", errors="ignore")
            for match in brand.finditer(text):
                violations.append(f"{path.relative_to(PACKAGE_ROOT)}: {match.group(0)}")
        self.assertEqual(violations, [])

        required_fragments = {
            "verl/trainer/ppo/core_algos.py": (
                'distillation_objective == "cwad_topk_reverse_kl"',
                "def compute_cwad_loss(",
                "bias_correction = teacher_probs - student_probs",
            ),
            "verl/workers/actor/dp_actor.py": (
                'self_distillation_enabled = loss_mode == "cwad"',
                "metrics[\"actor/cwad_loss\"]",
            ),
            "verl/workers/config/actor.py": (
                'valid_distillation_objectives = ["generalized_jsd", "cwad_topk_reverse_kl"]',
                "cwad_topk_reverse_kl requires alpha=1.0",
            ),
            "verl/utils/dataset/vision_utils.py": ('CWAD_MAX_IMAGE_PIXELS',),
        }
        for relative, fragments in required_fragments.items():
            path = PACKAGE_ROOT / relative
            text = path.read_text(encoding="utf-8")
            for fragment in fragments:
                self.assertIn(fragment, text, f"{fragment} in {relative}")

        # train.sh must pass exactly these two values on the command line.
        self.assertEqual(
            sorted(
                path.name
                for path in (PACKAGE_ROOT / "verl" / "trainer" / "config").glob("*.yaml")
                if path.name.startswith("cwad")
            ),
            ["cwad.yaml", "cwad_hint.yaml"],
        )

    def test_native_source_is_embedded_without_upstream_staging(self):
        required = [
            "verl/trainer/main_ppo.py",
            "verl/trainer/ppo/core_algos.py",
            "verl/trainer/ppo/ray_trainer.py",
            "verl/utils/dataset/vision_utils.py",
            "eval/infer.py",
            "eval/judge_qwenlm.py",
            "chat_templates/perception_chat_template_qwen35.jinja",
            "config/best.env",
            "config/best.yaml",
            "environment/environment.yml",
            "provenance/source_files.sha256",
            "scripts/build_dual_world_data.py",
            "scripts/build_cwad_data.sh",
            "scripts/train.sh",
            "scripts/run_verl.sh",
            "scripts/build_hint_parquet.py",
        ]
        for relative in required:
            self.assertTrue((PACKAGE_ROOT / relative).is_file(), relative)
        self.assertFalse((PACKAGE_ROOT / "patches").exists())
        self.assertFalse((PACKAGE_ROOT / "scripts/stage_source.sh").exists())

        operational_paths = [
            PACKAGE_ROOT / "config",
            PACKAGE_ROOT / "scripts",
            PACKAGE_ROOT / "run.sh",
        ]
        forbidden = [
            "VisionOPD/Vision-OPD",
            "RP_OPSD_VISION_OPD_",
            "stage_source.sh",
            "--upstream-source",
            "--source-dir",
            "git clone",
        ]
        for root in operational_paths:
            paths = [root] if root.is_file() else list(root.rglob("*"))
            for path in paths:
                if not path.is_file():
                    continue
                text = path.read_text(encoding="utf-8", errors="ignore")
                for marker in forbidden:
                    self.assertNotIn(marker, text, f"{marker} in {path}")

    def test_bias_corrected_reverse_kl_is_zero_for_equal_distributions(self):
        student = [0.2, 0.3, 0.1]
        teacher = list(student)
        value = sum(
            ps * math.log(ps / pt) - ps + pt
            for ps, pt in zip(student, teacher, strict=True)
        )
        self.assertAlmostEqual(value, 0.0, places=12)

    def test_bias_corrected_reverse_kl_is_positive(self):
        student = [0.4, 0.1, 0.05]
        teacher = [0.2, 0.25, 0.1]
        value = sum(
            ps * math.log(ps / pt) - ps + pt
            for ps, pt in zip(student, teacher, strict=True)
        )
        self.assertGreater(value, 0.0)

    @staticmethod
    def make_mini_dataset(root: Path) -> Path:
        """A two-row dataset tree: relative world A/B images plus one edit each."""
        Image = import_data_libraries()[2]
        dataset = root / "dataset"
        (dataset / "images").mkdir(parents=True)
        edits = root / "edits"
        edits.mkdir()
        for name in ("world_a.png", "world_b.png"):
            Image.new("RGB", (8, 6), color="white").save(dataset / "images" / name)
        for ordinal in (7, 8):
            Image.new("RGB", (10, 10), color="black").save(
                edits / f"idx{ordinal}_edited.jpg"
            )
        rows = [
            {
                "data_source": "unit",
                "prompt": [{"role": "user", "content": f"<image>\nQuestion {ordinal}?"}],
                "images": [{"image": "images/world_a.png"}],
                "teacher_images": [{"image": "images/world_b.png"}],
                "ability": "image_reasoning_mcq",
                "reward_model": {"style": "rule", "ground_truth": "A"},
                "extra_info": {"task_family": "mcq", "psr_ordinal": ordinal},
            }
            for ordinal in (7, 8)
        ]
        jsonl = dataset / "train.jsonl"
        jsonl.write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
        return jsonl

    def build_dual_world(self, root: Path, jsonl: Path, output: Path, *extra: str):
        subprocess.run(
            [
                sys.executable,
                str(PACKAGE_ROOT / "scripts" / "build_dual_world_data.py"),
                "--dataset",
                str(jsonl),
                "--output-dir",
                str(output),
                *extra,
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        return json.loads(
            (output / "dual_world_summary.json").read_text(encoding="utf-8")
        )

    @unittest.skipUnless(
        SCRIPTS_READY, "scripts/ has not been added to this tree yet"
    )
    def test_dual_world_builder_resolves_relative_paths(self):
        """Any dataset tree with per-row world A/B images builds a runtime parquet."""
        pq = import_data_libraries()[1]
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            jsonl = self.make_mini_dataset(root)
            output = root / "build"
            summary = self.build_dual_world(
                root, jsonl, output, "--edit-dir", str(root / "edits")
            )
            self.assertEqual(
                sorted(p.name for p in output.iterdir()),
                ["dual_world.parquet", "dual_world_summary.json"],
            )
            records = pq.read_table(output / "dual_world.parquet").to_pylist()
            world_a = str((root / "dataset" / "images" / "world_a.png").resolve())
            self.assertEqual([r["idx"] for r in records], [0, 1])
            self.assertEqual([r["images"][0]["image"] for r in records], [world_a] * 2)
            self.assertEqual(
                [r["teacher_images"][0]["image"] for r in records],
                [
                    str((root / "edits" / f"idx{ordinal}_edited.jpg").resolve())
                    for ordinal in (7, 8)
                ],
            )
            self.assertEqual(
                [r["extra_info"]["dual_world_has_edit"] for r in records], [True, True]
            )
            # Whatever metadata the dataset carried survives the rewrite.
            self.assertEqual([r["data_source"] for r in records], ["unit", "unit"])
            self.assertEqual(
                [r["extra_info"]["psr_ordinal"] for r in records], [7, 8]
            )
            self.assertEqual(
                [summary[key] for key in ("rows", "edits_available", "edits_used",
                                          "edits_unmatched", "world_b_dataset_fallback")],
                [2, 2, 2, 0, 0],
            )

            # Without an edit directory every row keeps its own world-B image.
            fallback = root / "no_edits"
            summary = self.build_dual_world(root, jsonl, fallback)
            records = pq.read_table(fallback / "dual_world.parquet").to_pylist()
            self.assertEqual(
                [r["teacher_images"][0]["image"] for r in records],
                [str((root / "dataset" / "images" / "world_b.png").resolve())] * 2,
            )
            self.assertEqual(
                [summary[key] for key in ("world_b_counterfactual_edit",
                                          "world_b_dataset_fallback")],
                [0, 2],
            )

    @unittest.skipUnless(
        SCRIPTS_READY, "scripts/ has not been added to this tree yet"
    )
    def test_dual_world_builder_id_filters(self):
        pq = import_data_libraries()[1]
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            jsonl = self.make_mini_dataset(root)

            # A QC list that disqualifies one edit keeps both rows and downgrades
            # that one to the dataset's own world-B image.
            (root / "good.jsonl").write_text('{"idx": 8}\n', encoding="utf-8")
            disqual = root / "disqualified"
            summary = self.build_dual_world(
                root,
                jsonl,
                disqual,
                "--edit-dir",
                str(root / "edits"),
                "--good-edit-ids-file",
                str(root / "good.jsonl"),
            )
            records = pq.read_table(disqual / "dual_world.parquet").to_pylist()
            self.assertEqual(
                [r["extra_info"]["dual_world_has_edit"] for r in records],
                [False, True],
            )
            self.assertEqual(
                records[0]["teacher_images"][0]["image"],
                str((root / "dataset" / "images" / "world_b.png").resolve()),
            )
            self.assertEqual([summary["rows"], summary["world_b_counterfactual_edit"]], [2, 1])

            # A restricted run drops the rows that were not listed, and renumbers
            # what remains so the hint tools can key on the ordinal.
            (root / "only.jsonl").write_text('{"idx": 7}\n', encoding="utf-8")
            kept = root / "kept"
            summary = self.build_dual_world(
                root,
                jsonl,
                kept,
                "--edit-dir",
                str(root / "edits"),
                "--only-ids-file",
                str(root / "only.jsonl"),
            )
            records = pq.read_table(kept / "dual_world.parquet").to_pylist()
            self.assertEqual([r["idx"] for r in records], [0])
            self.assertEqual(records[0]["extra_info"]["dual_world_has_edit"], True)
            self.assertEqual(
                [summary["rows"], summary["dataset_rows"], summary["ids_requested"]],
                [1, 2, 1],
            )

            # An id list nothing matches is a mistake, not an empty dataset.
            (root / "none.jsonl").write_text('{"idx": 999}\n', encoding="utf-8")
            rejected = subprocess.run(
                [
                    sys.executable,
                    str(PACKAGE_ROOT / "scripts" / "build_dual_world_data.py"),
                    "--dataset",
                    str(jsonl),
                    "--edit-dir",
                    str(root / "edits"),
                    "--only-ids-file",
                    str(root / "none.jsonl"),
                    "--output-dir",
                    str(root / "never"),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn("no rows selected", rejected.stderr + rejected.stdout)

    @unittest.skipUnless(
        SCRIPTS_READY, "scripts/ has not been added to this tree yet"
    )
    def test_hint_parquet_writes_hints_into_extra_info(self):
        pa, pq = import_data_libraries()[:2]
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            rows = [
                {
                    "prompt": [{"role": "user", "content": "<image>\nA?"}],
                    "images": [{"image": "/abs/a.png"}],
                    "teacher_images": [{"image": "/abs/b.png"}],
                    "extra_info": {"answer_hint_b": "because B"},
                },
                {
                    "prompt": [{"role": "user", "content": "<image>\nB?"}],
                    "images": [{"image": "/abs/c.png"}],
                    "teacher_images": [{"image": "/abs/d.png"}],
                    "extra_info": {"answer_hint_b": "because C"},
                },
            ]
            source = root / "dual_world.parquet"
            pq.write_table(pa.Table.from_pylist(rows), source)
            hints = root / "hints.jsonl"
            hints.write_text(
                "".join(
                    json.dumps({"idx": index, "reasoning": "R" * 120 + str(index)})
                    + "\n"
                    for index in range(2)
                ),
                encoding="utf-8",
            )
            output = root / "hinted"
            subprocess.run(
                [
                    sys.executable,
                    str(PACKAGE_ROOT / "scripts" / "build_hint_parquet.py"),
                    "--source-parquet",
                    str(source),
                    "--hints",
                    str(hints),
                    "--output-dir",
                    str(output),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            written = pq.read_table(output / "hinted.parquet").to_pylist()
            self.assertEqual(
                [record["extra_info"]["answer_hint"][:2] for record in written],
                ["RR", "RR"],
            )
            # The dataset's own world-B hint survives alongside the new field.
            self.assertEqual(
                [record["extra_info"]["answer_hint_b"] for record in written],
                ["because B", "because C"],
            )

            # An incomplete hint set must not silently produce a shorter run.
            hints.write_text(
                json.dumps({"idx": 0, "reasoning": "R" * 120}) + "\n", encoding="utf-8"
            )
            rejected = subprocess.run(
                [
                    sys.executable,
                    str(PACKAGE_ROOT / "scripts" / "build_hint_parquet.py"),
                    "--source-parquet",
                    str(source),
                    "--hints",
                    str(hints),
                    "--output-dir",
                    str(root / "never"),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn("usable hints", rejected.stderr)


    @unittest.skipUnless(SCRIPTS_READY, "scripts/ has not been added to this tree yet")
    def test_metric_collection_protocol(self):
        module = import_script("collect_metrics.py")
        records = [
            {"judge": "Yes", "category": "Easy"},
            {"judge": "No", "category": "Easy"},
            {"judge": "Yes", "category": "Medium"},
            {"judge": "Yes", "category": "Hard"},
        ]
        self.assertAlmostEqual(module.score(records, "visualprobe"), 83.3333333333)
        self.assertEqual(module.score([{"judge": "Yes"}, {"judge": "No"}], "mmstar"), 50.0)

    @unittest.skipUnless(
        SCRIPTS_READY, "scripts/ has not been added to this tree yet"
    )
    def test_verify_row_counts_are_opt_in(self):
        """The count guards belong to the dataset, so an unset pin checks nothing."""
        pq, Image = import_data_libraries()[1], import_data_libraries()[2]
        import pyarrow as pa

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for name in ("a.png", "b.png"):
                Image.new("RGB", (8, 6), color="white").save(root / name)
            rows = [
                {
                    "prompt": [{"role": "user", "content": "<image>\nQuestion?"}],
                    "images": [{"image": str(root / "a.png")}],
                    "teacher_images": [{"image": str(root / "b.png")}],
                    "extra_info": {"task_family": "mcq"},
                }
            ] * 2
            parquet = root / "dual_world.parquet"
            pq.write_table(pa.Table.from_pylist(rows), parquet)
            command = [
                sys.executable,
                str(PACKAGE_ROOT / "scripts" / "verify.py"),
                "--data-parquet",
                str(parquet),
                "--skip-resolution-ratio",
            ]

            def run(**env):
                environment = {
                    key: value
                    for key, value in os.environ.items()
                    if not key.startswith("CWAD_EXPECTED_")
                }
                environment.update(env)
                return subprocess.run(
                    command,
                    check=False,
                    capture_output=True,
                    text=True,
                    env=environment,
                )

            unpinned = run()
            self.assertEqual(unpinned.returncode, 0, unpinned.stderr)
            self.assertEqual(json.loads(unpinned.stdout)["data"]["rows"], 2)

            mismatched = run(CWAD_EXPECTED_ROWS="5")
            self.assertNotEqual(mismatched.returncode, 0)
            self.assertIn("expected 5 rows, found 2", mismatched.stderr)

            pinned = run(CWAD_EXPECTED_ROWS="2", CWAD_EXPECTED_MCQ_ROWS="2")
            self.assertEqual(pinned.returncode, 0, pinned.stderr)
            self.assertEqual(
                json.loads(pinned.stdout)["data"]["task_counts"], {"mcq": 2}
            )


if __name__ == "__main__":
    unittest.main()
