"""
tests/test_specific_fixes.py

Detailed verification of three specific bug fixes.

FIX-1  OOB dead code
    BEFORE: action was clipped inside get_action → env always received in-bounds
            positions → _env_rew never contained PENALTY_OOB → oob_rate = 0 always.
    AFTER:  clip moved to training loop, AFTER an explicit OOB check on the raw
            (unclipped) action.  PENALTY_OOB is injected manually.

FIX-2  Collision does not terminate episode
    BEFORE: terminated = False on collision → agent collided all 25 steps per episode
            with no "stop now" signal → bimodal collision_rate (0% or 100%).
    AFTER:  environment_odin.py sets terminated = True on first collision.

FIX-3  get_action clipped internally (corollary of FIX-1)
    BEFORE: np.clip(action_np, ...) inside get_action → OOB check in training loop
            always saw the clipped action → threshold detection always returned False.
    AFTER:  np.clip removed from get_action; clip lives only in the training loop,
            after the is_manual_oob detection.
"""

import sys
import os
import ast
import unittest
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

# ── Constants matching config.py ──────────────────────────────────────────────
ACTION_MIN = [0.2, -0.5, 0.0, -3.14, -3.14, -3.14]
ACTION_MAX = [0.8,  0.5, 0.8,  3.14,  3.14,  3.14]
PENALTY_OOB       = -2.0
PENALTY_COLLISION = -3.0
_COLL_THRESH = PENALTY_COLLISION / 2.0   # -1.5
_OOB_THRESH  = PENALTY_OOB       / 2.0   # -1.0 (used in old detection)


# ── Replicate the OOB logic from train_odin_sac_rl.py training loop ───────────
def _run_oob_logic(action_xyz_plus_rot, env_reward):
    """
    Simulates the exact block added to the training loop:

        is_manual_oob = ...
        action_np = np.clip(action_np, ACTION_MIN, ACTION_MAX)
        is_collision = _env_rew <= _coll_thresh
        is_oob = is_manual_oob and not is_collision
        if is_oob: _env_rew = PENALTY_OOB
    """
    action_np = np.asarray(action_xyz_plus_rot, dtype=np.float32)
    _amin3 = np.array(ACTION_MIN[:3], dtype=np.float32)
    _amax3 = np.array(ACTION_MAX[:3], dtype=np.float32)

    is_manual_oob = bool(
        np.any(action_np[:3] < _amin3) or np.any(action_np[:3] > _amax3)
    )
    action_clipped = np.clip(action_np, ACTION_MIN, ACTION_MAX)

    _env_rew = float(env_reward)
    is_collision = _env_rew <= _COLL_THRESH
    is_oob = is_manual_oob and not is_collision
    if is_oob:
        _env_rew = PENALTY_OOB

    return {
        "is_manual_oob": is_manual_oob,
        "is_oob": is_oob,
        "is_collision": is_collision,
        "final_env_rew": _env_rew,
        "action_clipped": action_clipped,
    }


# ── AST helpers ───────────────────────────────────────────────────────────────
def _load(rel_path):
    """Return (source_text, ast_tree) for a project file."""
    path = os.path.join(ROOT, rel_path)
    with open(path, encoding="utf-8") as f:
        src = f.read()
    return src, ast.parse(src)


def _method_source(src, tree, method_name):
    """Source text of a specific function/method."""
    lines = src.splitlines()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == method_name:
            return "\n".join(lines[node.lineno - 1: node.end_lineno])
    return ""


def _find_if_node(tree, test_var_name):
    """Return first ast.If whose test is a plain Name matching test_var_name."""
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.If)
            and isinstance(node.test, ast.Name)
            and node.test.id == test_var_name
        ):
            return node
    return None


def _block_assigns_true(block_stmts, var_name):
    """Return True if any statement in block_stmts assigns True to var_name."""
    for stmt in block_stmts:
        if not isinstance(stmt, ast.Assign):
            continue
        for target in stmt.targets:
            if isinstance(target, ast.Name) and target.id == var_name:
                val = stmt.value
                if isinstance(val, ast.Constant) and val.value is True:
                    return True
                # Python < 3.8 compat
                if isinstance(val, ast.NameConstant) and val.value is True:
                    return True
    return False


def _block_assigns_false(block_stmts, var_name):
    """Return True if any statement in block_stmts assigns False to var_name."""
    for stmt in block_stmts:
        if not isinstance(stmt, ast.Assign):
            continue
        for target in stmt.targets:
            if isinstance(target, ast.Name) and target.id == var_name:
                val = stmt.value
                if isinstance(val, ast.Constant) and val.value is False:
                    return True
                if isinstance(val, ast.NameConstant) and val.value is False:
                    return True
    return False


# =============================================================================
class TestOOBLogic(unittest.TestCase):
    """FIX-1 behavioral: OOB detection + penalty injection works correctly."""

    # ── out-of-bounds positions ───────────────────────────────────────────────

    def test_x_above_max_detected(self):
        r = _run_oob_logic([0.9, 0.0, 0.3, 0, 0, 0], env_reward=0.1)
        self.assertTrue(r["is_manual_oob"])
        self.assertTrue(r["is_oob"])

    def test_x_below_min_detected(self):
        r = _run_oob_logic([0.1, 0.0, 0.3, 0, 0, 0], env_reward=0.1)
        self.assertTrue(r["is_manual_oob"])
        self.assertTrue(r["is_oob"])

    def test_negative_x_detected(self):
        r = _run_oob_logic([-0.1, 0.0, 0.3, 0, 0, 0], env_reward=0.1)
        self.assertTrue(r["is_oob"])

    def test_y_above_max_detected(self):
        r = _run_oob_logic([0.5,  0.6, 0.3, 0, 0, 0], env_reward=0.1)
        self.assertTrue(r["is_manual_oob"])

    def test_y_below_min_detected(self):
        r = _run_oob_logic([0.5, -0.9, 0.3, 0, 0, 0], env_reward=0.1)
        self.assertTrue(r["is_manual_oob"])

    def test_z_above_max_detected(self):
        r = _run_oob_logic([0.5,  0.0, 1.0, 0, 0, 0], env_reward=0.1)
        self.assertTrue(r["is_manual_oob"])

    def test_z_below_min_detected(self):
        r = _run_oob_logic([0.5,  0.0, -0.1, 0, 0, 0], env_reward=0.1)
        self.assertTrue(r["is_manual_oob"])

    # ── penalty injection ─────────────────────────────────────────────────────

    def test_penalty_injected_when_oob(self):
        """final_env_rew must equal PENALTY_OOB when action is out of bounds."""
        r = _run_oob_logic([0.9, 0.0, 0.3, 0, 0, 0], env_reward=0.1)
        self.assertEqual(r["final_env_rew"], PENALTY_OOB)

    def test_penalty_overrides_positive_env_reward(self):
        """Even if the env returned a positive reward (clipped action was safe),
        OOB still overrides it with PENALTY_OOB."""
        r = _run_oob_logic([1.5, 0.0, 0.3, 0, 0, 0], env_reward=5.0)
        self.assertEqual(r["final_env_rew"], PENALTY_OOB)

    def test_env_reward_unchanged_when_in_bounds(self):
        """In-bounds actions must NOT have their env reward touched."""
        r = _run_oob_logic([0.5, 0.0, 0.3, 0, 0, 0], env_reward=2.5)
        self.assertFalse(r["is_oob"])
        self.assertAlmostEqual(r["final_env_rew"], 2.5)

    # ── in-bounds: no false positives ────────────────────────────────────────

    def test_valid_center_action_no_oob(self):
        r = _run_oob_logic([0.5, 0.0, 0.3, 0, 0, 0], env_reward=0.1)
        self.assertFalse(r["is_manual_oob"])
        self.assertFalse(r["is_oob"])

    def test_action_on_lower_boundary_not_oob(self):
        r = _run_oob_logic([0.2, -0.5, 0.0, 0, 0, 0], env_reward=0.1)
        self.assertFalse(r["is_manual_oob"])

    def test_action_on_upper_boundary_not_oob(self):
        r = _run_oob_logic([0.8, 0.5, 0.8, 0, 0, 0], env_reward=0.1)
        self.assertFalse(r["is_manual_oob"])

    def test_only_rotation_oob_no_penalty(self):
        """OOB only checks position dims (0:3); large rotation values are irrelevant."""
        r = _run_oob_logic([0.5, 0.0, 0.3, 5.0, -5.0, 5.0], env_reward=0.1)
        self.assertFalse(r["is_manual_oob"])
        self.assertAlmostEqual(r["final_env_rew"], 0.1)

    # ── collision takes priority ──────────────────────────────────────────────

    def test_collision_overrides_oob(self):
        """If both OOB and collision, collision takes priority; is_oob = False."""
        r = _run_oob_logic([1.5, 0.0, 0.3, 0, 0, 0], env_reward=PENALTY_COLLISION)
        self.assertTrue(r["is_manual_oob"])
        self.assertTrue(r["is_collision"])
        self.assertFalse(r["is_oob"])
        self.assertEqual(r["final_env_rew"], PENALTY_COLLISION)

    # ── clipping after detection ──────────────────────────────────────────────

    def test_clipped_action_within_bounds(self):
        """The action sent to env must always be inside bounds, regardless of OOB."""
        for action in [
            [1.5,  0.0, 0.3, 0, 0, 0],
            [0.5,  1.5, 0.3, 0, 0, 0],
            [0.5,  0.0, 2.0, 0, 0, 0],
            [-0.5, 0.0, 0.3, 0, 0, 0],
        ]:
            with self.subTest(action=action):
                r = _run_oob_logic(action, env_reward=0.1)
                self.assertGreaterEqual(r["action_clipped"][0], ACTION_MIN[0])
                self.assertLessEqual(r["action_clipped"][0],    ACTION_MAX[0])
                self.assertGreaterEqual(r["action_clipped"][1], ACTION_MIN[1])
                self.assertLessEqual(r["action_clipped"][1],    ACTION_MAX[1])
                self.assertGreaterEqual(r["action_clipped"][2], ACTION_MIN[2])
                self.assertLessEqual(r["action_clipped"][2],    ACTION_MAX[2])

    # ── prove old detection was broken ────────────────────────────────────────

    def test_old_threshold_detection_misses_oob_after_clip(self):
        """Reproduces the original bug.

        The action is OOB (x=0.9), but the env saw the clipped action (x=0.8)
        and returned a NORMAL positive reward.  The old check
        `is_oob = (not is_collision) and (_env_rew <= _oob_thresh)` would
        evaluate as False because 0.1 > -1.0.

        The NEW check catches it through is_manual_oob.
        """
        # Simulate: env received clipped x=0.8 and returned a normal reward
        env_reward_for_clipped_action = 0.1

        # Old detection: threshold on env reward
        old_is_oob = (env_reward_for_clipped_action <= _OOB_THRESH)
        self.assertFalse(old_is_oob, "Old detection correctly evaluates to False (bug)")

        # New detection: checks unclipped action
        r = _run_oob_logic([0.9, 0.0, 0.3, 0, 0, 0], env_reward=env_reward_for_clipped_action)
        self.assertTrue(r["is_oob"], "New detection correctly catches OOB (fix)")
        self.assertEqual(r["final_env_rew"], PENALTY_OOB)


# =============================================================================
class TestGetActionNoClip(unittest.TestCase):
    """FIX-3 source: get_action must NOT contain np.clip (moved to training loop)."""

    @classmethod
    def setUpClass(cls):
        cls.src, cls.tree = _load("train_odin_sac_rl.py")
        cls.get_action_body = _method_source(cls.src, cls.tree, "get_action")

    def test_get_action_method_found(self):
        self.assertGreater(len(self.get_action_body), 0, "get_action not found in source")

    def test_no_np_clip_inside_get_action(self):
        """np.clip must NOT appear inside get_action — it was the root cause of FIX-1."""
        self.assertNotIn(
            "np.clip", self.get_action_body,
            "np.clip still present inside get_action — FIX-3 not applied"
        )

    def test_is_manual_oob_exists_in_source(self):
        """is_manual_oob must be defined somewhere in the training script."""
        self.assertIn(
            "is_manual_oob", self.src,
            "is_manual_oob variable not found in train_odin_sac_rl.py"
        )

    def test_np_clip_follows_is_manual_oob(self):
        """np.clip in the training loop must appear AFTER is_manual_oob detection."""
        oob_pos = self.src.find("is_manual_oob")
        self.assertGreater(oob_pos, 0, "is_manual_oob not found")
        clip_pos = self.src.find("np.clip", oob_pos)
        self.assertGreater(
            clip_pos, oob_pos,
            "np.clip not found after is_manual_oob — clip must follow OOB detection"
        )

    def test_clip_in_training_loop_uses_config_bounds(self):
        """The clip after is_manual_oob must reference config.ACTION_MIN / ACTION_MAX."""
        oob_pos = self.src.find("is_manual_oob")
        clip_pos = self.src.find("np.clip", oob_pos)
        line_end = self.src.find("\n", clip_pos)
        clip_line = self.src[clip_pos:line_end]
        self.assertIn("ACTION_MIN", clip_line)
        self.assertIn("ACTION_MAX", clip_line)

    def test_is_oob_uses_manual_flag_not_threshold(self):
        """The is_oob assignment must reference is_manual_oob, not _oob_thresh."""
        is_oob_pos = self.src.find("is_oob = is_manual_oob")
        self.assertGreater(
            is_oob_pos, 0,
            "is_oob assignment must use is_manual_oob, not threshold-based detection"
        )

    def test_env_rew_override_present(self):
        """When is_oob, _env_rew must be overridden with PENALTY_OOB."""
        self.assertIn(
            "PENALTY_OOB", self.src[self.src.find("is_oob"):],
            "_env_rew = config.PENALTY_OOB override not found after is_oob block"
        )


# =============================================================================
class TestCollisionTermination(unittest.TestCase):
    """FIX-2 source: environment_odin.py must set terminated=True on collision."""

    @classmethod
    def setUpClass(cls):
        cls.src, cls.tree = _load("src/simulation/environment_odin.py")
        cls.collision_node = _find_if_node(cls.tree, "collision")

    def test_collision_if_block_exists(self):
        self.assertIsNotNone(
            self.collision_node,
            "if collision: block not found in environment_odin.py"
        )

    def test_terminated_true_inside_collision_block(self):
        """terminated = True must be directly inside the if collision: body."""
        found = _block_assigns_true(self.collision_node.body, "terminated")
        self.assertTrue(
            found,
            "terminated = True not found inside 'if collision:' — FIX-2 not applied"
        )

    def test_terminated_false_not_inside_collision_block(self):
        """terminated = False must NOT appear inside the collision block
        (it would immediately undo the fix)."""
        wrongly_reset = _block_assigns_false(self.collision_node.body, "terminated")
        self.assertFalse(
            wrongly_reset,
            "terminated = False found inside collision block — fix is broken"
        )

    def test_penalty_collision_assigned_in_collision_block(self):
        """reward = config.PENALTY_COLLISION must be in the collision block."""
        lines = self.src.splitlines()
        block_src = "\n".join(
            lines[self.collision_node.lineno - 1: self.collision_node.body[-1].end_lineno]
        )
        self.assertIn("PENALTY_COLLISION", block_src)

    def test_terminated_false_is_default_before_block(self):
        """terminated = False must be initialised somewhere BEFORE the collision check,
        so non-collision steps start with the correct default."""
        collision_line = self.collision_node.lineno
        pre_block_src = "\n".join(self.src.splitlines()[:collision_line - 1])
        self.assertIn(
            "terminated = False", pre_block_src,
            "terminated = False initialisation not found before collision block"
        )

    def test_terminated_true_not_in_else_branch(self):
        """terminated = True should NOT appear in the elif/else branch
        (the OOB branch or success branch sets it independently, not as a side-effect
        of the collision fix)."""
        # The orelse contains the elif: check its body doesn't wrongly set terminated=True
        # due to our edit bleeding into the wrong branch.
        if self.collision_node.orelse:
            orelse_body = self.collision_node.orelse
            # orelse might be another If node (elif) or a list of stmts (else)
            if isinstance(orelse_body[0], ast.If):
                elif_stmts = orelse_body[0].body
            else:
                elif_stmts = orelse_body
            # terminated = True in the elif/else would mean collision fix leaked
            for stmt in elif_stmts:
                if isinstance(stmt, ast.Assign):
                    for tgt in stmt.targets:
                        if isinstance(tgt, ast.Name) and tgt.id == "terminated":
                            val = stmt.value
                            is_true = (
                                (isinstance(val, ast.Constant) and val.value is True)
                                or (isinstance(val, ast.NameConstant) and val.value is True)
                            )
                            self.assertFalse(
                                is_true,
                                "terminated = True leaked into elif/else branch — check edit"
                            )


# =============================================================================
class TestOOBCollisionInteraction(unittest.TestCase):
    """Cross-cutting: OOB and collision interact correctly in combined scenarios."""

    def test_multiple_oob_axes_detected(self):
        """Action violating multiple axes at once still registers as OOB."""
        r = _run_oob_logic([1.5, 1.5, 2.0, 0, 0, 0], env_reward=0.1)
        self.assertTrue(r["is_oob"])
        self.assertEqual(r["final_env_rew"], PENALTY_OOB)

    def test_collision_near_oob_threshold(self):
        """env_reward just below collision threshold → collision, not OOB."""
        r = _run_oob_logic([1.5, 0.0, 0.3, 0, 0, 0], env_reward=-1.6)  # < -1.5
        self.assertTrue(r["is_collision"])
        self.assertFalse(r["is_oob"])

    def test_valid_reward_just_above_thresholds_not_flagged(self):
        """env_reward = -1.4 (between oob_thresh and coll_thresh) is neither."""
        r = _run_oob_logic([0.5, 0.0, 0.3, 0, 0, 0], env_reward=-1.4)
        self.assertFalse(r["is_collision"])
        self.assertFalse(r["is_oob"])

    def test_oob_detection_purely_positional(self):
        """Verify OOB depends only on position dims, not rotation dims."""
        in_bounds_pos   = [0.5, 0.0, 0.3, 99.0, -99.0, 99.0]  # rotation extreme
        out_bounds_pos  = [0.9, 0.0, 0.3,  0.0,   0.0,  0.0]  # x out
        r_in  = _run_oob_logic(in_bounds_pos,  env_reward=0.1)
        r_out = _run_oob_logic(out_bounds_pos, env_reward=0.1)
        self.assertFalse(r_in["is_manual_oob"],  "Rotation OOB incorrectly flagged")
        self.assertTrue(r_out["is_manual_oob"], "Position OOB not detected")


if __name__ == "__main__":
    unittest.main(verbosity=2)
