"""Closed-loop LinearBot + Pi-policy check with a fake server: the 'model' holds the current pose."""
import runpy
import sys

import numpy as np

import molmo_spaces.policy.learned_policy.bimanual_yam_pi_policy as m

SEEN = {}


class FakeClient:
    def __init__(self, *a, **k):
        pass

    def connect(self, *a, **k):
        pass

    def reset(self):
        pass

    def close(self):
        pass

    def infer(self, obs):
        names = m.BimanualYamPiPolicy.STATE_NAMES
        SEEN["keys"] = sorted(k for k in obs if k not in names)
        SEEN["img_shapes"] = {k: obs[k].shape for k in ("left", "right", "top")}
        state = np.array([obs[n] for n in names], dtype=np.float32)
        return [state.copy() for _ in range(50)]


m.LeRobotGRPCClient = FakeClient
sys.argv = ["run_pipeline.py", "--robot", "linearbot", "--policy", "pi", "--task_type", "pick",
            "--house_inds", "1", "--samples_per_house", "1", "--seed", "2",
            "--run_name_prefix", "fakepi"]
try:
    runpy.run_path("run_pipeline.py", run_name="__main__")
finally:
    print("FAKE_SEEN", SEEN)
