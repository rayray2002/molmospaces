"""LinearBot-only bimanual pickup datagen configuration."""

from molmo_spaces.configs.base_pick_config import PickBaseConfig
from molmo_spaces.configs.policy_configs import PickPlannerPolicyConfig
from molmo_spaces.configs.task_configs import BimanualPickTaskConfig
from molmo_spaces.configs.task_sampler_configs import PickTaskSamplerConfig
from molmo_spaces.policy.solvers.object_manipulation.bimanual_pick_planner_policy import (
    BimanualPickPlannerPolicy,
)
from molmo_spaces.tasks.bimanual_pick_task import BimanualPickTask
from molmo_spaces.tasks.bimanual_pick_task_sampler import BimanualPickTaskSampler


class BimanualPickBaseConfig(PickBaseConfig):
    task_type: str = "bimanual_pick"
    task_config: BimanualPickTaskConfig = BimanualPickTaskConfig(task_cls=BimanualPickTask)
    task_sampler_config: PickTaskSamplerConfig = PickTaskSamplerConfig(
        task_sampler_class=BimanualPickTaskSampler
    )
    policy_config: PickPlannerPolicyConfig = PickPlannerPolicyConfig(
        policy_cls=BimanualPickPlannerPolicy, policy_factory=BimanualPickPlannerPolicy
    )
