from typing import Tuple, Any

from gymnasium import Space
from ml_collections import ConfigDict
from polaris.models import BaseModel


class ResNetModel(BaseModel):
    is_recurrent = False

    def __init__(
            self,
            observation_space: Space,
            action_space: Space,
            config: ConfigDict,
    ):
        super().__init__(
            name="ResNetModel",
            observation_space=observation_space,
            action_space=action_space,
            config=config
        )

    def forward_single_action_with_extras(
            self,
            obs,
            prev_action,
            prev_reward,
            state
    ):
        pass

    def forward_single_action(
            self,
            obs,
            prev_action,
            prev_reward,
            state
    ):
        pass

    def __call__(
            self,
            *,
            obs,
            prev_action,
            prev_reward,
            state,
            seq_lens
    ) -> Tuple[Any, Any]:
        pass


