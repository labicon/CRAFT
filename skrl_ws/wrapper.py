import isaacgym

from skrl.envs.wrappers.torch.base import Wrapper, MultiAgentEnvWrapper
from typing import Any, Tuple, Union, Mapping, Sequence

import gymnasium
import torch
import numpy as np

from skrl.utils.spaces.torch import (
    convert_gym_space,
    flatten_tensorized_space,
    tensorize_space,
    unflatten_tensorized_space,
)

from mqe.envs.utils import make_mqe_env, custom_cfg

class MQESingleAgentWrapper(Wrapper):
    def __init__(self, env: Any) -> None:
        """Isaac Gym environment (preview 3) wrapper

        :param env: The environment to wrap
        :type env: Any supported Isaac Gym environment (preview 3) environment
        """
        super().__init__(env)

        self._reset_once = True
        self._observations = None
        self._info = {}

    @property
    def observation_space(self) -> gymnasium.Space:
        """Observation space"""
        return convert_gym_space(self._env.observation_space)

    @property
    def action_space(self) -> gymnasium.Space:
        """Action space"""
        return convert_gym_space(self._env.action_space)

    @property
    def state_space(self) -> Union[gymnasium.Space, None]:
        """State space"""
        try:
            if self.num_states:
                return convert_gym_space(self._env.state_space)
        except:
            pass
        return None

    def step(self, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Any]:
        """Perform a step in the environment

        :param actions: The actions to perform
        :type actions: torch.Tensor

        :return: Observation, reward, terminated, truncated, info
        :rtype: tuple of torch.Tensor and any other info
        """
        unflattened_action = unflatten_tensorized_space(self.action_space, actions)
        unflattened_action = torch.tensor(unflattened_action, dtype=torch.float).unsqueeze(1)
        observations, reward, terminated, self._info = self._env.step(unflattened_action)
        self._observations = flatten_tensorized_space(tensorize_space(self.observation_space, observations))
        truncated = self._info["time_outs"] if "time_outs" in self._info else torch.zeros_like(terminated)
        return self._observations, reward.view(-1, 1), terminated.view(-1, 1), truncated.view(-1, 1), self._info

    def reset(self) -> Tuple[torch.Tensor, Any]:
        """Reset the environment

        :return: Observation, info
        :rtype: torch.Tensor and any other info
        """
        if self._reset_once:
            observations = self._env.reset()
            self._observations = flatten_tensorized_space(tensorize_space(self.observation_space, observations))
            self._reset_once = False
        return self._observations, self._info

    def render(self, *args, **kwargs) -> None:
        """Render the environment"""
        return None

    def close(self) -> None:
        """Close the environment"""
        pass

class MQECentralizedWrapper(Wrapper):
    def __init__(self, env: Any) -> None:
        """Isaac Gym environment (preview 3) wrapper

        :param env: The environment to wrap
        :type env: Any supported Isaac Gym environment (preview 3) environment
        """
        super().__init__(env)

        self._reset_once = True
        self._observations = None
        self._info = {}

    @property
    def observation_space(self) -> gymnasium.Space:
        """Observation space"""
        obs_space = self._env.observation_space
        global_state_low = np.tile(obs_space.low, 2)
        global_state_high = np.tile(obs_space.high, 2)
        global_state_shape = 2 * obs_space.shape[0]
        global_state_space = gymnasium.spaces.Box(low=global_state_low, high=global_state_high, shape=(global_state_shape,))
        return global_state_space

    @property
    def action_space(self) -> gymnasium.Space:
        """Action space"""
        action_space = self._env.action_space
        global_action_low = np.tile(action_space.low, 2)
        global_action_high = np.tile(action_space.high, 2)
        global_action_shape = 2 * action_space.shape[0]
        global_action_space = gymnasium.spaces.Box(low=global_action_low, high=global_action_high, shape=(global_action_shape,))
        return global_action_space

    @property
    def state_space(self) -> Union[gymnasium.Space, None]:
        """State space"""
        return None
    
    @property
    def num_agents(self) -> int:
        return 1

    def step(self, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Any]:
        """Perform a step in the environment

        :param actions: The actions to perform
        :type actions: torch.Tensor

        :return: Observation, reward, terminated, truncated, info
        :rtype: tuple of torch.Tensor and any other info
        """
        unflattened_action = unflatten_tensorized_space(self.action_space, actions)
        unflattened_action = torch.tensor(unflattened_action, dtype=torch.float).reshape(self.num_envs, 2, -1)
        observations, reward, terminated, info = self._env.step(unflattened_action)
        self._observations = flatten_tensorized_space(tensorize_space(self.observation_space, observations))
        truncated = torch.zeros_like(terminated)
        reward = reward.sum(dim=1)  # Sum the rewards for all agents

        # Average out the info for logging
        for key in info.keys():
            if isinstance(info[key], torch.Tensor):
                info[key] = info[key].mean()
            else:
                info[key] = np.mean(info[key])

        self._info["reward"] = info

        return self._observations, reward.view(-1, 1), terminated.view(-1, 1), truncated.view(-1, 1), self._info

    def reset(self) -> Tuple[torch.Tensor, Any]:
        """Reset the environment

        :return: Observation, info
        :rtype: torch.Tensor and any other info
        """
        if self._reset_once:
            observations = self._env.reset()
            self._state = self._env.get_state()
            self._observations = flatten_tensorized_space(tensorize_space(self.observation_space, observations))
            self._reset_once = False
        return self._observations, self._info

    def render(self, *args, **kwargs) -> None:
        """Render the environment"""
        return None

    def close(self) -> None:
        """Close the environment"""
        pass

class MQEMultiAgentWrapper(MultiAgentEnvWrapper):
    def __init__(self, env: Any) -> None:
        """Isaac Lab environment wrapper for multi-agent implementation

        :param env: The environment to wrap
        :type env: Any supported Isaac Lab environment
        """
        super().__init__(env)

        self._reset_once = True
        self._observations = {}
        self._info = {}

    @property
    def max_num_agents(self) -> int:
        """Number of possible agents the environment could generate

        Read from the length of the ``possible_agents`` property if the wrapped environment doesn't define it
        """
        return self._env.num_agents

    @property
    def agents(self) -> Sequence[str]:
        """Names of all current agents

        These may be changed as an environment progresses (i.e. agents can be added or removed)
        """
        return ["agent_" + str(i) for i in range(self.num_agents)]

    @property
    def possible_agents(self) -> Sequence[str]:
        """Names of all possible agents the environment could generate

        These can not be changed as an environment progresses
        """
        return ["agent_" + str(i) for i in range(self.max_num_agents)]

    @property
    def state_spaces(self) -> Mapping[str, gymnasium.Space]:
        """State spaces

        Since the state space is a global view of the environment (and therefore the same for all the agents),
        this property returns a dictionary (for consistency with the other space-related properties) with the same
        space for all the agents
        """
        obs_space = self._env.observation_space
        global_state_low = np.tile(obs_space.low, self.num_agents)
        global_state_high = np.tile(obs_space.high, self.num_agents)
        global_state_shape = self.num_agents * obs_space.shape[0]
        global_state_space = gymnasium.spaces.Box(low=global_state_low, high=global_state_high, shape=(global_state_shape,))
        return {agent: global_state_space for agent in self.agents}
    
        # return {agent: convert_gym_space(self._env.state_space) for agent in self.agents}

    @property
    def observation_spaces(self) -> Mapping[str, gymnasium.Space]:
        """Observation spaces"""
        return {agent: convert_gym_space(self._env.observation_space) for agent in self.agents}

    @property
    def action_spaces(self) -> Mapping[str, gymnasium.Space]:
        """Action spaces"""
        return {agent: convert_gym_space(self._env.action_space) for agent in self.agents}

    def step(self, actions: Mapping[str, torch.Tensor]) -> Tuple[
        Mapping[str, torch.Tensor], # observations
        Mapping[str, torch.Tensor], # rewards
        Mapping[str, torch.Tensor], # terminated
        Mapping[str, torch.Tensor], # truncated
        Mapping[str, Any], # info
    ]:
        """Perform a step in the environment

        :param actions: The actions to perform
        :type actions: dictionary of torch.Tensor

        :return: Observation, reward, terminated, truncated, info
        :rtype: tuple of dictionaries torch.Tensor and any other info
        """
        action_list = []
        for agent in self.agents:
            action = actions[agent]
            action_list.append(action.reshape(self.num_envs, -1).unsqueeze(1))
        # Make the actions to one tensor
        actions = torch.concat(action_list, dim=1)
        actions = actions.type(torch.float32)
        # print("actions", actions)
        observations, rewards, terminated, info = self._env.step(actions)
        truncated = info["time_outs"] if "time_outs" in info else torch.zeros_like(terminated)
        observations = observations.reshape(self.num_envs, self.num_agents, -1)
        rewards = rewards.reshape(self.num_envs, self.num_agents)
        terminated = terminated.unsqueeze(1).repeat(1, self.num_agents)
        truncated = truncated.unsqueeze(1).repeat(1, self.num_agents)

        # Average out the info for logging
        reward_info = {}
        for key in info.keys():
            if key == "step count":
                continue  # Skip step_count as it is not a reward-related info
            for agent_idx in range(info[key].shape[0]):
                agent_name = f"agent_{agent_idx}"
                if isinstance(info[key], torch.Tensor):
                    reward_info[agent_name + "_" + key] = info[key][agent_idx].mean()
                else:
                    reward_info[agent_name + "_" + key] = np.mean(info[key][agent_idx])
        self._info = reward_info
        agent_wise_rewards = {}
        agent_wise_terminated = {}
        agent_wise_truncated = {}
        agent_wise_info = {}
        for (i, agent) in enumerate(self.agents):
            self._observations[agent] = flatten_tensorized_space(
                tensorize_space(self.observation_spaces[agent], observations[:, i, :])
            )
            agent_wise_rewards[agent] = rewards[:, i].unsqueeze(1)
            agent_wise_terminated[agent] = terminated[:, i].unsqueeze(1)
            agent_wise_truncated[agent] = truncated[:, i].unsqueeze(1)
            agent_wise_info[agent] = self._info.copy()

        # print("agent_wise_rewards", agent_wise_rewards)
        # print("agent wise observations", self._observations)
        return (
            self._observations,
            agent_wise_rewards,
            agent_wise_terminated,
            agent_wise_truncated,
            agent_wise_info,
        )

    def reset(self) -> Tuple[Mapping[str, torch.Tensor], Mapping[str, Any]]:
        """Reset the environment

        :return: Observation, info
        :rtype: torch.Tensor and any other info
        """
        if self._reset_once:
            observations = self._env.reset()
            observations = observations.reshape(self.num_envs, self.num_agents, -1)
            for (i, agent) in enumerate(self.agents):
                self._observations[agent] = flatten_tensorized_space(
                    tensorize_space(self.observation_spaces[agent], observations[:, i, :])
                )
            self._reset_once = False
        return self._observations, self._info

    def state(self) -> torch.Tensor:
        """Get the environment state

        :return: State
        :rtype: torch.Tensor
        """
        state = torch.zeros(self.num_envs, self.num_agents, self._env.observation_space.shape[0], device=self.device)
        for (i, agent) in enumerate(self.agents):
            state[:, i, :] = flatten_tensorized_space(
                tensorize_space(self.observation_spaces[agent], self._observations[agent])
            )
        state = state.reshape(self.num_envs, self.num_agents * self._env.observation_space.shape[0])
        return state

        # state = self._env.get_state()
        # return state

    def render(self, *args, **kwargs) -> None:
        """Render the environment"""
        # current_frame = self._env.get_current_frame()
        # resized_frame = cv2.resize(current_frame, (512, 512), interpolation=cv2.INTER_AREA)
        # # print("current_frame", current_frame.shape)
        # # current_frame = np.expand_dims(current_frame, axis=0)
        # # current_frame = current_frame.swapaxes(0, 2).swapaxes(1, 2)
        # # current_frame = np.transpose(current_frame, (1, 2, 0))
        # # current_frame = current_frame.astype(np.uint8)
        # return resized_frame
        return None

    def close(self) -> None:
        """Close the environment"""
        pass


# # Test the wrapper
# if __name__ == "__main__":
#     from mqe.utils import get_args

#     args = get_args()
#     args.headless = False
#     args.task = "go1navigation"
#     args.num_envs = 1
#     args.record_video = False
#     env, _ = make_mqe_env(args.task, args, custom_cfg(args))
#     env = MQESingleAgentWrapper(env)
#     env.reset()
#     for _ in range(1000):
#         # action = env.action_space.sample()
#         action = torch.rand((env.num_envs, env.num_agents, 3), dtype=torch.float32, device="cuda") * 3
#         obs, reward, done, truncated, info = env.step(action)
#         print("action", action)
#         print("obs", obs)
#         print("reward", reward)
#         # if done or truncated:
#         #     break

if __name__ == "__main__":
    from mqe.utils import get_args
    import time

    args = get_args()
    args.headless = False
    args.task = "go1gate"
    args.num_envs = 1
    args.record_video = False
    env, _ = make_mqe_env(args.task, args, custom_cfg(args))
    env = MQEMultiAgentWrapper(env)
    env.reset()
    for _ in range(1000):
        action_spaces = env.action_spaces
        action = {agent: torch.tensor([1.0, 0.0, 0.0], device='cuda') for agent, space in action_spaces.items()}
        obs, reward, done, truncated, info = env.step(action)
        for key in obs.keys():
            print(key, obs[key])
        print("env.state()", env.state())
        print("reward", reward)
        print("info", info) 
        # If any agent has done or truncated, terminate
        for agent in env.agents:
            if torch.any(done[agent]) or torch.any(truncated[agent]):
                print("agent", agent, "is terminated")
                break