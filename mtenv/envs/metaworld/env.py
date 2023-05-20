# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import random
from typing import Any, Callable, Dict, List, Optional, Tuple

import metaworld
from gym import Env

from mtenv import MTEnv
from mtenv.envs.metaworld.wrappers.normalized_env import (  # type: ignore[attr-defined]
    NormalizedEnvWrapper,
)
from mtenv.envs.shared.wrappers.multienv import MultiEnvWrapper

# import gym
# from goal_env.mujoco import *
EnvBuilderType = Callable[[], Env]
TaskStateType = int
TaskObsType = int
EnvIdToTaskMapType = Dict[str, metaworld.Task]


class MetaWorldMTWrapper(MultiEnvWrapper):
    def __init__(
        self,
        funcs_to_make_envs: List[EnvBuilderType],
        initial_task_state: TaskStateType,
        env_id_to_task_map: EnvIdToTaskMapType,
    ) -> None:
        """Wrapper to make MetaWorld environment compatible with Multitask
        Environment API.  See :cite:`yu2020meta` for more details about
        MetaWorld.

        Args:
            funcs_to_make_envs (List[EnvBuilderType]): list of constructor
                functions to make the environments.
            initial_task_state (TaskStateType): initial task/environment
                to select.
            env_id_to_task_map (EnvIdToTaskMapType): In MetaWorld, each
                environment can be associated with multiple tasks. This
                dict persists the mapping between environment ids and tasks.
        """
        super().__init__(
            funcs_to_make_envs=funcs_to_make_envs,
            initial_task_state=initial_task_state,
        )
        self.env_id_to_task_map = env_id_to_task_map


def get_list_of_func_to_make_envs(
    benchmark: Optional[metaworld.Benchmark],
    benchmark_name: str,
    env_id_to_task_map: Optional[EnvIdToTaskMapType],
    should_perform_reward_normalization: bool = True,
    task_name: str = "pick-place-v1",
    num_copies_per_env: int = 1,
    fix_goal=True,
    task_idx=None
) -> Tuple[List[Any], Dict[str, Any]]:
    """Return a list of functions to construct the MetaWorld environments
    and a mapping of environment ids to tasks.

    Args:
        benchmark (Optional[metaworld.Benchmark]): `benchmark` to create
            tasks from.
        benchmark_name (str): name of the `benchmark`. This is used only
            when the `benchmark` is None.
        env_id_to_task_map (Optional[EnvIdToTaskMapType]): In MetaWorld,
            each environment can be associated with multiple tasks. This
            dict persists the mapping between environment ids and tasks.
        should_perform_reward_normalization (bool, optional): Defaults to
            True.
        task_name (str, optional): In case of MT1, only . Defaults to
            "pick-place-v1".
        num_copies_per_env (int, optional): Number of copies to create for
            each environment. Defaults to 1.

    Raises:
        ValueError: if `benchmark` is None and `benchmark_name` is not
            MT1, MT10, or MT50.

    Returns:
        Tuple[List[Any], Dict[str, Any]]: A tuple of two elements. The
        first element is a list of functions to construct the MetaWorld
        environments and the second is a mapping of environment ids
        to tasks.

    """
    if not benchmark:
        if benchmark_name == "MT1":
            benchmark = metaworld.ML1(task_name,seed=1)
        elif benchmark_name == "MT10":
            benchmark = metaworld.MT10(seed=1)
        elif benchmark_name == "MT50":
            benchmark = metaworld.MT50(seed=1)
        elif benchmark_name == "ML1":
            benchmark = metaworld.ML1(task_name,seed=1)
        elif benchmark_name == "ML10":
            benchmark = metaworld.ML10(seed=1)
        elif benchmark_name == "ML50":
            benchmark = metaworld.ML50(seed=1)
        else:
            raise ValueError(f"benchmark_name={benchmark_name} is not valid.")

    env_id_list = list(benchmark.train_classes.keys())

    def _get_class_items(current_benchmark):
        return current_benchmark.train_classes.items()

    def _get_tasks(current_benchmark):
        return current_benchmark.train_tasks

    def _get_env_id_to_task_map(task_idx = None) -> EnvIdToTaskMapType:
        env_id_to_task_map: EnvIdToTaskMapType = {}
        current_benchmark = benchmark
        for env_id in env_id_list:
            for name, _ in _get_class_items(current_benchmark):
                if name == env_id:
                    if task_idx ==None:
                        task = random.choice(
                            [
                                task
                                for task in _get_tasks(current_benchmark)
                                if task.env_name == name
                            ]
                        )
                    else:
                        tasks = [
                                task
                                for task in _get_tasks(current_benchmark)
                                if task.env_name == name
                            ]
                        task = tasks[task_idx]
                    env_id_to_task_map[env_id] = task
        return env_id_to_task_map

    if env_id_to_task_map is None:
        env_id_to_task_map: EnvIdToTaskMapType = _get_env_id_to_task_map(task_idx=task_idx)  # type: ignore[no-redef]
    assert env_id_to_task_map is not None
    print(env_id_to_task_map)
    def get_func_to_make_envs(env_id: str):
        current_benchmark = benchmark

        def _make_env():
            for name, env_cls in _get_class_items(current_benchmark):
                if name == env_id:
                    env = env_cls()
                    task = env_id_to_task_map[env_id]
                    env.set_task(task)
                    env = NormalizedEnvWrapper(env, normalize_reward=should_perform_reward_normalization)
                    return env

        return _make_env

    def get_func_to_make_deterministic_envs(env_id: str):
        current_benchmark = benchmark

        def _make_env():
            for name, env_cls in _get_class_items(current_benchmark):
                if name == env_id:
                    env = env_cls()
                    task = env_id_to_task_map[env_id]
                    print(task)
                    env.set_task(task)
                    env._freeze_rand_vec = True
                    env = NormalizedEnvWrapper(env, normalize_reward=should_perform_reward_normalization)
                    return env

        return _make_env

    def get_func_to_make_stochastic_envs(env_id: str):
        current_benchmark = benchmark

        def _make_env():
            # print(env_id_to_task_map,env_id,env_id_list)
            for name, env_cls in _get_class_items(current_benchmark):
                # print(name)
                if name == env_id:
                    env = env_cls()
                    task = env_id_to_task_map[env_id]
                    env.set_task(task)
                    env._freeze_rand_vec = False
                    env = NormalizedEnvWrapper(env, normalize_reward=should_perform_reward_normalization)
                    return env

        return _make_env

    def get_func_to_make_extra_envs(env_id: str):
        current_benchmark = benchmark

        def _make_env():
            # print(env_id_to_task_map,env_id,env_id_list)
            for name, env_cls in _get_class_items(current_benchmark):
                # print(name)
                if 1:
                    import goal_env.mujoco
                    import goal_env
                    # env = gym.make('AntMaze1Test-v1')
                    name='AntUp-v2'
                    name='AntMaze1-v1'
                    env = gym.make(name)
                    print(name)
                    # import dm_control2gym
                    # env = dm_control2gym.make(domain_name="reacher", task_name='hard')
                    env=gym.wrappers.RescaleAction(env,-1,1)
                    env = NormalizedEnvWrapper(env, normalize_reward=should_perform_reward_normalization)
                    return env

        return _make_env

    if num_copies_per_env > 1:
        env_id_list = [
            [env_id for _ in range(num_copies_per_env)] for env_id in env_id_list
        ]
        env_id_list = [
            env_id for env_id_sublist in env_id_list for env_id in env_id_sublist
        ]
    if fix_goal:
        funcs_to_make_envs = [get_func_to_make_deterministic_envs(env_id) for env_id in env_id_list]
    else:
        funcs_to_make_envs = [get_func_to_make_stochastic_envs(env_id) for env_id in env_id_list]
        print('stochastic')
    # funcs_to_make_envs = [get_func_to_make_extra_envs(env_id) for env_id in env_id_list]
    # env = funcs_to_make_envs[0]()
    # env.reset()
    # print(env.env._get_pos_goal())
    return funcs_to_make_envs, env_id_to_task_map


def get_list_of_func_to_make_envs_ML(
    benchmark: Optional[metaworld.Benchmark],
    benchmark_name: str,
    env_id_to_task_map: Optional[EnvIdToTaskMapType],
    should_perform_reward_normalization: bool = True,
    task_name: str = "pick-place-v1",
    num_copies_per_env: int = 1,
    fix_goal=True,
    task_idx=None
) -> Tuple[List[Any], Dict[str, Any]]:
    """Return a list of functions to construct the MetaWorld environments
    and a mapping of environment ids to tasks.

    Args:
        benchmark (Optional[metaworld.Benchmark]): `benchmark` to create
            tasks from.
        benchmark_name (str): name of the `benchmark`. This is used only
            when the `benchmark` is None.
        env_id_to_task_map (Optional[EnvIdToTaskMapType]): In MetaWorld,
            each environment can be associated with multiple tasks. This
            dict persists the mapping between environment ids and tasks.
        should_perform_reward_normalization (bool, optional): Defaults to
            True.
        task_name (str, optional): In case of MT1, only . Defaults to
            "pick-place-v1".
        num_copies_per_env (int, optional): Number of copies to create for
            each environment. Defaults to 1.

    Raises:
        ValueError: if `benchmark` is None and `benchmark_name` is not
            MT1, MT10, or MT50.

    Returns:
        Tuple[List[Any], Dict[str, Any]]: A tuple of two elements. The
        first element is a list of functions to construct the MetaWorld
        environments and the second is a mapping of environment ids
        to tasks.

    """
    if not benchmark:
        if benchmark_name == "MT1":
            benchmark = metaworld.ML1(task_name,seed=1)
        elif benchmark_name == "MT10":
            benchmark = metaworld.MT10(seed=1)
        elif benchmark_name == "MT50":
            benchmark = metaworld.MT50(seed=1)
        elif benchmark_name == "ML1":
            benchmark = metaworld.ML1(task_name,seed=1)
        elif benchmark_name == "ML10":
            benchmark = metaworld.ML10(seed=1)
        elif benchmark_name == "ML50":
            benchmark = metaworld.ML50(seed=1)
        else:
            raise ValueError(f"benchmark_name={benchmark_name} is not valid.")

    env_id_list = list(benchmark.train_classes.keys())

    def _get_class_items(current_benchmark):
        return current_benchmark.train_classes.items()

    def _get_tasks(current_benchmark):
        return current_benchmark.train_tasks

    def _get_env_id_to_task_map(task_idx = None) -> EnvIdToTaskMapType:
        env_id_to_task_map: EnvIdToTaskMapType = {}
        current_benchmark = benchmark
        for env_id in env_id_list:
            for name, _ in _get_class_items(current_benchmark):
                if name == env_id:
                    if task_idx ==None:
                        task = random.choice(
                            [
                                task
                                for task in _get_tasks(current_benchmark)
                                if task.env_name == name
                            ]
                        )
                    else:
                        tasks = [
                                task
                                for task in _get_tasks(current_benchmark)
                                if task.env_name == name
                            ]
                        task = tasks[task_idx]
                    env_id_to_task_map[env_id] = task
        return env_id_to_task_map

    if env_id_to_task_map is None:
        env_id_to_task_map: EnvIdToTaskMapType = _get_env_id_to_task_map(task_idx=None)  # type: ignore[no-redef]

    tasks_list = benchmark.train_tasks
    assert env_id_to_task_map is not None
    print(env_id_to_task_map)


    def get_func_to_make_deterministic_envs(env_id,index):
        current_benchmark = benchmark

        def _make_env():
            for name, env_cls in _get_class_items(current_benchmark):
                if name == env_id:
                    env = env_cls()
                    task = tasks_list[index]
                    print(task)
                    env.set_task(task)
                    env._freeze_rand_vec = True
                    env = NormalizedEnvWrapper(env, normalize_reward=should_perform_reward_normalization)
                    return env

        return _make_env

    def get_func_to_make_stochastic_envs(env_id: str):
        current_benchmark = benchmark

        def _make_env():
            # print(env_id_to_task_map,env_id,env_id_list)
            for name, env_cls in _get_class_items(current_benchmark):
                # print(name)
                if name == env_id:
                    env = env_cls()
                    task = tasks_list[index]
                    env.set_task(task)
                    env._freeze_rand_vec = False
                    env = NormalizedEnvWrapper(env, normalize_reward=should_perform_reward_normalization)
                    return env

        return _make_env

    if num_copies_per_env > 1:
        env_id_list = [
            [env_id for _ in range(num_copies_per_env)] for env_id in env_id_list
        ]
        env_id_list = [
            env_id for env_id_sublist in env_id_list for env_id in env_id_sublist
        ]
    print(task_idx)
    if fix_goal:
        funcs_to_make_envs = [get_func_to_make_deterministic_envs(env_id_list[0],idx) for idx in task_idx]
    else:
        funcs_to_make_envs = [get_func_to_make_stochastic_envs(env_id_list[0],idx) for idx in task_idx]
    # env = funcs_to_make_envs[0]()
    # env.reset()
    # print(env.env._get_pos_goal())
    return funcs_to_make_envs, env_id_to_task_map

def build(
    benchmark: Optional[metaworld.Benchmark],
    benchmark_name: str,
    env_id_to_task_map: Optional[EnvIdToTaskMapType],
    should_perform_reward_normalization: bool = True,
    task_name: str = "pick-place-v1",
    num_copies_per_env: int = 1,
    initial_task_state: int = 1,
) -> MTEnv:
    """Build a MTEnv comptaible variant of MetaWorld.

    Args:
        benchmark (Optional[metaworld.Benchmark]): `benchmark` to create
            tasks from.
        benchmark_name (str): name of the `benchmark`. This is used only
            when the `benchmark` is None.
        env_id_to_task_map (Optional[EnvIdToTaskMapType]): In MetaWorld,
            each environment can be associated with multiple tasks. This
            dict persists the mapping between environment ids and tasks.
        should_perform_reward_normalization (bool, optional): Defaults to
            True.
        task_name (str, optional): In case of MT1, only . Defaults to
            "pick-place-v1".
        num_copies_per_env (int, optional): Number of copies to create for
            each environment. Defaults to 1.
        initial_task_state (int, optional): initial task/environment to
            select. Defaults to 1.

    Returns:
        MTEnv:
    """
    funcs_to_make_envs, env_id_to_task_map = get_list_of_func_to_make_envs(
        benchmark=benchmark,
        benchmark_name=benchmark_name,
        env_id_to_task_map=env_id_to_task_map,
        should_perform_reward_normalization=should_perform_reward_normalization,
        task_name=task_name,
        num_copies_per_env=num_copies_per_env,
    )

    assert env_id_to_task_map is not None

    mtenv = MetaWorldMTWrapper(
        funcs_to_make_envs=funcs_to_make_envs,
        initial_task_state=initial_task_state,
        env_id_to_task_map=env_id_to_task_map,
    )
    return mtenv
