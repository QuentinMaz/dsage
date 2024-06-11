from functools import partial
from typing import Tuple
import numpy as np
import torch
import pickle
from stable_baselines3.common.vec_env import DummyVecEnv

from src.maze.envs.maze import MazeEnv
from src.maze.envs.wrappers import VecMonitor, VecPreprocessImageWrapper
from src.maze.agents.multigrid_network import MultigridNetwork
from src.maze.agents.rl_agent import RLAgentConfig, TestResult
from src.maze.module import MazeModule

from scipy.sparse import csgraph


def save_pickle(title: str, data):
    pikd = open(title + ".pickle", "wb")
    pickle.dump(data, pikd)
    pikd.close()


def load_pickle(title: str):
    pikd = open(title + ".pickle", "rb")
    data = pickle.load(pikd)
    pikd.close()
    return data


class MazeAgentTester:
    """My base class for agents solving mazes.
    Args:
        - num_evals (int): Number of evaluations for a single maze (stochastic executions).
        - config (RLAgentConfig): See `RLAgentConfig` for required configs.
        - max_env_steps (int): Maximum number of steps to solve a maze. (default: 648)
    """

    def __init__(self, num_evals: int, config: RLAgentConfig, max_env_steps: int = 648):
        self.num_evals = num_evals
        self.recurrent_hidden_size = config.recurrent_hidden_size
        self.n_envs = config.n_envs
        self.max_steps = max_env_steps

        env_func = partial(MazeEnv)
        env_fns = [env_func for _ in range(self.n_envs)]
        self.vec_env = DummyVecEnv(env_fns)

        self.vec_env = VecMonitor(
            venv=self.vec_env,
            filename=None,
            keep_buf=100
        )

        self.vec_env = VecPreprocessImageWrapper(
            venv=self.vec_env,
            obs_key="image",
            transpose_order=[2, 0, 1],
            scale=10.0,
            device="cpu"
        )

        num_directions = self.vec_env.observation_space["direction"].high[0] + 1
        self.model = MultigridNetwork(
            observation_space=self.vec_env.observation_space,
            action_space=self.vec_env.action_space,
            scalar_dim=num_directions,
            recurrent_hidden_size=config.recurrent_hidden_size,
        )

        model_path = config.model_path
        checkpoint = torch.load(model_path, map_location="cpu")
        self.model.load_state_dict(checkpoint)


    def generate_level(self, rng: np.random.Generator) -> np.ndarray:
        """Returns a level as a Numpy array."""
        return rng.integers(2, size=(16, 16))


    def compute_positions(self, level: np.ndarray) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """
        Computes the start and goal positions for a given `level`.

        Returns:
            The start and goal positions or (None, None) if the level is not valid.
        """
        adj = MazeModule._get_adj(level)
        dist, _predecessors = csgraph.floyd_warshall(
            adj,
            return_predecessors=True
        )
        dist[dist == np.inf] = -np.inf

        if dist.max() >= 1:
            # print(f"Optimal path length: {dist.max()}")
            # labels the start and the end point
            endpoints = np.unravel_index(dist.argmax(), dist.shape)
            start_cell, end_cell = zip(*np.unravel_index(endpoints, level.shape))

            # endpoint_level = level.copy()
            # endpoint_level[start_cell] = OBJ_TYPES_TO_INT["S"]
            # endpoint_level[end_cell] = OBJ_TYPES_TO_INT["G"]

            # bonus: labels the path between start and goal
            # path_level = level.copy()
            # path_level[start_cell] = OBJ_TYPES_TO_INT["S"]
            # path_level[end_cell] = OBJ_TYPES_TO_INT["G"]
            # cur_cell_n = endpoints[0]
            # end_cell_n = endpoints[1]
            # while True:
            #     cur_cell_n = predecessors[end_cell_n, cur_cell_n]
            #     if cur_cell_n == end_cell_n:
            #         break
            #     cur_cell = np.unravel_index(cur_cell_n, level.shape)
            #     path_level[cur_cell] = OBJ_TYPES_TO_INT["P"]
            start_pos = (start_cell[1] + 1, start_cell[0] + 1)
            goal_pos = (end_cell[1] + 1, end_cell[0] + 1)
            return start_pos, goal_pos
        else:
            # raise ValueError("Level is not valid.")
            return None, None


    def generate_maze(self, rng: np.random.Generator, num_attempts: int = 10) -> MazeEnv:
        """Attempts to create a random MazeEnv.

        Args:
            - num_attempts: Number of attempts. (default: 10)

        Returns:
            None if the level turns out to be invalid
        """
        success = False

        for _ in range(num_attempts):
            level = self.generate_level(rng)
            start_pos, goal_pos = self.compute_positions(level)
            if (start_pos is not None) and (goal_pos is not None):
                success = True
                break

        if success:
            return MazeEnv(
                agent_view_size=5,
                size=(level.shape[0] + 2),
                bit_map=level,
                start_pos=start_pos,
                goal_pos=goal_pos,
                max_steps=self.max_steps
            )
        else:
            return None


    def get_vec_env(
        self,
        level: np.ndarray,
        start_pos: Tuple[int, int],
        goal_pos: Tuple[int, int],
    ) -> VecPreprocessImageWrapper:
        env_func = partial(
            MazeEnv,
            size=level.shape[0] + 2,
            bit_map=level,
            start_pos=start_pos,
            goal_pos=goal_pos,
            max_steps=self.max_steps
        )
        env_fns = [env_func for _ in range(self.n_envs)]
        vec_env = DummyVecEnv(env_fns)
        vec_env = VecMonitor(
            venv=vec_env,
            filename=None,
            keep_buf=100
        )
        return VecPreprocessImageWrapper(
            venv=vec_env,
            obs_key="image",
            transpose_order=[2, 0, 1],
            scale=10.0,
            device="cpu"
        )

    def eval_input_test(
            self,
            level: np.ndarray,
            start_pos: Tuple[int, int],
            goal_pos: Tuple[int, int],
            ) -> TestResult:
        """
        Evaluate the agent on a particular maze.

        Args:
            - level: The level for setting the environment.
            - start_pos: Start position.
            - goal_pos: Goal position.

        Returns:
            A custom object containing the occupancy grid and the (`self.num_evals`) episodes' lengths, failures and rewards.
        """
        level_shape = level.shape

        returns = []
        path_lengths = []
        n_left_turns = []
        n_right_turns = []
        failed_list = []
        reward_list = []

        # recreates the environment
        self.vec_env = self.get_vec_env(level, start_pos, goal_pos)
        obs = self.vec_env.reset()

        recurrent_hidden_states = (
            torch.zeros(
                self.n_envs,
                self.recurrent_hidden_size,
                device="cpu"
                ),
            torch.zeros(self.n_envs,
                self.recurrent_hidden_size,
                device="cpu"
                )
        )

        masks = torch.ones(1, device="cpu")

        aug_level = np.zeros(shape=level_shape)

        left_turns = np.zeros(self.n_envs)
        right_turns = np.zeros(self.n_envs)
        rewards = np.zeros(self.n_envs)
        aug_level_ind = np.zeros((self.n_envs, *level_shape))
        num_repeated_cells = 0
        while len(returns) < self.num_evals:
            xs = obs.get("x").detach().cpu().numpy().astype(int)
            ys = obs.get("y").detach().cpu().numpy().astype(int)
            with torch.no_grad():
                _, action, _, recurrent_hidden_states = self.model.act(
                    obs, recurrent_hidden_states, masks)

            # Observe reward and next obs
            action = action.cpu().numpy()
            left_turns += (action == self.vec_env.envs[0].actions.left
                          ).flatten().astype(int)
            right_turns += (action == self.vec_env.envs[0].actions.right
                           ).flatten().astype(int)

            for i, (x, y) in enumerate(zip(xs, ys)):
                aug_level[y - 1,
                            x - 1] += 1  # Offset due to added outer walls
                if aug_level_ind[i, y - 1, x - 1]:
                    num_repeated_cells += 1
                else:
                    aug_level_ind[i, y - 1, x - 1] = 1

            obs, reward, done, infos = self.vec_env.step(action)

            # print("rewards:", reward.cpu().numpy().flatten())
            rewards += reward.cpu().numpy().flatten()
            # print("Acc. rewards:", rewards)
            # input()

            masks = torch.tensor(
                [[0.0] if done_ else [1.0] for done_ in done],
                dtype=torch.float32,
                device="cpu",
            )

            for i, info in enumerate(infos):
                # episode `i` terminates
                if "episode" in info.keys():
                    # print(f"Env {i} has terminated with acc. reward of {rewards[i]} and length of {info['episode']['l']}.")
                    returns.append(info["episode"]["r"])
                    path_lengths.append(info["episode"]["l"])
                    n_left_turns.append(left_turns[i])
                    n_right_turns.append(right_turns[i])
                    reward_list.append(rewards[i])
                    if returns[-1] > 0:
                        failed_list.append(0)
                    else:
                        failed_list.append(1)

                    # zero hidden states
                    recurrent_hidden_states[0][i].zero_()
                    recurrent_hidden_states[1][i].zero_()
                    left_turns[i] = 0
                    right_turns[i] = 0
                    rewards[i] = 0
                    # the shape is preserved
                    aug_level_ind[i] = 0

                    if len(returns) >= self.num_evals:
                        break

        # average number of times the agent has come back to already visited cells (during the episodes)
        num_repeated_cells /= self.num_evals
        aug_level /= self.num_evals
        return TestResult(
            np.array(path_lengths),
            np.array(failed_list),
            np.array(reward_list),
            aug_level,
            # np.array(n_left_turns),
            # np.array(n_right_turns),
            # n_repeated_cells
        )

if __name__=="__main__":
    import tqdm
    import argparse

    parser = argparse.ArgumentParser()
    # parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("outputpath", type=str, help="Path of the pickle to record the levels and grids")
    parser.add_argument("-n",  type=int, default=10_000, help="Number of levels to generate and execute")
    parser.add_argument("--seed", default=0, help="Seed for generating levels")
    args = parser.parse_args()

    model_path = "src/maze/agents/saved_models/accel_seed_1/model_20000.tar"
    tester = MazeAgentTester(
        num_evals=20,
        config=RLAgentConfig(recurrent_hidden_size=256, model_path=model_path)
    )

    rng = np.random.default_rng(args.seed)

    n = args.n
    path = args.outputpath

    i = 0
    pbar = tqdm.tqdm(total=n)
    lvls, aug_lvls = [], []
    num_failed_generations = 0
    while i < n:
        lvl = tester.generate_level(rng)
        s_pos, g_pos = tester.compute_positions(lvl)
        if (s_pos is not None) and (g_pos is not None):
            aug_lvl = tester.eval_input_test(lvl, s_pos, g_pos).aug_level
            # aug_lvl /= np.sum(aug_lvl)
            lvls.append(lvl)
            aug_lvls.append(aug_lvl)
            pbar.update(1)
            i += 1
        else:
            num_failed_generations += 1
            pbar.set_postfix_str(f"Failed Gen: {num_failed_generations:02d}")
    pbar.close()
    save_pickle(path + "_lvls", lvls)
    save_pickle(path + "_aug_lvls", aug_lvls)
