from functools import partial

import fire
import numpy as np
from scipy.sparse import csgraph
from skimage.segmentation import flood_fill

from src.maze.agents.rl_agent import RLAgent, RLAgentConfig
from src.maze.envs.maze import MazeEnv
from src.maze.level import MazeLevel, OBJ_TYPES_TO_INT
from src.maze.module import MazeModule


class LabyrinthEnv(MazeEnv):
    """A short but non-optimal path is 118 moves."""

    def __init__(self):
        # positions go col, row
        start_pos = np.array([1, 13])
        goal_pos = np.array([7, 7])
        bit_map = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                            [0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0],
                            [0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0],
                            [0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0],
                            [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
                            [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
                            [0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0],
                            [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0],
                            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0],
                            [1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0],
                            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]])
        super().__init__(size=15,
                         bit_map=bit_map,
                         start_pos=start_pos,
                         goal_pos=goal_pos)


class LargeCorridorEnv(MazeEnv):
    """A long backtracking env."""

    def __init__(self):
        # positions go col, row and indexing starts at 1
        start_pos = np.array([1, 10])
        row = np.random.choice([9, 11])
        col = np.random.choice([3, 5, 7, 9, 11, 13, 15, 17])
        goal_pos = np.array([col, row])
        bit_map = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
            [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
            [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
            [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
            [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
            [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
            [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
            [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
            [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
            [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
            [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
            [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
            [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
            [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
            [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ])
        super().__init__(size=21,
                         bit_map=bit_map,
                         start_pos=start_pos,
                         goal_pos=goal_pos)


class CustomEnv(MazeEnv):
    """Custom env."""

    def __init__(self):
        # positions go col, row and indexing starts at 1
        start_pos = np.array([9, 4])
        goal_pos = np.array([7, 8])
        bit_map = np.array([[1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1],
                            [0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0],
                            [0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1],
                            [1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0],
                            [0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0],
                            [0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1],
                            [1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1],
                            [1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0],
                            [1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                            [1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0],
                            [0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0],
                            [1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1],
                            [1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1],
                            [0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0],
                            [0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0],
                            [0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0]])
        super().__init__(size=18,
                         bit_map=bit_map,
                         start_pos=start_pos,
                         goal_pos=goal_pos)


def main():
    def get_cells(level: np.ndarray):
        """
        Computes the start and goal cells of the given `level`.
        The latter is assumed to be valid.

        Args:
            - level (np.ndarray): bitmap of a Maze.
        """
        adj = MazeModule._get_adj(level)

        dist, _predecessors = csgraph.floyd_warshall(
            adj,
            return_predecessors=True
        )
        dist[dist==np.inf] = -np.inf
        assert dist.max() >= 1
        endpoints = np.unravel_index(dist.argmax(), dist.shape)
        start_cell, end_cell = zip(*np.unravel_index(endpoints, level.shape))
        return start_cell, end_cell

    rng = np.random.default_rng(24)
    random = True
    if random:

        while True:
            level = rng.integers(2, size=(16, 16))
            # print("Generated level:")
            # print(MazeLevel(level).to_str())
            # print("Bit map:")
            # print(level.tolist())
            adj = MazeModule._get_adj(level)

            # Find the best distances
            dist, predecessors = csgraph.floyd_warshall(adj,
                                                        return_predecessors=True)
            dist[dist == np.inf] = -np.inf  # For easier argmax to find the diameter

            if dist.max() >= 1:
                # print(f"Optimal path length: {dist.max()}")
                # Label the start and the end point
                endpoints = np.unravel_index(dist.argmax(), dist.shape)
                start_cell, end_cell = zip(
                    *np.unravel_index(endpoints, level.shape))

                endpoint_level = level.copy()
                endpoint_level[start_cell] = OBJ_TYPES_TO_INT["S"]
                endpoint_level[end_cell] = OBJ_TYPES_TO_INT["G"]

                np.savetxt("level.txt", level, delimiter=',')
                break
    else:

        f = 'level_0.txt'
        level = np.loadtxt(f, delimiter=',')
        start_cell, end_cell = get_cells(level)

    # Offset start, goal to account for the added outer walls
    start_pos = (start_cell[1] + 1, start_cell[0] + 1)
    goal_pos = (end_cell[1] + 1, end_cell[0] + 1)
    print(f"Start: {start_pos}; End: {goal_pos}")
    env_func = partial(
        MazeEnv,
        size=level.shape[0] + 2,
        bit_map=level,
        start_pos=start_pos,
        goal_pos=goal_pos
    )
    # env_func = LabyrinthEnv
    # env_func = LargeCorridorEnv
    # env_func = CustomEnv
    rl_agent_conf = RLAgentConfig(
        recurrent_hidden_size=256,
        model_path="accel_seed_1/model_20000.tar"
    )
    # n_evals is the number of episodes (!!!) evaluated
    # what about the occupancy grid?

    rl_agent = RLAgent(env_func, n_evals=20, config=rl_agent_conf)
    # print('n_envs', rl_agent.n_envs)
    # print('n_evals', rl_agent.n_evals)
    # print(type(obs), obs['image'].shape, obs['x'])
    # import torch
    # actions = []
    # masks = torch.ones(1, device="cpu")
    # recurrent_hidden_states = (torch.zeros(rl_agent.n_envs,
    #                                     rl_agent.recurrent_hidden_size,
    #                                     device="cpu"),
    #                         torch.zeros(rl_agent.n_envs,
    #                                     rl_agent.recurrent_hidden_size,
    #                                     device="cpu"))
    # ref_obs = rl_agent.vec_env.reset()['image'].cpu().numpy()[0]
    # print(ref_obs)
    # from PIL import Image
    # for i in range(10):
    #     obs = rl_agent.vec_env.reset()

    #     with torch.no_grad():
    #         _, action, _, recurrent_hidden_states = rl_agent.model.act(
    #             obs, recurrent_hidden_states, masks)
    #     action = action.cpu().numpy()
    #     actions.append([a for a in action])
    #     o = obs['image'].cpu().numpy()[0]
    #     print(np.array_equal(ref_obs, o))
    #     Image.fromarray(rl_agent.vec_env.render("rgb_array")).save(f"{i}.png")

    # print(len(np.unique(actions)))
    # print(np.unique(actions))

    print(rl_agent.vec_env.envs[0].max_steps)
    rl_result = rl_agent.eval_and_track(level_shape=level.shape)
    print(rl_result.path_lengths)
    print(rl_result.failed_list)

    # grid = rl_result.aug_level.tolist()
    # print('')
    # for g in grid:
    #     print(g)
    # print('')
    return

    # objs, aug_level, n_left_turns, n_right_turns = rl_agent.eval_and_track(
    #     level_shape=level.shape,
    #     obj_type="path_length",
    #     aug_type="agent_occupancy")
    # objs, aug_level, n_left_turns, n_right_turns = rl_agent.eval_and_track(
    #     level_shape=level.shape, obj_type="fail_rate", aug_type="turns")
    # objs = rl_agent.eval_and_track(level_shape=level.shape)

    flood_fill_level = flood_fill(level, start_cell, -1, connectivity=1)
    n_reachable_cells = np.sum(flood_fill_level == -1)
    n_explored_cells = np.sum(rl_result.aug_level > 0)
    frac_explored_cells = n_explored_cells / n_reachable_cells

    print(f"Path lengths: {rl_result.path_lengths}")
    print(f"Fails: {rl_result.failed_list}")
    print(f"Left turns: {rl_result.n_left_turns}")
    print(f"Right turns: {rl_result.n_right_turns}")
    print(f"Repeated cells: {rl_result.n_repeated_cells}")
    print(f"Frac explored: {frac_explored_cells}")
    print(f"Aug shape: {rl_result.aug_level.shape}")


if __name__ == '__main__':
    fire.Fire(main)
