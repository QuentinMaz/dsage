# coding=utf-8
# Copyright 2021 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Code from https://github.com/ucl-dark/paired.

Implements single-agent manually generated Maze environments.

Humans provide a bit map to describe the position of walls, the starting
location of the agent, and the goal location.
"""
import logging

import gym_minigrid.minigrid as minigrid
import numpy as np

from .multigrid import MultiGridEnv, Grid

logger = logging.getLogger(__name__)


class MazeEnv(MultiGridEnv):
    """Single-agent maze environment specified via a bit map."""

    def __init__(
        self,
        agent_view_size=5,
        minigrid_mode=True,
        max_steps=None,
        bit_map=None,
        start_pos=None,
        goal_pos=None,
        size=15,
    ):
        default_agent_start_x = 7
        default_agent_start_y = 1
        default_goal_start_x = 7
        default_goal_start_y = 13
        self.start_pos = (np.array([
            default_agent_start_x, default_agent_start_y
        ]) if start_pos is None else start_pos)
        self.goal_pos = ((default_goal_start_x, default_goal_start_y)
                         if goal_pos is None else goal_pos)

        if max_steps is None:
            max_steps = 2 * size * size

        if bit_map is not None:
            bit_map = np.array(bit_map)
            if bit_map.shape != (size - 2, size - 2):
                logger.warning(
                    "Error! Bit map shape does not match size. Using default maze."
                )
                bit_map = None

        if bit_map is None:
            self.bit_map = np.array([
                [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
                [0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0],
                [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0],
                [0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1],
                [0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0],
                [1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0],
                [1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
                [1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0],
                [0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0],
            ])
        else:
            self.bit_map = bit_map

        super().__init__(
            n_agents=1,
            grid_size=size,
            agent_view_size=agent_view_size,
            max_steps=max_steps,
            see_through_walls=True,  # Set this to True for maximum speed
            minigrid_mode=minigrid_mode,
        )

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Goal
        self.put_obj(minigrid.Goal(), self.goal_pos[0], self.goal_pos[1])

        # Agent
        self.place_agent_at_pos(0, self.start_pos)

        # Walls
        for x in range(self.bit_map.shape[0]):
            for y in range(self.bit_map.shape[1]):
                if self.bit_map[y, x]:
                    # Add an offset of 1 for the outer walls
                    self.put_obj(minigrid.Wall(), x + 1, y + 1)


if __name__=="__main__":
    from src.maze.level import MazeLevel
    from src.maze.level import MazeLevel, OBJ_TYPES_TO_INT
    from src.maze.module import MazeModule

    from scipy.sparse import csgraph
    from skimage.segmentation import flood_fill

    rng = np.random.default_rng(0)
    num_attempts = 0
    while True:
        num_attempts += 1
        level = rng.integers(2, size=(16, 16))
        adj = MazeModule._get_adj(level)
        # Find the best distances
        dist, predecessors = csgraph.floyd_warshall(adj,
                                                    return_predecessors=True)
        dist[dist == np.inf] = -np.inf  # For easier argmax to find the diameter

        if dist.max() >= 1:
            print(f"Optimal path length: {dist.max()}")
            # Label the start and the end point
            endpoints = np.unravel_index(dist.argmax(), dist.shape)
            start_cell, end_cell = zip(
                *np.unravel_index(endpoints, level.shape))

            endpoint_level = level.copy()
            endpoint_level[start_cell] = OBJ_TYPES_TO_INT["S"]
            endpoint_level[end_cell] = OBJ_TYPES_TO_INT["G"]

            # bonus: labels the path between start and goal
            path_level = level.copy()
            path_level[start_cell] = OBJ_TYPES_TO_INT["S"]
            path_level[end_cell] = OBJ_TYPES_TO_INT["G"]
            cur_cell_n = endpoints[0]
            end_cell_n = endpoints[1]
            while True:
                cur_cell_n = predecessors[end_cell_n, cur_cell_n]
                if cur_cell_n == end_cell_n:
                    break
                cur_cell = np.unravel_index(cur_cell_n, level.shape)
                path_level[cur_cell] = OBJ_TYPES_TO_INT["P"]

            break
        else:
            print(f"Attempt {num_attempts} failed.")
    print(f"Total number of attempts: {num_attempts}")
    print("Raw input:")
    print(level)
    print("Generated level:")
    print(MazeLevel(level).to_str())
    print("Bit map:")
    print(level.tolist())
    # Offset start, goal to account for the added outer walls
    start_pos = (start_cell[1] + 1, start_cell[0] + 1)
    goal_pos = (end_cell[1] + 1, end_cell[0] + 1)
    print(f"Start: {start_pos}; End: {goal_pos}")
    # print("Endpoint level:")
    # print(MazeLevel(endpoint_level).to_str())
    # print("Path:")
    # print(MazeLevel(path_level).to_str())
    env = MazeEnv(
        agent_view_size=5,
        size=level.shape[0] + 2,
        bit_map=level,
        start_pos=start_pos,
        goal_pos=goal_pos
    )
