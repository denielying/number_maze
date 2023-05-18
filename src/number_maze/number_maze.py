import typing
import numpy as np
import pandas as pd

Position = typing.Tuple[int, int]


class NumberMaze:
    """
    Class of number maze

    Parameters
    ----------
    shape: int, tuple of int, numpy array
        When given an int or tuple of int, a random maze of the size is initialized.
        When given a numpy array, the maze is initialized from the given array.

    verbose: bool
        Whether to print internal steps when initializing a random maze

    Attributes
    ----------
    maze: numpy ndarray
        the number maze grid in numpy array

    shape: (int, int)
        tuple indicating the shape of the maze

    maze_start: (int, int)
        the start position of the maze

    maze_exit: (int, int)
        the exit position of the maze

    maze_adjacency_list: dict[Position, list[Position]]
        the maze graph as an adjacency list

    all_paths:
        List of paths of all the paths starting from the maze start

    all_paths_to_exit:
        List of all the paths from maze start to exit
    """

    moving_directions = ['up', 'down', 'left', 'right']
    moving_direction_dimension = {
        _d: _i // 2 for _i, _d in enumerate(moving_directions)
    }  # {'up': 0, 'down': 0, 'left': 1, 'right': 1}
    moving_direction_forward = {
        _d: bool(_i % 2) for _i, _d in enumerate(moving_directions)
    }  # {'up': False, 'down': True, 'left': False, 'right': True}
    num_dim = len(moving_directions) // 2

    def __init__(self, shape: typing.Union[int, typing.Tuple, np.ndarray], verbose=False):
        """
        :param shape: int, tuple of int, numpy array. When given an int or tuple of int, a random maze of the size is
            initialized. When given a numpy array, the maze is initialized from the given array.
        :param verbose: whether to print internal steps when initializing a random maze
        """
        if isinstance(shape, np.ndarray):
            self.maze = shape
            self.shape = self.maze.shape
            self.maze_start = np.unravel_index(0, shape=self.shape)
            self.maze_exit = np.unravel_index(np.prod(self.shape) - 1, shape=self.shape)
            self.maze[self.maze_exit] = 0
        else:
            self.shape = shape if isinstance(shape, tuple) else tuple([shape] * self.num_dim)
            self.maze = self.random_step_num(max(self.shape) - 1, shape=self.shape)
            self.maze_start = np.unravel_index(0, shape=self.shape)
            self.maze_exit = np.unravel_index(np.prod(self.shape) - 1, shape=self.shape)
            self.maze[self.maze_exit] = 0
            init_path, init_steps = self._init_random_path(verbose=verbose)
            self._set_path_steps(self.maze, init_path, init_steps)
        self.maze_adjacency_list = self._maze_to_adjacency_list()
        self.all_paths: typing.List[typing.List[Position]] = self.find_all_paths(self.maze_adjacency_list, self.maze_start)
        self.all_paths_to_exit: typing.List[typing.List[Position]] = [p for p in self.all_paths if p[-1] == self.maze_exit]

    def in_grid(self, pos: Position) -> bool:
        invalid_dims = [not 0 <= pos[_i] < self.shape[_i] for _i in range(self.num_dim)]
        return not any(invalid_dims)

    def return_if_in_grid(self, pos) -> Position:
        if not self.in_grid(pos):
            raise RuntimeError(f"Position {pos} is out of the grid")
        return pos

    @staticmethod
    def moving_dimension_forward(direction: str):
        return NumberMaze.moving_direction_dimension[direction], NumberMaze.moving_direction_forward[direction]

    def max_steps(self, pos) -> typing.Dict[str, int]:
        """Find the max number of steps of each direction, return a dict with direction as key"""
        return dict(
            map(lambda _d: (_d, self._max_steps(*self.moving_dimension_forward(_d), pos)), self.moving_directions)
        )

    def move(self, direction: str, pos: Position, steps: int) -> Position:
        """
        Move in the specified direction in the maze from one position by n steps

        :param direction: str, direction of movement, "up", "down", "left", or "right"
        :param pos: 2d tuple, the current position in the grid maze
        :param steps: int, number of steps to move
        :return path: the new position moved to in the grid
        :return step_list: raise error if the new position is out of the grid
        """
        if (direction := direction.lower()) not in self.moving_directions:
            raise RuntimeError(f'Moving direction \'{direction}\' is not in {self.moving_directions}')

        new_pos = self._move(*self.moving_dimension_forward(direction), pos, steps)

        if not self.in_grid(new_pos):
            raise RuntimeError(f'Moving {direction} from {pos} by {steps} step(s) is not allowed')
        return new_pos

    def get_all_next_positions(self, pos: Position, steps: int) -> typing.List[Position]:
        """
        Find all the possible positions that can be moved to by n steps from the current position
        """
        position_list = []
        for _d in self.moving_directions:
            mv_dim, mv_f = self.moving_dimension_forward(_d)
            max_num_steps = self._max_steps(mv_dim, mv_f, pos)
            if max_num_steps >= steps:
                new_pos = self._move(mv_dim, mv_f, pos, steps)
                position_list.append(new_pos)

        return position_list

    def move_backward(self, direction: str, pos: Position, steps: int) -> Position:
        """
        Move in the opposite of the specified direction for n steps
        """
        d, f = self.moving_dimension_forward(direction)
        new_pos = self._move(d, not f, pos, steps)
        if not self.in_grid(new_pos):
            raise RuntimeError(
                f'Moving from {pos} by {steps} step(s) in opposite of {direction} direction is not allowed')
        return new_pos

    def maze_to_df(self):
        return pd.DataFrame(data=self.maze, index=range(self.shape[0]), columns=range(self.shape[1]))

    def all_paths_to_exit_to_str(self) -> str:
        """
        All the paths to exit to string

        :return: string representing all the paths to exist
        """
        return self.paths_to_str(self.all_paths_to_exit)

    def _max_steps(self, dim: int, forward: bool, pos) -> int:
        return (self.shape[dim] - 1 - pos[dim]) if forward else pos[dim]

    @staticmethod
    def _move(dim: int, forward: bool, pos: Position, steps: int) -> Position:
        new_pos_list = list(pos)
        new_pos_list[dim] += (steps if forward else -steps)
        return tuple(new_pos_list)

    @staticmethod
    def _set_path_steps(maze_ndarray, path: typing.List[Position], steps: typing.List[int]) -> None:
        rows, cols = zip(*path)
        maze_ndarray[rows, cols] = steps

    def _get_path_in_maze(self, path: typing.List[Position], steps: typing.List[int]) -> np.ndarray:
        m = np.zeros(self.maze.shape)
        self._set_path_steps(m, path, steps)
        return m

    def _init_random_path(self, max_path_steps: int = 0, max_step_try: int = 0, verbose=False) -> \
            typing.Tuple[typing.List[Position], typing.List[int]]:
        """
        Initial a random path form start to exist on an empty maze

        :param max_path_steps: maximum steps of the path to try, when set to 0, it uses `max(self.shape) ** 2`
        :param max_step_try: for each random step, the number of try of the next movement, for 2d maze, default is 4
        :param verbose: whether print internal steps or not
        :return: position list of the path, number of grid of each movement
        """
        if max_path_steps <= 0:
            max_path_steps = max(self.shape) ** 2
        if max_step_try <= 0:
            max_step_try = self.num_dim * 2

        path = [self.maze_start]
        step_list = []
        _i = 0
        pos = path[_i]
        dirc = ""
        while _i < max_path_steps:
            _try = 1
            find_next_pos = False
            if verbose:
                print(f"Random path position {_i + 1}:")
            while _try <= max_step_try:
                max_steps_dict: typing.Dict[str, int] = self.max_steps(pos)
                new_dirc = np.random.choice([d for d, n in max_steps_dict.items() if n > 0 and d != dirc])
                mv_steps = self.random_step_num(max_steps_dict[new_dirc])
                new_pos = self._move(*self.moving_dimension_forward(new_dirc), pos, mv_steps)
                possible_new_pos = self.get_all_next_positions(pos, mv_steps)
                if self.maze_exit in possible_new_pos:
                    new_pos = self.maze_exit
                    find_next_pos = True
                    if verbose:
                        print(f"Try {_try}: moving to maze exit {new_pos} succeeded")
                    break
                if new_pos not in path:
                    if verbose:
                        print(f"Try {_try}: moving to {new_pos} succeeded")
                    find_next_pos = True
                    break
                if verbose:
                    print(f"Try {_try}: moving to {new_pos} failed")
                _try += 1
            if find_next_pos:
                path.append(new_pos)
                step_list.append(mv_steps)
                if new_pos == self.maze_exit:
                    break
                pos = new_pos
                dirc = new_dirc
            _i += 1
        if path[-1] != self.maze_exit:
            raise RuntimeError(f"Unable to initialize a random path to the exit within {max_path_steps} steps")
        step_list.append(self.maze[self.maze_exit])
        return path, step_list

    def _maze_to_adjacency_list(self) -> typing.Dict[Position, typing.List[Position]]:
        adjacency_dict = {}
        for pos in np.ndindex(self.shape):
            adjacency_dict[pos] = self.get_all_next_positions(pos, self.maze[pos]) if self.maze[pos] > 0 else []
        return adjacency_dict

    @staticmethod
    def find_all_paths(graph, start, path=None) -> typing.List[typing.List[Position]]:
        """
        Find all paths starting from the start

        :param graph: maze graph as an adjacency list, i.e, a dict with current node as key, and list of node as its value
        :param start: the starting node
        :param path: list of nodes in the path travelled
        :return:
        """
        if path is None:
            path = []  # Initialize path for the first call

        path = path + [start]  # Append the current node to the path

        if start not in graph:
            return [path]  # If the start node is not in the graph, return the path

        paths = []  # Store all the paths from start

        for node in graph[start]:
            if node not in path:  # Avoid cycles
                new_paths = NumberMaze.find_all_paths(graph, node, path)  # Recursive call
                paths.extend(new_paths)  # Extend the current paths with new paths

        if not paths:
            paths = [path]  # Handle case where a node has no child nodes

        return paths

    @staticmethod
    def random_step_num(n, shape=None):
        k = np.random.randint(1, n * (n + 1) / 2 + 1, size=shape)
        return np.ceil((np.sqrt(8 * k + 1) - 1) / 2).astype(int)

    @staticmethod
    def paths_to_str(path_list: typing.List[typing.List[Position]]):
        return "\n".join(map(lambda path: ' -> '.join([str(p) for p in path]), path_list))
