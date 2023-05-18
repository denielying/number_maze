import numpy as np

from number_maze import NumberMaze

if __name__ == '__main__':
    # --------------------------------------------
    # create a maze from an 2D array
    maze_array = np.array([
        [2, 2, 2, 3],
        [3, 3, 3, 2],
        [3, 1, 1, 3],
        [3, 3, 3, 0]
    ])
    maze = NumberMaze(maze_array)

    print("Number Maze:")
    print(maze.maze_to_df())
    print()
    print("Path to exit:")
    print(maze.all_paths_to_exit_to_str())

    # --------------------------------------------
    # randomly create a maze of a given size
    rand_maze = NumberMaze(shape=5)

    print("Number Maze:")
    print(rand_maze.maze_to_df())
    print()
    print("Path to exit:")
    print(rand_maze.all_paths_to_exit_to_str())




