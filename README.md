# Number Maze

Find a path from the maze start (upper-left corner cell) to the exit (lower-right corner cell, marked with -1 value).
The number on each cell indicate the number of steps you can move from the cell.  You can move either vertically or
horizontally, but you need to move exactly the number of steps shown on the cell and can not move out of the grid.
For example, in the example maze below, if you are on position (a, B),
you can only move down for 3 steps to (d, B), since move to the left or right for 3 steps would move out of the grid.

|     |  A  |  B  |  C  |  D  |
|----:|:---:|:---:|:---:|:---:|
|   a |  2  |  3  |  3  |  2  |
|   b |  2  |  1  |  2  |  2  |
|   c |  2  |  2  |  1  |  3  |
|   d |  3  |  3  |  3  | -1  |
