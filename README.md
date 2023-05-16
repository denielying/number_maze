# Number Maze

Find a path from maze start (upper-left corner cell) to the exit (lower-right corner cell, with -1 value).
The number on each cell indicate the number of cells/steps you can move from the cell once you reach the cell.
You can move either vertically or horizontally, and you need to move exactly the number of steps shown on the cell,
but can not move out of the grid. For example, in the example maze below, if you are on position (1, 2),
you can only move down for 3 steps since move to the left or right for 3 steps would move out of the grid.

|     |  1  |  2  |  3  |   4 |
|:----|:---:|:---:|:---:|----:|
| 1   |  2  |  3  |  3  |   2 |
| 2   |  2  |  1  |  2  |   2 |
| 3   |  2  |  2  |  1  |   3 |
| 4   |  3  |  3  |  3  |  -1 |
