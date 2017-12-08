# Project Title

Sudoku Solver


## Description

The completion of the project requires the following 5 steps:
---Preprocess Image
---Extract Sudoku
---Digit Recognition and Sudoku Matrix formation
---Solve the sudoku

### Preprocess Image

### Extract Sudoku

### Digit Recognition and Sudoku Matrix formation

### Solve the sudoku
For solving the sudoku, a backtracking algorithm is used, which is a special case type of Brute Force search. 
A cell is tried with digits from 1-9 and checked for validity. If not a valid move, the next digit is placed and checked for again. If valid, then the next empty cell is searched for and filled the same way. In this way, it is a depth first search with depth N (no. of unfilled cells) and a maximum branching factor 9. Worst case performance can be O(9^N).

## Authors
Kunal Jain
