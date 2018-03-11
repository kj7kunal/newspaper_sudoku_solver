#!/usr/bin/env python

class Sudoku:
	'''
	Class for Sudoku
	Constructor takes a sudoku 2D list, with blank values as 0s.
	Backtracking - DFS approach
	'''

	def __init__(self, Valar):
		self.valar = Valar
		self.cursor = [0,0]

	def findNextEmpty(self):
		# Finds the next empty value and puts the cursor on that value.
		for i in range(0,9):
			for j in range(0,9):
				if self.valar[i][j]==0:
					self.cursor = [i,j]
					return True
		return False

	def rowCheck(self,num,row):
		# Checks row condtion
		for i in range(9):
			if self.valar[row][i] == num:
				return True 
		return False

	def colCheck(self,num,col):
		# Checks column condition
		for i in range(9):
			if self.valar[i][col] == num:
				return True 
		return False

	def boxCheck(self,num,row,col):
		# Checks box condition
		for i in range(row-row%3,row-row%3+3):
			for j in range(col-col%3,col-col%3+3):
				if self.valar[i][j] == num:
					return True 
		return False

	def solve(self):

		# Check if another empty cell is available or not.
		if not self.findNextEmpty():
			return True
		# Extract row and column of the current blank (cursor)
		[row,col] = self.cursor
	    # Put possible cell value
		for num in range(1,10):

		    if not (self.colCheck(num,col) or self.rowCheck(num,row) or self.boxCheck(num,row,col)):
		        self.valar[row][col] = num

		        # Try to solve using the possible value
		        if(self.solve()):
		        	return True
		        self.valar[row][col] = 0

		return False

	def show(self):
		# Prints the current sudoku
		for i in range(9):
			for j in range(9):
				print self.valar[i][j],
			print ''
		print '\n'

	def return_solved(self):
		# Returns the solved sudoku
		if self.solve():
			return S.valar
		raise("Sudoku cannot be solved!!!")



# if __name__=="__main__":
     
#     grid=[[3,0,6,5,0,8,4,0,0],
#           [5,2,0,0,0,0,0,0,0],
#           [0,8,7,0,0,0,0,3,1],
#           [0,0,3,0,1,0,0,8,0],
#           [9,0,0,8,6,3,0,0,5],
#           [0,5,0,0,9,0,6,0,0],
#           [1,3,0,0,0,0,2,5,0],
#           [0,0,0,0,0,0,0,7,4],
#           [0,0,5,2,0,6,3,0,0]]

#     S = Sudoku(grid)
#     S.show()

#     if(S.solve()):
#         S.show()
#     else:
#         print "No solution exists"





