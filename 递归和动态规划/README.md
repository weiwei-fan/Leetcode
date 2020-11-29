## 139. Word Break 
Given a non-empty string s and a dictionary wordDict containing a list of non-empty words, determine if s can be segmented into a space-separated sequence of one or more dictionary words.


Note:</br>
The same word in the dictionary may be reused multiple times in the segmentation.</br>
You may assume the dictionary does not contain duplicate words.</br>

Tags: Medium, DFS, DP

Approach 1: DFS
```python
from collections import deque

class Solution:
    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        stack = []
        visited = []
        for word in wordDict:
            if s.startswith(word):
                stack.append(s[len(word):])
                visited.append(s[len(word):])
        while stack:
            substring = stack.pop()
            
            if not substring:
                return True
            
            for word in wordDict:
                if substring.startswith(word) and substring[len(word):] not in visited:
                    stack.append(substring[len(word):])
                    visited.append(substring[len(word):])
        return False
```

Approach 2: DP(recursion with memo)
```python
from collections import deque

class Solution:
    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        wordSet = set(wordDict)
        memo = {}
        
        def dfs(s):
            if not s:
                return True
            if s in memo:
                return memo[s]
            memo[s] = False
            for i in range(1, len(s) + 1):
                word = s[:i]
                if word in wordSet:
                    if dfs(s[len(word):]):
                        memo[s] = dfs(s[len(word):])  
                        return memo[s]
            return memo[s]
        return dfs(s)
```
## 140. Word Break II
Given a non-empty string s and a dictionary wordDict containing a list of non-empty words, add spaces in s to construct a sentence where each word is a valid dictionary word. Return all such possible sentences.


Note:</br>
The same word in the dictionary may be reused multiple times in the segmentation.</br>
You may assume the dictionary does not contain duplicate words.</br>

Tags: Hard, DP

Approach: DP(recursion with memo)
```python
from collections import deque

class Solution:
    def wordBreak(self, s: str, wordDict: List[str]) -> List[str]:
        wordSet = set(wordDict)
        memo = {}
        
        def helper(s):
            if not s:
                return [[]]
            if s in memo:
                return memo[s]
            memo[s] = []
            for i in range(1, len(s) + 1):
                word = s[:i]
                if word in wordSet:
                    for each in helper(s[len(word):]):
                        memo[s].append([word] + each)
            return memo[s]
        
        return [" ".join(each) for each in helper(s)]
```
### 小技巧总结
```python
遍历word : wordDict，并且用s.startswith(word)是非常慢的，
有个小技巧就是遍历s能切分的每个单词s[:i] for i in range(1, len(s) + 1)，判断s[:i]是否在wordSet里面
```
### 62. Unique Paths
A robot is located at the top-left corner of a m x n grid (marked 'Start' in the diagram below).


The robot can only move either down or right at any point in time. The robot is trying to reach the bottom-right corner of the grid (marked 'Finish' in the diagram below).


How many possible unique paths are there?


Tags：Medium, DP, backtracking


Approach 1: Backtracking
```python
class Solution:
    def uniquePaths(self, m: int, n: int) -> int:
        
        # start to target all possible path  -->  backtracking
        # the selections of each pos is right and down
        # Time Limit Exceeded
        
        def backtrack(pos, path, paths, m, n):
            if pos[0] == n - 1 and pos[1] == m - 1:
                paths.append(path)
            
            # right
            if pos[0] + 1 < n:
                next_pos = (pos[0] + 1, pos[1])
                path.append(next_pos)
                backtrack(next_pos, path, paths, m, n)
                path.pop()
            
            # down
            if pos[1] + 1 < m:
                next_pos = (pos[0], pos[1] + 1)
                path.append(next_pos)
                backtrack(next_pos, path, paths, m, n)
                path.pop()
        
        paths, paths = [], []    
        backtrack((0, 0), path, paths, m, n)
        return len(paths)
```
Approach 2: DP
```python
        # DP: dp(row, col) = dp(row - 1, col) + dp(row, col - 1)
        
        def dp(target, memo):
            if target in memo:
                return memo[target]
            if target == (0, 0):
                return 1
            value = 0
            # upper
            if target[1] - 1 >= 0:
                value += dp((target[0], target[1] - 1), memo)
            # left
            if target[0] - 1 >= 0:
                value += dp((target[0] - 1, target[1]), memo)
            
            memo[target] = value
            return value
        
        memo = {}
        return dp((m - 1, n - 1), memo)
```

### 63. Unique Paths II
A robot is located at the top-left corner of a m x n grid (marked 'Start' in the diagram below).


The robot can only move either down or right at any point in time. The robot is trying to reach the bottom-right corner of the grid (marked 'Finish' in the diagram below).


Now consider if some obstacles are added to the grids. How many unique paths would there be?


An obstacle and space is marked as 1 and 0 respectively in the grid.


Tags: Medium, DP

```python
class Solution:
    def uniquePathsWithObstacles(self, obstacleGrid: List[List[int]]) -> int:
        def dp(target, memo, obstacleGrid):
            if target in memo:
                return memo[target]
            if target == (0, 0):
                return 1
            value = 0
            # upper
            if target[1] - 1 >= 0 and obstacleGrid[target[0]][target[1] - 1] == 0:
                value += dp((target[0], target[1] - 1), memo, obstacleGrid)
            # left
            if target[0] - 1 >= 0 and obstacleGrid[target[0] - 1][target[1]] == 0:
                value += dp((target[0] - 1, target[1]), memo, obstacleGrid)
            
            memo[target] = value
            return value
        
        memo = {}
        m, n = len(obstacleGrid), len(obstacleGrid[0])
        if obstacleGrid[-1][-1] == 1:
            return 0
        return dp((m - 1, n - 1), memo, obstacleGrid)
        
```
### 263. Ugly Number
Write a program to check whether a given number is an ugly number.

Ugly numbers are positive numbers whose prime factors only include 2, 3, 5.

Tags: Easy, Math

```python
class Solution:
    def isUgly(self, num: int) -> bool:
        if num <= 0:
            return False
        for i in [2, 3, 5]:
            while num % i == 0:
                num = num // i
        return True if num == 1 else False
```

### 264. Ugly Number II
Write a program to find the n-th ugly number.


Ugly numbers are positive numbers whose prime factors only include 2, 3, 5. 


Tags: Medium, Heap, DP


Approach 1: Heap
```python
import heapq

class Solution:
    def nthUglyNumber(self, n: int) -> int:
        heap = [1]
        res, visited = set(), set()
        visited.add(1)
        while heap:
            cur = heapq.heappop(heap)
            res.add(cur)
            if len(res) == n:
                break
            for i in [2, 3, 5]:
                if i * cur not in visited:
                    heap.append(i * cur)
                    visited.add(i * cur)
            heapq.heapify(heap)
        lis = list(res)
        lis.sort()
        return lis[-1]
```
Approach 2: DP
```python
import heapq

class Solution:
    def nthUglyNumber(self, n: int) -> int:
        nums = [1]
        i2, i3, i5 = 0, 0, 0
        for i in range(1, 1690):
            num = min(nums[i2] * 2, nums[i3] * 3, nums[i5] * 5)
            nums.append(num)
            if num == nums[i2] * 2:
                i2 += 1
            if num == nums[i3] * 3:
                i3 += 1
            if num == nums[i5] * 5:
                i5 += 1
        return nums[n - 1]
```
130. Surrounded Regions
Given a 2D board containing 'X' and 'O' (the letter O), capture all regions surrounded by 'X'.


A region is captured by flipping all 'O's into 'X's in that surrounded region.


Surrounded regions shouldn’t be on the border, which means that any 'O' on the border of the board are not flipped to 'X'. Any 'O' that is not on the border and it is not connected to an 'O' on the border will be flipped to 'X'. Two cells are connected if they are adjacent cells connected horizontally or vertically.


Tags: Medium, DFS, BFS

Approach 1: DFS
```python
class Solution:
    def dfs(self, board, row, col):
        if board[row][col] != 'O':
            return
        board[row][col] = 'E'
        if col < len(board[0]) - 1:
            self.dfs(board, row, col + 1)
        if row < len(board) - 1:
            self.dfs(board, row + 1, col)
        if col > 0:
            self.dfs(board, row, col - 1)
        if row > 0:
            self.dfs(board, row - 1, col)
        
    def solve(self, board: List[List[str]]) -> None:
        if not board or not board[0]:
            return
        
        # retrieve all border cells
        borders = []
        for row in range(len(board)):
            for col in range(len(board[0])):
                if row == 0 or col == 0 or row == len(board) - 1 or col == len(board[0]) - 1:
                    if board[row][col] == 'O':
                        borders.append((row, col))
        
        # mark the escaped cells with 'E'
        for row, col in borders:
            self.dfs(board, row, col)
        
        
        # flip the captured cells and recover escaped ones
        for row in range(len(board)):
            for col in range(len(board[0])):
                if board[row][col] == 'O':
                    board[row][col] = 'X'
                elif board[row][col] == 'E':
                    board[row][col] = 'O'
```

Approach 2: BFS
```python
from collections import deque

class Solution:
    def bfs(self, board, row, col):
        queue = deque()
        visited = set()
        queue.append((row, col))
        visited.add((row, col))
        while queue:
            (row, col) = queue.popleft()
            if board[row][col] != 'O':
                continue
            board[row][col] = 'E'
            if col < len(board[0]) - 1:
                queue.append((row, col + 1))
            if row < len(board) - 1:
                queue.append((row + 1, col))
            if col > 0:
                queue.append((row, col - 1))
            if row > 0:
                queue.append((row - 1, col))
        
    def solve(self, board: List[List[str]]) -> None:
        if not board or not board[0]:
            return
        
        # retrieve all border cells
        borders = []
        for row in range(len(board)):
            for col in range(len(board[0])):
                if row == 0 or col == 0 or row == len(board) - 1 or col == len(board[0]) - 1:
                    if board[row][col] == 'O':
                        borders.append((row, col))
        
        # mark the escaped cells with 'E'
        for row, col in borders:
            self.bfs(board, row, col)     
        
        # flip the captured cells and recover escaped ones
        for row in range(len(board)):
            for col in range(len(board[0])):
                if board[row][col] == 'O':
                    board[row][col] = 'X'
                elif board[row][col] == 'E':
                    board[row][col] = 'O'
```

### 42. Trapping Rain Water
Given n non-negative integers representing an elevation map where the width of each bar is 1, compute how much water it can trap after raining.


Tags: Hard, Two Pointer, Stack, DP


Approach 1: Brute force
```python
from collections import deque

class Solution:
    def trap(self, height: List[int]) -> int:
        res = []
        for i in range(len(height)):
            left_max, right_max = 0, 0
            for j in range(0, i):
                left_max = max(left_max, height[j])
            for k in range(i, len(height)):
                right_max = max(right_max, height[k])
            if min(left_max, right_max) - height[i] > 0:
                res.append(min(left_max, right_max) - height[i])
        return sum(res)
```

Approach 2: DP
```python
class Solution:
    def trap(self, height: List[int]) -> int:
        res = []
        left_max, right_max = [0] * len(height), [0] * len(height)
        for i in range(len(height)):
            if i == 0:
                left_max[i] = height[i]
            else:
                left_max[i] = max(left_max[i - 1], height[i])
        for i in range(len(height) - 1, -1, -1):
            if i == len(height) - 1:
                right_max[i] = height[i]
            else:
                right_max[i] = max(right_max[i + 1], height[i])
        for i in range(len(height)):
            if min(right_max[i], left_max[i]) - height[i] > 0:
                res.append(min(right_max[i], left_max[i]) - height[i])
        
        return sum(res)
```

Approach 3: Stack
```python
from collections import deque

class Solution:
    def trap(self, height: List[int]) -> int:
        stack = deque()
        res = 0
        for i in range(len(height)):
            print(stack)
            print(res)
            while stack and height[i] > height[stack[-1]]:
                top = stack.pop()
                if not stack:
                    break
                distance = i - stack[-1] - 1
                bounded_height = min(height[i], height[stack[-1]]) - height[top]
                res += distance * bounded_height
            stack.append(i)
        return res
```
Approach 4: Two pointer
```python
class Solution:
    def trap(self, height: List[int]) -> int:
        low, high = 0, len(height) - 1
        cnt = 0
        while low < high:
            if height[low] < height[high]:
                cnt += self.fill(height, low, high)
                low += 1
            elif height[low] > height[high]:
                cnt += self.fill(height, low, high)
                high -= 1
            else:
                cnt += self.fill(height, low, high)
                low += 1
                high -= 1
        return cnt
        
    def fill(self, height, low, high):
        cnt = 0
        h = min(height[low], height[high])
        for i in range(low + 1, high):
            if height[i] < h:
                cnt += (h - height[i])
                height[i] = h
        return cnt
```
### 131. Palindrome Partitioning
Given a string s, partition s such that every substring of the partition is a palindrome.


Return all possible palindrome partitioning of s.


Tags: Medium, backtracing


Approach 1: backtracing


The backtracking algorithms consists of the following steps:


Choose: Choose the potential candidate. Here, our potential candidates are all substrings that could be generated from the given string.


Constraint: Define a constraint that must be satisfied by the chosen candidate. In this case, the constraint is that the string must be a palindrome.


Goal: We must define the goal that determines if have found the required solution and we must backtrack. Here, our goal is achieved if we have reached the end of the string.
```python
class Solution:
    def dfs(self, start, res, cur, s):
        if start >= len(s):
            res.append(list(cur))
        for end in range(start, len(s)):
            if self.isPalindrome(s, start, end):
                cur.append(s[start: end + 1])
                self.dfs(end + 1, res, cur, s)
                cur.pop()
                
    def isPalindrome(self, s, low, high):
            while low <= high:
                if s[low] != s[high]:
                    return False
                low += 1
                high -= 1     
            return True
        
    def partition(self, s: str) -> List[List[str]]:
        cur, res = [], []
        self.dfs(0, res, cur, s)
        return res 
```
