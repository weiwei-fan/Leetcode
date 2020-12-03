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
Approach 2: Backtracing + DP
we are repeatedly iterating over the same substring multiple times and the result is always the same. There are Overlapping Subproblems and we could further optimize the approach by using dynamic programming to determine if a string is a palindrome in constant time. 
```python
class Solution:
    def dfs(self, start, res, cur, s, dp):
        if start >= len(s):
            res.append(list(cur))
        for end in range(start, len(s)):
            if s[start] == s[end] and (end - start <= 1 or dp[start + 1][end - 1]):
                dp[start][end] = True
                cur.append(s[start: end + 1])
                self.dfs(end + 1, res, cur, s, dp)
                cur.pop()
        
    def partition(self, s: str) -> List[List[str]]:
        cur, res = [], []
        dp = [[False for i in range(len(s))] for j in range(len(s))]
        self.dfs(0, res, cur, s, dp)
        return res 
```
### 523. Continuous Subarray Sum
Given a list of non-negative numbers and a target integer k, write a function to check if the array has a continuous subarray of size at least 2 that sums up to a multiple of k, that is, sums up to n*k where n is also an integer.


Tags: Medium, DP


Approach 1: DP
```
累计求和的问题可以先用O(n)的时间把从第一个数到index的数先计算好保存下来summ[i]
每次要算部分和的时候可以用减法summ[j] - summ[i] + nums[i]求出i-j的sum
```
```python
class Solution:
    def checkSubarraySum(self, nums: List[int], k: int) -> bool:
        summ = []
        summ.append(nums[0])
        for i in range(1, len(nums)):
            summ.append(summ[-1] + nums[i])
        for i in range(len(nums) - 1):
            for j in range(i + 1, len(nums)):
                cur = summ[j] - summ[i] + nums[i]
                if cur == k or (k and cur % k == 0):
                    return True
        return False
```
### 200. Number of Islands
Given an m x n 2d grid map of '1's (land) and '0's (water), return the number of islands.


An island is surrounded by water and is formed by connecting adjacent lands horizontally or vertically. You may assume all four edges of the grid are all surrounded by water.


Tags: Medium, BFS, DFS


#### Approach 1: BFS
```python
from collections import deque

class Solution:
    def bfs(self, grid, row, col):
        queue = deque()
        queue.append((row, col))
        grid[row][col] = "2"
        while queue:
            r, c = queue.popleft()
            if r + 1 < len(grid) and grid[r + 1][c] == "1":
                grid[r + 1][c] = "2"
                queue.append((r + 1, c))
            if c + 1 < len(grid[0]) and grid[r][c + 1] == "1":
                grid[r][c + 1] = "2"
                queue.append((r, c + 1))
            if r - 1 >= 0 and grid[r - 1][c] == "1":
                grid[r - 1][c] = "2"
                queue.append((r - 1, c))
            if c - 1 >= 0 and grid[r][c - 1] == "1":
                grid[r][c - 1] = "2"
                queue.append((r, c - 1))
        
    def numIslands(self, grid: List[List[str]]) -> int:
        cnt = 0
        for row in range(len(grid)):
            for col in range(len(grid[0])):
                if grid[row][col] == "1":
                    cnt += 1
                    self.bfs(grid, row, col)
        return cnt
```
### 907. Sum of Subarray Minimums
Given an array of integers A, find the sum of min(B), where B ranges over every (contiguous) subarray of A.


Since the answer may be large, return the answer modulo 10^9 + 7.


Tags: Medium, Stack


Approach 1: Prev and Next Less Element
```python
class Solution:
    def sumSubarrayMins(self, arr: List[int]) -> int:
        left = [1] * len(arr)
        stack = [(arr[0], 1)]
        for i in range(1, len(arr)):
            while stack and arr[i] <= stack[-1][0]:
                left[i] += stack.pop()[1]
            stack.append((arr[i], left[i]))
        right = [1] * len(arr)
        stack = [(arr[-1], 1)]
        for i in range(len(arr) - 2, -1, -1):
            while stack and arr[i] <= stack[-1][0]:
                right[i] += stack.pop()[1]
            stack.append((arr[i], right[i])) 
        res = 0
        for i in range(len(arr)):
            res += arr[i] * left[i] * right[i]
        return res % (10 ** 9 + 7)
```
Approach 2: Stack
```python
class Solution:
    def sumSubarrayMins(self, arr: List[int]) -> int:
        MOD = 10 ** 9 + 7
        stack = []
        res = cur = 0
        for index, value in enumerate(arr):
            count = 1
            while stack and stack[-1][0] >= value:
                val, cnt = stack.pop()
                count += cnt
                cur -= val * cnt
            stack.append((value, count))
            cur += value * count
            res += cur
        return res % MOD
```
### 39. Combination Sum
Given an array of distinct integers candidates and a target integer target, return a list of all unique combinations of candidates where the chosen numbers sum to target. You may return the combinations in any order.


The same number may be chosen from candidates an unlimited number of times. Two combinations are unique if the frequency of at least one of the chosen numbers is different.


It is guaranteed that the number of unique combinations that sum up to target is less than 150 combinations for the given input.


Tags: Medium, BackTracing
```python
class Solution:
    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        ret = []
        self.dfs(candidates, target, [], ret)
        return ret
    
    def dfs(self, nums, target, path, ret):
        if target < 0:
            return 
        if target == 0:
            ret.append(list(path))
            return 
        for i in range(len(nums)):
            path.append(nums[i])
            self.dfs(nums[i:], target-nums[i], path, ret)
            path.pop()
```

### 40. Combination Sum II
Given a collection of candidate numbers (candidates) and a target number (target), find all unique combinations in candidates where the candidate numbers sum to target.


Each number in candidates may only be used once in the combination.


Note: The solution set must not contain duplicate combinations.


Tags: Meduim, BackTracing(sort + skip repeating)
```python
class Solution:
    def combinationSum2(self, candidates: List[int], target: int) -> List[List[int]]:
        res = []
        candidates = sorted(candidates)
        self.backtrack(candidates, target, [], res)
        return res
    
    def backtrack(self, candidates, target, cur, res):
        if target < 0:
            return
        if target == 0:
            res.append(list(cur))
            return
        for i in range(len(candidates)):
            if i > 0 and candidates[i] == candidates[i - 1]:
                continue
            cur.append(candidates[i])
            self.backtrack(candidates[i + 1:], target - candidates[i], cur, res)
            cur.pop()
```

### 64. Minimum Path Sum
Given a m x n grid filled with non-negative numbers, find a path from top left to bottom right, which minimizes the sum of all numbers along its path.


Note: You can only move either down or right at any point in time.


Tags: Medium, BackTracing, DP


#### Approach 1: BackTracing[TLE]
```python
class Solution:
    def minPathSum(self, grid: List[List[int]]) -> int:
        def backtrack(row, col, cur_len, min_len):
            if row == len(grid) - 1 and col == len(grid[0]) - 1:
                cur_len += grid[row][col]
                min_len[0] = min(cur_len, min_len[0])
                return

            if row + 1 < len(grid):
                cur_len += grid[row][col]
                backtrack(row + 1, col, cur_len, min_len)
                cur_len -= grid[row][col]

            if col + 1 < len(grid[0]):
                cur_len += grid[row][col]
                backtrack(row, col + 1, cur_len, min_len)
                cur_len -= grid[row][col]
        
        min_len = [float('inf')]
        backtrack(0, 0, 0, min_len)
        return min_len[0]
```

#### Approach 2: DP
```python
class Solution:
    def minPathSum(self, grid: List[List[int]]) -> int:
        m = len(grid)
        n = len(grid[0])
        for i in range(1, n):
            grid[0][i] += grid[0][i-1]
        for i in range(1, m):
            grid[i][0] += grid[i-1][0]
        for i in range(1, m):
            for j in range(1, n):
                grid[i][j] += min(grid[i-1][j], grid[i][j-1])
        return grid[-1][-1]
```
### 53. Maximum Subarray
Given an integer array nums, find the contiguous subarray (containing at least one number) which has the largest sum and return its sum.


Follow up: If you have figured out the O(n) solution, try coding another solution using the divide and conquer approach, which is more subtle.


Tags: DP, Divide and Conquer

#### Approach 1 : DP
DP is one idea to optimize the result. In this case, we want to find maximum sum of subarray. If the previous value is a negative num, the previous one plus current one cannot be the maximum, so we will not update the current num. we just update the max_sum with a bigger one(origional max_sum or the current num).
```python
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        max_sum = nums[0]
        for i in range(1, len(nums)):
            if nums[i - 1] > 0:
                nums[i] += nums[i - 1]
            max_sum = max(max_sum, nums[i])
        return max_sum
```

#### Approach 2: Divide and Conquer
Let's follow here a solution template for the divide and conquer problems :


Define the base case(s), which means the sub-problem can be solved easily. In this case, when the sub-array contains only one element, it can be returned directly. 


Split the problem into subproblems and solve them recursively. In divide and conquer solution, always divide problem into half-and-half parts.


Merge the solutions for the subproblems to obtain the solution for the original problem. In this situation, collect the maximum from bottom. 
```python
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        return self.helper(nums, 0, len(nums) - 1)
    
    def helper(self, nums, left, right):
        if left == right:
            return nums[left]
        mid = (left + right) // 2
        max_left = self.helper(nums, left, mid)
        max_right = self.helper(nums, mid + 1, right)
        cross_sum = self.cross_sum(nums, left, mid, right)
        return max(max_left, max_right, cross_sum)
    
    def cross_sum(self, nums, left, mid, right):
        if left == right:
            return nums[left]
        
        left_max = float('-inf')
        cur_sum = 0
        for i in range(mid, left - 1, -1):
            cur_sum += nums[i]
            left_max = max(left_max, cur_sum)
        
        right_max = float('-inf')
        cur_sum = 0
        for i in range(mid + 1, right + 1):
            cur_sum += nums[i]
            right_max = max(right_max, cur_sum)
            
        return right_max + left_max
```
