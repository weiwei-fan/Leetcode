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
