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
遍历word : wordDict，并且用s.startswith(word)是非常慢的，有个小技巧就是遍历s能切分的每个单词s[:i] for i in range(1, len(s) + 1)，判断s[:i]是否在wordSet里面
```


