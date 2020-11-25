# 常见算法总结
## 递归和动态规划
动态规划问题的一般形式就是**求最值**，比如**最长**递增子序列，**最小**编辑距离等。


动态规划的核心思想是**穷举求最值**，因为要求最值，肯定要把所有可行的答案穷举出来，然后在其中找最值。


但是不是暴力穷举，因为动态规划问题的一个特点是存在**重叠子问题**，所以需要**备忘录**或者**DP Table**来优化穷举过程，避免不必要的计算。


动态规划问题的第二个特点是具备**最优子结构**，才能通过子问题的最值得到原问题的最值。


如何理解最优子结构，比如说，假设你考试，每门科目的成绩都是互相独立的。你的原问题是考出最高的总成绩，那么你的子问题就是要把语文考到最高，数学考到最高…… 为了每门课考到最高，你要把每门课相应的选择题分数拿到最高，填空题分数拿到最高…… 当然，最终就是你每门课都是满分，这就是最高的总成绩。得到了正确的结果：最高的总成绩就是总分。因为这个过程符合最优子结构，“每门科目考到最高”这些子问题是互相独立，互不干扰的。


但是，如果加一个条件：你的语文成绩和数学成绩会互相制约，数学分数高，语文分数就会降低，反之亦然。这样的话，显然你能考到的最高总成绩就达不到总分了，按刚才那个思路就会得到错误的结果。因为子问题并不独立，语文数学成绩无法同时最优，所以最优子结构被破坏。


动态规划的三要素是重叠子问题，最优子结构和状态转移方程。


如何思考状态转移方程？</br>
明确base case -> 明确状态 -> 明确选择 -> 定义dp数组的含义</br>


以凑零钱的问题为例，给你 k 种面值的硬币，面值分别为 c1, c2 ... ck，每种硬币的数量无限，再给一个总金额 amount，问你最少需要几枚硬币凑出这个金额，如果不可能凑出，算法返回 -1 。

1. **确定base case**，目标金额amount为0时算法返回0，因为不需要任何硬币就已经凑出目标金额。


2. **确定状态（原问题和子问题中会变化的量）**，由于硬币的数量无限，硬币的面额也是给定的，只有目标金额会不断的向base case靠近，所以唯一的状态就是目标金额amount。

3. **确定选择（导致状态产生变化的行为）**，目标金额为什么变化呢，因为在选择硬币的时候，每次选择一枚硬币，就会减少目标金额。所以所有硬币的面值就是选择。

4. **确定dp数组的定义**， 这里采用自顶向下的解法，所以会有一个递归的dp函数，一般来说函数的参数就是状态转移中会变化的量，也就是上面说到的状态；函数的返回值是题目要求我们计算的量。就本题来说，状态只有一个即目标金额，题目要求我们计算凑出目标金额所需的最少硬币数量。

所以dp(n)的定义：输入一个目标金额n，返回凑出目标金额n的最少硬币数量。

```python
# 伪代码
def coinChange(coins: List[int], amount: int):
  # dp函数（递归函数）
  def dp(n):
    # 做选择，选择需要硬币最少的那个结果
    for coin in coins:
      res = main(res, dp(n - coin) + 1)
     return res
  # 最终结果
  return dp(amount)
```
这类问题通常有两种思路且这两种思路通常时间复杂度相同。
### 自顶向下：带备忘录的递归
需要构造dp函数
```python
def coinChange(coins, amount):
  # 备忘录
  memo = dict()
  def dp(n):
      # 查看备忘录，避免重复计算
      if n in memo: return memo[n]
      # base case
      if n == 0: return 0
      if n < 0: return -1
      res = float('INF)
      for coin in coins:
          subproblem = dp(n - coin)
          # 子问题无解，跳过
          if subproblem == -1: continue
          res = min(res, 1 + subproblem)
      # 计入备忘录
      memo[n] = res if res != float('INF') else -1
      return res if res != float('INF') else -1
   
   return dp(amount)
```

### 自底向上：dp数组的迭代
需要构造dp数组，与dp函数类似，也是把状态作为变量，不过dp函数体现在函数参数上，而dp数组体现在数组索引上。</br>
dp数组定义：当目标金额为i时，至少需要dp[i]美硬币凑出。
```python
def coinChange(coins, amount):
    dp = [amount + 1] * n for n in range(amount + 1)
    # 外层for循环遍历所有状态的所有取值
    for i in range(amount + 1):
        # 内层for循环求所有选择的最小值
        for coin in coins:
            # 子问题无解，跳过
            if i - coin < 0: continue
            dp[i] = min(dp[i], 1 + dp[i - coin])
    return dp[amount] if dp[amount + 1] != amount + 1 else -1
```
PS：为啥 dp 数组初始化为 amount + 1 呢?因为凑成 amount 金额的硬币数最多只可能等于 amount（全用 1 元面值的硬币），所以初始化为 amount + 1 就相当于初始化为正无穷，便于后续取最小值。

#### 总结
不管通过备忘录还是dp table的方法优化递归树，本质上都是穷举解空间，通过将重复的解保存起来，修剪递归树，减少计算量，只是自顶向下和自底向上的区别。备忘录、DP table 就是在追求“如何聪明地穷举”。用空间换时间的思路，是降低时间复杂度的不二法门。

## 回溯法Backtracing
解决一个回溯问题，实际上是一个决策树的遍历过程。需要考虑三个问题：</br>
1.路径：已经做出的选择</br>
2.选择列表：当前可以做出的选择</br>
3.结束条件：到达决策树底层，无法再做选择的条件</br>

#### 回溯算法的框架
```python
res = []
def backtrack(路径, 选择列表):
    if 满足结束条件：
        res.append(路径)
        return
    for 选择 in 选择列表：
        # 做出选择
        path.append(选择)
        backtrack(路径， 选择列表)
        # 撤销选择
        path.pop()
```
其核心就是for循环里面的递归，在递归之前做选择，在递归之后撤销选择。</br>


回溯法不像动态规划存在最优子结构，回溯法就是纯暴力穷举，复杂度一般都比较高。

### 例1 全排列不包含重复数字
```python
def permutation(nums):
    path, res = [], []
    backtrack(nums, path, res)
    return res

def backtrack(nums, path, res):
    # 递归出口
    if len(path) == len(nums):
        res.append(path)
        return
    # 假设现在站在决策树的某一个决策点上，path记录了到达该决策点的路径，该决策点的字节点为可以做的选择
    # 遍历选择列表
    for num : nums:
        # 排除不合法的选择
        if num in path: continue
        # 做选择
        path.append(num)
        # 进入下一层决策树
        backtrack(nums, path, res)
        # 取消选择
        path.pop()
```

### 例2 n皇后问题
给你一个 N×N 的棋盘，让你放置 N 个皇后，使得它们不能互相攻击。皇后可以攻击同一行、同一列、左上左下右上右下四个方向的任意单位。


这个问题本质上跟全排列问题差不多，决策树的每一层表示棋盘上的每一行；每个节点可以做出的选择是，在该行的任意一列放置一个皇后。
```python
def nqueens(n):
    board = [['.'] * n for i in range(n)]
    res = []
    backtrack(board, 0, res)
    return res
     
 def backtrack(board, row, res):
    # 递归出口
    if row == len(board):
        res.append(board)
        return
    for col in range(len(board[0])):
        # 排除不合法的选择
        if not isValid(board, row, col):
            continue
        # 做选择
        board[row][colum] = 'Q'
        # 进入下一行决策
        backtrack(board, row + 1, res)
        # 取消选择
        board[row][colum] = '.'
```

有的时候，我们并不想得到所有合法的答案，只想要一个答案，怎么办呢？比如解数独的算法，找所有解法复杂度太高，只要找到一种解法就可以。
```python
# 函数找到一个答案就返回true
def backtrack(board, row, res):
    if row == len(board):
        res.append(board)
        return True
    for col in range(len(board[0])):
        board[row][col] = 'Q'
        if backtrack(board, row + 1, res):
            return True
        board[row][col] = '.'
    return False
```
修改后只要找到一个答案，for 循环的后续递归穷举都会被阻断。

## 广度优先搜素BFS
BFS问题的本质是在一幅图中找到起点到终点的最短距离。
```python
# 计算从起点start到终点target的最短距离
def BFS(start, target):
```
