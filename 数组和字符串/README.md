# 560. Subarray Sum Equals K
Given an array of integers nums and an integer k, return the total number of continuous subarrays whose sum equals to k.


Tags: Medium, Array, HashMap

## Approach 1: HashMap
```python
class Solution:
    def subarraySum(self, nums: List[int], k: int) -> int:
        mapp = dict()
        count, sums = 0, 0
        mapp[0] = 1
        for i in range(len(nums)):
            sums += nums[i]
            if sums-k in mapp:
                count += mapp[sums - k]
            mapp[sums] = mapp.get(sums, 0) + 1
        return count
```
### Tips
1. Subarray sum --> prefix sum </br>
Use an array called presumto save all the sum from index 0 to i </br>
eg. sum[1:2] = sum[0:2] - sum[0:0] </br>
presum[0] = 0
presum[1] = nums[0]
presum[2] = nums[0] + nums[1]


2. Counter the frequency --> Hash Table </br>
In this case, keep a record that how many times a speific sum appears.

