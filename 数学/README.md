# 29. Divide Two Integers

Given two integers dividend and divisor, divide two integers without using multiplication, division, and mod operator.


Return the quotient after dividing dividend by divisor.


The integer division should truncate toward zero, which means losing its fractional part. For example, truncate(8.345) = 8 and truncate(-2.7335) = -2.


Note:


Assume we are dealing with an environment that could only store integers within the 32-bit signed integer range: [−231,  231 − 1]. For this problem, assume that your function returns 231 − 1 when the division result overflows.


Tags: Math, Bit Manipulation
## Approach 1: Bit manipulation
Left shif: multiple by 2 </br>
eg: 1 << 1 = 1 * 2 </br>
1 << 2 = 1 * 2 * 2 = 4 </br>
Idea: Every time increase more steps(times) to speed up but if the step is too big, go into cur < 0 to decrease step until it comes to a sutiable position.
```python
class Solution:
    def divide(self, dividend: int, divisor: int) -> int:
        sign = -1 if ((dividend > 0 and divisor < 0) or (dividend < 0 and divisor > 0)) else 1
        dividend, divisor = abs(dividend), abs(divisor)
        
        times, res = 0, 0
        while dividend >= divisor:
            cur = dividend - (divisor << times)
            if cur >= 0:
                res += 1 << times
                times += 1
                dividend = cur
            else:
                times -= 1
        
        return max(-2 ** 31, min(sign * res, 2 ** 31 - 1))
```
