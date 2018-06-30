# Database
1. database 里边得财务数据都已经填充好了，直接rolling(4).sum() 就可以算ttm
2. trd_dt 表示数据最新可得的交易日，但是这个数据只能用于下一个交易日的交易，trd_dt
    当日交易。




# Ideas
1. 先整理好数据格式，填充financial data
2. 填充trading data
3. 所有index 都是基于交易日期
4. 对于monthly 或quarterly的数据，使用calendar date,有点如下：
    1) resample 后得到的是calendar date
    2) 在真实做交易的时候，calendar date 转 trading date 是容易的。只需要在resample
        的时候一直把trading date 带在DataFrame 的某列中。

5. 对于financial report 中的数据，应该使用最早公布的数据，三个表的公告日期应该相同
   向前填充8个季度
6. 标准化各个表的index，用 trd_dt 或report_period  先用公告日期构建一个mould (span)

7. 其他频率的数据也是类似，先标准化一个mould，对齐格式
8. 对于不同频率数据的结合，先统一频率格式，比如quarterly to monthly, daily to monthly等
9. 对于月频统一使用calendar month_end，因为resample('M') 返回的是month_end.



5. 回测和真实交易由很大的不同：
    1) rebalance的时候可能会每天观测，而回测一般是在每个月月末rebalance。
# TODO：
1. 构建 datastream  自己整理数据 格式

