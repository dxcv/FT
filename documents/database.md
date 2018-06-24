# Database
1. database 里边得财务数据都已经填充好了，直接rolling(4).sum() 就可以算ttm


# Ideas
1. 先整理好数据格式，填充financial data
2. 填充trading data
3. 所有index 都是基于交易日期
4. 对于monthly 或quarterly的数据，使用calendar date,有点如下：
    1) resample 后得到的是calendar date
    2) 在真实做交易的时候，calendar date 转 trading date 是容易的。只需要在resample
        的时候一直把trading date 带在DataFrame 的某列中。


5. 回测和真实交易由很大的不同：
    1) rebalance的时候可能会每天观测，而回测一般是在每个月月末rebalance。
# TODO：
1. 构建 datastream  自己整理数据 格式

