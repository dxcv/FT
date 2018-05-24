# 数据库访问的python接口

## 使用方法

* 请参考示例文件`example.py`中的调用方法。

* 建议从数据库中读取数据保存到本地，这样可以大大提高访问速度。

* 数据库在本地中是以HDF格式保存的，文件后缀名为`.h5`，本地数据的读取参考[`pandas.read_hdf()`](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_hdf.html)文档。


## 数据库中数据表名称和字段名

### 股票数据表

#### `equity_cash_dividend` 股票分红数据
字段名| 释义
-----|-----
stkcd | 股票代码
report_period | 财报日期
trd_dt | 最新公告日期 (匹配交易日)
ann_dt | 最新公告日期 (动态)
cash_div | 财年总现金分红 (万元)
counts | 财年分红次数

#### `equity_consensus_forecast` 分析师一致预期数据
字段名 | 释义
------|-----
stkcd | 股票代码
trd_dt | 交易日期
benchmark_yr | 预测基准年份
est_net_profit_FTTM | 未来12个月预测净利润，w=(预测日+365-(基准日+1年)) / 365, FTTM = FY2\*w+FY1\*(1-w), 后同
est_net_profit_FT24M | 未来24个月预测净利润，w=(预测日+730-(基准日+2年)) / 365, FTTM = FY3\*w+FY2\*(1-w), 后同
est_oper_revenue_FTTM | 未来12个月预测营收
est_oper_revenue_FT24M | 未来24个月预测营收
est_baseshare_FTTM | 未来12个月预测基准股份
est_baseshare_FT24M | 未来24个月预测基准股份
est_bookvalue_FTTM | 未来12个与预测净资产
est_bookvalue_FT24M | 未来24个与预测净资产
rating_dt | 评级日期
rating_avg_30 | 平均评级得分（30天）
rating_avg_90 | 平均评级得分（90天）
rating_avg_180 | 平均评级得分（180天）
rating_instnum_30 | 评级机构数（30天）
rating_instnum_90 | 评级机构数（90天）
rating_instnum_180 | 评级机构数（180天）
est_price_30 | 预测目标价（30天）
est_price_90 | 预测目标价（90天）
est_price_180 | 预测目标价（180天）
est_price_instnum_30 | 预测价格机构数（30天）
est_price_instnum_90 | 预测价格机构数（90天）
est_price_instnum_180 | 预测价格机构数（180天）

#### `equity_fundamental_info` 股票基本信息
字段名 | 释义
------|------
stkcd | 股票代码
trd_dt | 交易日期
stkname | 股票简称（早期名称可能有问题）
type_st | ST类型（1=ST,包括特别处理、暂停上市、特别转让、退市整理等）
wind_indcd | wind行业代码（2017年以前的房地产代码改为626010开头）
tot_shr | 总股本（万股）
float_a_shr | 流通A股（万股）
freeshares | 自由流通市值（万元，以上市日和公告日更晚者为知晓日期）
cap | 总市值（万元）
freefloat_cap | 流通A股市值（万元）
uplimit | 涨停价，昨收\*1.1（正常）/1.05（ST, type_st字段不为空），保留两位小数
downlimit | 跌停价，昨收\*0.9（正常）/0.95（ST），保留两位小数
adjuplimit | 复权涨停价，涨停价\*复权因子
adjdownlimit | 复权跌停价，跌停价\*复权因子
listdate | 上市日期
delistdate | 退市日期
wind_concept | 所属wind概念
index_member | 所属交易所指数的代码（上证50，深证100，沪深300，中证500，中证800，中证1000）

#### `equity_selected_balance_sheet` 资产负债表
字段名 | 释义
------|------
stkcd | 股票代码
trd_dt | 披露日期（匹配交易日）
ann_dt | 披露日期（公告日）
report_period | 财报日期
monetary_cap | 货币资金
notes_rcv | 应收票据
acct_rcv | 应收账款
prepay | 预付款项
inventories | 存货
tot_cur_assets | 流动资产
long_term_eqy_invest | 长期股权投资
long_term_rec | 长期应收款
fix_assets | 固定资产
const_in_prog | 在建工程
intang_assets | 无形资产
goodwill | 商誉
tot_non_cur_assets | 非流动资产
tot_assets | 总资产
st_borrow | 短期借款
notes_payable | 应付票据
acct_payables | 应付账款
adv_from_cust | 预收款项
non_cur_liab_due_within_1y | 一年内到期的非流动负债
tot_cur_liab | 流动负债
lt_borrow | 长期借款
bonds_payables | 应付债券
lt_payable | 长期应付款
tot_non_cur_liab | 非流动负债
tot_liab | 总负债
cao_stk | 股本/实收资本
cap_rsrv | 资本公积
surplus_rsrv | 盈余公积
undistributed_profit | 未分配利润
tot_shrhldr_eqy_excl_min_int | 所有者权益（不含少数股东权益）
tot_shrhldr_eqy_incl_min_int | 所有者权益（含少数股东权益）

#### `equity_selected_cashflow_sheet` 现金流量表
字段名 | 释义
------|------
stkcd | 股票代码
trd_dt | 披露日期（匹配交易日）
ann_dt | 披露日期（公告日）
report_period | 财报日期
net_cash_flows_oper_act | 经营活动产生的现金流净额
net_cash_recp_disp_fiolta | 处置固定资产、无形资产和其他长期资产收回现金净额
cash_pay_acq_const_fiolta | 购建固定资产、无形资产和其他长期资产支付的现金
net_cash_flows_inv_act | 投资活动产生的现金流净额
net_cash_flows_fnc_act | 筹资活动产生的现金流净额
net_incr_cash_cash_equ | 现金及现金等价物净增加额
depr_fa_coga_dpba | 固定资产折旧、油气资产折耗、生产性生物资产折旧
amort_intang_assets | 无形资产摊销
amort_lt_deferred_exp | 长期待摊费用摊销

#### `equity_selected_cashflow_sheet_q` 现金流量表（单季度）
字段名 | 释义
------|------
stkcd | 股票代码
trd_dt | 披露日期（匹配交易日）
ann_dt | 披露日期（公告日）
report_period | 财报日期
net_cash_flows_oper_act | 经营活动产生的现金流净额
net_cash_recp_disp_fiolta | 处置固定资产、无形资产和其他长期资产收回现金净额
cash_pay_acq_const_fiolta | 购建固定资产、无形资产和其他长期资产支付的现金
net_cash_flows_inv_act | 投资活动产生的现金流净额
net_cash_flows_fnc_act | 筹资活动产生的现金流净额
net_incr_cash_cash_equ | 现金及现金等价物净增加额
depr_fa_coga_dpba | 固定资产折旧、油气资产折耗、生产性生物资产折旧
amort_intang_assets | 无形资产摊销
amort_lt_deferred_exp | 长期待摊费用摊销

#### `equity_selected_income_sheet` 利润表
字段名 | 释义
------|------
stkcd | 股票代码
trd_dt | 披露日期（匹配交易日）
ann_dt | 披露日期（公告日）
report_period | 财报日期
tot_oper_rev | 营业总收入
oper_rev | 营业收入
tot_oper_cost | 营业总成本
oper_cost | 营业成本
selling_dist_exp | 销售费用
oper_admin_exp | 管理费用
fin_exp | 财务费用
net_gain_chg_fv | 公允价值变动净收益
net_invest_inc | 投资净收益
impair_loss_assets | 资产减值损失
oper_profit | 营业利润
non_oper_rev | 营业外收入
non_oper_exp | 营业外支出
inc_tax | 所得税
net_profit_incl_min_int_inc | 净利润（含少数股东损益）
net_profit_excl_min_int_inc | 净利润（不含少数股东损益）
net_profit_after_ded_nr_lp | 扣除非经常性损益后净利润

#### `equity_selected_income_sheet_q` 利润表（单季度）
字段名 | 释义
------|------
stkcd | 股票代码
trd_dt | 披露日期（匹配交易日）
ann_dt | 披露日期（公告日）
report_period | 财报日期
tot_oper_rev | 营业总收入
oper_rev | 营业收入
tot_oper_cost | 营业总成本
oper_cost | 营业成本
selling_dist_exp | 销售费用
oper_admin_exp | 管理费用
fin_exp | 财务费用
net_gain_chg_fv | 公允价值变动净收益
net_invest_inc | 投资净收益
impair_loss_assets | 资产减值损失
oper_profit | 营业利润
non_oper_rev | 营业外收入
non_oper_exp | 营业外支出
inc_tax | 所得税
net_profit_incl_min_int_inc | 净利润（含少数股东损益）
net_profit_excl_min_int_inc | 净利润（不含少数股东损益）
net_profit_after_ded_nr_lp | 扣除非经常性损益后净利润

#### `equtiy_selected_indice_ir` 股票指数
字段名 | 释义
------|------
trd_dt | 交易日期
sz50 | 上证50（收盘价，后同）
hs300 | 沪深300
zz500 | 中证500
shirborlm | Shibor1月（%）

#### `equity_selected_trading_data` 股票交易数据
字段名 | 释义
------|------
stkcd | 股票代码
trd_dt | 交易日期
preclose | 昨收盘价
open | 开盘价
high | 最高价
low | 最低价
close | 收盘价
pctchange | 日收益率（%）
volume | 成交量（股）
amount | 成交金额（元）
adjpreclose | 复权昨收盘价
adjopen | 复权开盘价
adjhigh | 复权最高价
adjlow | 复权最低价
adjclose | 复权收盘价
adjfactor | 复权因子
avgprice | 成交均价（VWAP）
tradestatus | 交易状态（DR-除权除息，N-新股，XD-除息，XR-除权）

#### `equity_shareholder_big10` 前十大股东
字段名 | 释义
------|------
stkcd | 股票代码
trd_dt | 披露日期（匹配交易日）
ann_dt | 披露日期（公告日）
holder_enddate | 截止日期
holder_holderacategory | 股东类型，1-个人，2-公司
holder_quantity | 持股数量
holder_name | 股东名称
holder_windname | 股东名称（万得。容错后）
holder_sharecategoryname | 持股性质

#### `equity_shareholder_float_big10` 流通股前十大股东
字段名 | 释义
------|------
stkcd | 股票代码
trd_dt | 披露日期（匹配交易日）
ann_dt | 披露日期（公告日）
holder_enddate | 截止日期
holder_holderacategory | 股东类型，1-个人，2-公司
holder_quantity | 持股数量
holder_name | 股东名称
holder_windname | 股东名称（万得。容错后）
holder_sharecategoryname | 持股性质

#### `equity_shareholder_number` 股东数量
字段名 | 释义
------|------
stkcd | 股票代码
trd_dt | 披露日期（匹配交易日）
ann_dt | 披露日期（公告日）
holder_enddate | 截止日期
hodler_num | A股股东户数
holder_total_num | 股东总户数

### 期货数据表

#### `commodityitems_daily` 期货日频数据
字段名 | 释义
------|------
code | 合约代码
name | 合约名称
arbridge | 商品种类代码
exchange | 交易所
dates | 交易日期
LCP | 昨收盘价
LSP | 昨结算价
OP | 开盘价
HIP | 最高价
LOP | 最低价
CP | 收盘价
SP | 结算价
CQ | 成交量（手）
CM | 成交金额（千元）
Rtn | 收益率
Vwap | 成交均价
OI | 合约持仓量
OBPD | 开盘基准价
ContractMulti | 合约乘数
Amplitude | 振幅
CPChange | 涨跌幅（收盘价）
SPChange | 涨跌幅（结算价）
PrePosition | 昨日持仓

#### `commodityitems_minutely` 期货分钟频数据
字段名 | 释义
------|------
code | 合约代码
name | 合约名称
arbridge | 商品种类代码
exchange | 交易所
dates | 交易日期
OP | 开盘价
HIP | 最高价
LOP | 最低价
CP | 收盘价
CQ | 成交量（手）
CM | 成交金额（元）
Rtn | 收益率
Vwap | 成交均价
OI | 合约持仓量
B1 | 买一价
S1 | 卖一价
BV1 | 买一量
SV1 | 卖一量
B2 | 买二价
S2 | 卖二价
BV2 | 买二量
SV2 | 卖二量
B3 | 买三价
S3 | 卖三价
BV3 | 买三量
SV3 | 卖三量
B4 | 买四价
S4 | 卖四价
BV4 | 买四量
SV4 | 卖四量
B5 | 买五价
S5 | 卖五价
BV5 | 买五量
SV5 | 卖五量

#### 期货品种代码表
代码 | 名称 | 交易所
------|------|------
A | 豆一 | 大商所
AG | 沪银 | 上期所
AL | 沪铝 | 上期所
AP | 苹果 | 郑商所
AU | 沪金 | 上期所
B | 豆二 | 大商所
BB | 胶合板 | 大商所
BU | 沥青 | 上期所
C | 玉米 | 大商所
CF | 郑棉 | 郑商所
CS | 玉米淀粉 | 大商所
CU | 沪铜 | 上期所
FB | 纤维板 | 大商所
FG | 玻璃 | 郑商所
FU | 燃料油 | 上期所
HC | 热轧卷板 | 上期所
I | 铁矿石 | 大商所
IC | 中证500 | 中金所
IF | 沪深300 | 中金所
IH | 上证50 | 中金所
J | 焦炭 | 大商所
JD | 鸡蛋 | 大商所
JM | 焦煤 | 大商所
JR | 粳稻 | 郑商所
L | 塑料 | 大商所
LR | 晚籼稻 | 郑商所
M | 豆粕 | 大商所
MA | 甲醇 | 郑商所
NI | 沪镍 | 上期所
OI | 菜籽油 | 郑商所
P | 棕榈油 | 大商所
PB | 铅 | 上期所
PM | 普麦 | 郑商所
PP | 聚丙烯 | 大商所
RB | 螺纹钢 | 上期所
RI | 早籼稻 | 郑商所
RM | 菜粕 | 郑商所
RS | 油菜籽 | 郑商所
RU | 橡胶 | 上期所
SF | 硅铁 | 郑商所
SM | 硅锰 | 郑商所
SN | 沪锡 | 上期所
SR | 白糖 | 郑商所
T | 10年期国债 | 中金所
TA | PTA | 郑商所
TF | 5年期国债 | 中金所
V | PVC | 大商所
WH | 强麦 | 郑商所
WR | 线材 | 上期所
WT | 硬麦 | 郑商所
Y | 豆油 | 大商所
ZC | 动力煤 | 郑商所
ZN | 沪锌 | 上期所
