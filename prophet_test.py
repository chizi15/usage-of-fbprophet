import time
import pandas as pd
import numpy as np
from prophet import Prophet  # 1.0及以上版本的名称为prophet
import datetime as dt
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import sys
sys.path.append('/Users/zc/PycharmProjects/fbprophet/')
import regression_evaluation_def as ref
pd.set_option('mode.chained_assignment', None)
# pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# 获取计算所需原始数据
start_0 = time.monotonic()
df_ori = pd.read_excel('/Users/zc/PycharmProjects/fbprophet/gas_2015-2017.xlsx')
df = df_ori.copy()
df = df.interpolate(method='spline', order=3)  # 如果用dropna()不带参数，则行数可能会减少；也可不进行缺失值插值处理而用prophet自带插值方法
df = df.rename(columns={"日期": "ds", "修正后用气量": "y", '最低气温': 'min_temp', '前一日': 'l1', '前两日': 'l2', '前三日': 'l3',
                        '前7日': 'l7', '季节': 'seas', '待预测日温度': 'f1_temp', '日期类型': 'datetype'})
df.drop(columns=['l1', 'l2', 'l3', 'l7'], inplace=True)  # , 'seas', 'datetype'
df['ds'] = pd.to_datetime(df['ds'])  # 确保日期类型为datetime64，才能与date_ranges和forecast做merge，因为它们的ds也是datetime64类型
for i in range(1, len(df.columns)):
    df[df.columns[i]] = pd.to_numeric(df[df.columns[i]])
if pearsonr(df['min_temp'], df['f1_temp'])[0] > 0.95:
    df = df.drop(columns='f1_temp')  # 此时这两个变量相关性高于95%，应剔除一个

# 生成回测的训练集和预测期数据：回测pred个点，回测频率为freq个点/次
pred, freq = 365, 7
n = list(np.arange(freq, pred+1, freq))
df_train, df_test = [], []
df_test_all = df.drop(columns='y')
for i, j in zip(n, range(len(n))):
    df_train.append(df[:-i])
    df_test.append(df_test_all[len(df_train[j]): len(df_train[j])+freq])

# 生成自定义节假日df，此处表示节前4天，节后9天，节日当天1天；则认为此数据集的春节效应期一共有14天，为周周期7天的倍数
lower_window = -4
upper_window = 9
holidays_custom = pd.DataFrame({
  'holiday': 'spring festival',
  # 本测试数据集只到2017年的春节，向后多设置春节表示后续自定义季节性超过预测期也不影响计算
  'ds': pd.to_datetime(['2015-02-19', '2016-02-08', '2017-01-28', '2018-02-16', '2019-02-05', '2020-01-25', '2021-02-12']),
  'lower_window': lower_window,
  'upper_window': upper_window})

# 根据每个自定义节假日日期，生成对应的日期范围用于自定义季节性的自定义列
dates, lower_date, upper_date = [], [], []
i = 0
for i in range(len(str(holidays_custom['ds'][i]).split()[0].split('-'))):  # 使i的长度包含所需要年月日数值，这里是range(0, 3)
    for j in range(len(holidays_custom['ds'])):  # 使j的长度为节假日个数，这里只有7个春节，这里是range(0, 7)
        dates.append(int(str(holidays_custom['ds'][j]).split()[0].split('-')[i]))
df_dates = pd.DataFrame(np.array(dates).reshape((len(holidays_custom['ds']), len(str(holidays_custom['ds'][i]).split()[0].split('-'))), order='F'))
for i in range(len(df_dates)):
    # 因为年、月、日是三组，所以df_dates只有3个元素
    lower_date.append(dt.date(df_dates[0][i], df_dates[1][i], df_dates[2][i]) + dt.timedelta(days=lower_window))
    upper_date.append(dt.date(df_dates[0][i], df_dates[1][i], df_dates[2][i]) + dt.timedelta(days=upper_window))
date_ranges = []
for _ in range(len(df_dates)):
    date_ranges.append(pd.date_range(lower_date[_], upper_date[_]))
for i in range(len(holidays_custom['ds'])-1):
    date_ranges[i+1] = date_ranges[i].union(date_ranges[i+1])
date_ranges = date_ranges[-1]

# 根据每个日期范围生成自定义季节性的自定义列
choice = ['merge', 'apply']  # 对比两种方法的速度
index = 0
if index not in (0, 1):
    raise Exception('\'index\' must be 0 or 1')
m = []
if 0 == index:
    choice = choice[0]
    start_1 = time.process_time()
    df_r_on = pd.DataFrame({'ds': pd.Series(date_ranges), 'bool': True})
    df_r_off = pd.DataFrame({'ds': pd.Series(date_ranges), 'bool': False})
    for i in range(len(df_train)-1, -1, -1):  # 使循环变量i按降序取值
        df_train[i]['on_season'] = pd.merge(df_train[i], df_r_on, how='left', on='ds')['bool'].replace(np.nan, False).values
        df_train[i]['off_season'] = pd.merge(df_train[i], df_r_off, how='left', on='ds')['bool'].replace(np.nan, True).values
        a = pd.merge(df_test[i], df_r_on, how='left', on='ds')['bool'].replace(np.nan, False)
        a.index = df_test[i].index
        df_test[i]['on_season'] = a
        a = pd.merge(df_test[i], df_r_off, how='left', on='ds')['bool'].replace(np.nan, True)
        a.index = df_test[i].index
        df_test[i]['off_season'] = a
        if sum(df_train[i]['on_season']) + sum(df_train[i]['off_season']) != len(df_train[i]):
            raise Exception('回测时第 %s 个训练集上自定义季节性设置有误' % i)
        elif sum(df_test[i]['off_season']) + sum(df_test[i]['on_season']) != len(df_test[i]):
            raise Exception('回测时第 %s 个预测期上自定义季节性设置有误' % i)
        m.append(Prophet(holidays=holidays_custom, holidays_prior_scale=100,
                         yearly_seasonality=12, seasonality_prior_scale=50, weekly_seasonality=False,
                         daily_seasonality=False,
                         seasonality_mode='additive', mcmc_samples=0, uncertainty_samples=100))
    end_1 = time.process_time()
else:
    choice = choice[1]
    start_1 = time.process_time()
    def is_sf_season(ds, date_ranges):
        ds = pd.to_datetime(ds)
        date_ranges = pd.to_datetime(date_ranges)
        if sum(ds == date_ranges) == 1:
            return True
        else:
            return False
    for i in range(len(df_train) - 1, -1, -1):  # 使循环变量i按降序取值
        # 此循环中，apply要做 len(df_train[i]['ds']) × len(date_ranges) × len(df_train[i]) 次判断，速度较慢；可改用pd.merge生成自定义列
        df_train[i]['on_season'] = df_train[i]['ds'].apply(is_sf_season, args=(date_ranges, ))  # args必须加(,)否则报错
        df_train[i]['off_season'] = ~df_train[i]['ds'].apply(is_sf_season, args=(date_ranges, ))
        df_test[i]['on_season'] = df_test[i]['ds'].apply(is_sf_season, args=(date_ranges, ))
        df_test[i]['off_season'] = ~df_test[i]['ds'].apply(is_sf_season, args=(date_ranges, ))
        if sum(df_train[i]['on_season']) + sum(df_train[i]['off_season']) != len(df_train[i]):
            raise Exception('回测时第 %s 个训练集上自定义季节性设置有误' % i)
        elif sum(df_test[i]['off_season']) + sum(df_test[i]['on_season']) != len(df_test[i]):
            raise Exception('回测时第 %s 个预测期上自定义季节性设置有误' % i)
        m.append(Prophet(holidays=holidays_custom, holidays_prior_scale=100,
                         yearly_seasonality=10, seasonality_prior_scale=50, weekly_seasonality=False,
                         daily_seasonality=False,
                         seasonality_mode='additive', mcmc_samples=0, uncertainty_samples=100))
    end_1 = time.process_time()

# 对每套回测数据（包括对自定义季节性和相关变量）做拟合和预测，并拼接成最终回测结果
forecast = pd.DataFrame()
start_2 = time.process_time()
for i in range(len(df_train)-1, -1, -1):  # 此for循环不能和前一个for循环合并，否则因为i按降序取值，则m[i]会超出索引范围
    for j in range(1, len(df_test[0].columns) - 2):
        m[i].add_regressor(df_test[0].columns[j])
    # 对周季节性做自定义季节性处理，分为春节期间和非春节期间的周季节性
    m[i].add_seasonality(name='weekly_on_sf', period=7, fourier_order=5, prior_scale=10, condition_name='on_season',
                      mode='additive')
    m[i].add_seasonality(name='weekly_off_sf', period=7, fourier_order=5, prior_scale=50, condition_name='off_season',
                      mode='additive')
    m[i].add_country_holidays(country_name='China')
    forecast = pd.concat([forecast, m[i].fit(df_train[i]).predict(df_test[i])], ignore_index=True)  # ignore_index=True忽略掉每一个df的索引，拼接后的df重新生成从0开始的索引
    # print('\n', m[i].train_holiday_names)
end_2 = time.process_time()

# 对回测结果和真实值进行可视化评估和指标评估
y_true = df.loc[list(df.index)[-pred:]]
y_true.rename(columns={'y': 'y_true'}, inplace=True)
y_true.reset_index(drop=True, inplace=True)  # 将y_true的index设置为与forecast一致
comp = pd.merge(forecast, y_true, on='ds', how="left")
if sum(df['y'][-pred:].values != y_true['y_true'].values) > 0:
    raise Exception('用于最后对比的真实值与初始插值后的真实值存在不一致的情况')

pd.set_option('plotting.backend', 'prophet.plot')
m[-1].plot(forecast)
m[-1].plot_components(forecast, figsize=(12, 9.5))

pd.set_option('plotting.backend', 'matplotlib')
plt.figure('回测结果对比时序', figsize=(12, 9.5))
ax = plt.subplot(1,1,1)
comp[['ds', 'y_true']].plot(x='ds', y='y_true', ax=ax, color='orangered', legend=True)
comp[['ds', 'yhat']].plot(x='ds', y='yhat', ax=ax, color='royalblue', legend=True)
comp[['ds', 'yhat_upper']].plot(x='ds', y='yhat_upper', ax=ax, color='skyblue', legend=True)
comp[['ds', 'yhat_lower']].plot(x='ds', y='yhat_lower', ax=ax, color='skyblue', legend=True)

MAPE = ref.mape(y_true=comp['y_true'], y_pred=comp['yhat'])  # y_true为0时MAPE为inf
SMAPE = ref.smape(y_true=comp['y_true'], y_pred=comp['yhat'])  # y_true与y_pred均为0时SMAPE为inf
RMSE = ref.rmse(y_true=comp['y_true'], y_pred=comp['yhat'])
naive = df['y'][-len(comp['y_true'])-1:-1].reset_index(drop=True)  # 对y_true做单步naive
accuracy = ref.regression_accuracy(y_true=[comp['y_true']]*2, y_pred=[comp['yhat'], naive])
evaluation = ref.regression_evaluation(y_true=[comp['y_true']]*2, y_pred=[comp['yhat'], naive])
print('\n', '回测结果的MAPE为百分之 %.2f （此数值越接近0越好）' % (MAPE*100))
print(' 回测结果的SMAPE为百分之 %.2f （此数值越接近0越好）' % (SMAPE*100))
print(' 回测结果的RMSE为 %.4f （此数值越接近0越好）' % RMSE)
print(' 该算法回测结果与单步Naive回测结果的综合准确度指标之比为 %.2f （该数值越小越好）' % (accuracy[0][0] / accuracy[0][1]))
print(' 该算法回测结果与单步Naive回测结果的综合评估指标之比为 %.2f （该数值越小越好）' % (evaluation[0][0] / evaluation[0][1]))

# 打印各部分运行时间
end_0 = time.monotonic()
print('\n', '用 %s 方法生成所有回测序列的自定义列时，CPU运行时间为 %.3f 秒' % (choice, (end_1 - start_1)), '\n')
print(' 所有回测序列做拟合和预测时，CPU运行时间为 %.3f 秒' % (end_2 - start_2), '\n')
print(' 程序运行总时间为 %.3f 分' % ((end_0 - start_0)/60))
