import time
t0 = time.clock()
import logging
logging.getLogger('fbprophet').setLevel(logging.ERROR)
import warnings
warnings.simplefilter("ignore", DeprecationWarning)
warnings.simplefilter("ignore", FutureWarning)
import pandas as pd
import fbprophet
Prophet = fbprophet.Prophet
import numpy as np


# 导入并整理目标变量
data = pd.read_csv('test.csv')
data.rename(columns={data.columns[0]:'ds', data.columns[1]:'y'}, inplace=True)


data_train = data[data[data.columns[0]] <= '2017-12-31']
data_train[data_train.columns[0]] = pd.to_datetime(data_train[data_train.columns[0]])  # 为了后续的resample，将data_train中str格式的时间列转为datetime格式。
data_train = pd.DataFrame(data=list(data_train[data_train.columns[1]]),
                          index=data_train[data_train.columns[0]], columns=['y'])  # 将data_train的index变为datetime，以便下一步将日序列重采样为周序列。
data_train_week = data_train.resample('W').sum()  # 将index为日序列的dataframe降采样为周序列的dataframe
date_str_list = []
for i in range(len(data_train_week)):
    date_str_list.append(data_train_week.index[i].strftime('%Y-%m-%d'))  # 为了后续做出符合fbprophet要求的datefrmae，将timestamp的时间列转成str
dictionary = {data_train_week.index.name: date_str_list,
              data_train_week.columns[0]: list(data_train_week[data_train_week.columns[0]])}
data_train_week = pd.DataFrame(data=dictionary)


data_test = data[data[data.columns[0]] > '2017-12-31']
data_test[data_test.columns[0]] = pd.to_datetime(data_test[data_test.columns[0]])
data_test = pd.DataFrame(data=list(data_test[data_test.columns[1]]),
                          index=data_test[data_test.columns[0]], columns=['y'])
data_test_week = data_test.resample('W').sum()
date_str_list = []
for i in range(len(data_test_week)):
    date_str_list.append(data_test_week.index[i].strftime('%Y-%m-%d'))
dictionary = {data_test_week.index.name: date_str_list,
              data_test_week.columns[0]: list(data_test_week[data_test_week.columns[0]])}
data_test_week = pd.DataFrame(data=dictionary)


fc = 20  # 设置预测步数
end = str(pd.date_range(start=data_train_week['ds'].max(), periods=fc+1, freq='W').max())
data_test_week = data_test_week[data_test_week['ds'] <= end]


# 导入并整理温度变量
temp = pd.read_csv('temp_day.csv', parse_dates=True)
temp.rename(columns={temp.columns[0]: 'ds'}, inplace=True)
tempt = temp.loc[:, ['ds', 'Tmean(C)']]


tempt_train = tempt[(tempt.index >= tempt[tempt.ds == '2013-01-01'].index[0])
                    & (tempt.index <= tempt[tempt.ds == '2017-12-31'].index[0])]
tempt_train[tempt_train.columns[0]] = pd.to_datetime(tempt_train[tempt_train.columns[0]])
tempt_train = pd.DataFrame(data=list(tempt_train[tempt_train.columns[1]]),
                           index=tempt_train[tempt_train.columns[0]], columns=['Tmean(C)'])
tempt_train_week = tempt_train.resample('W').mean()
pydate_array = tempt_train_week.index.to_pydatetime()
date_str_array = np.vectorize(lambda x: x.strftime('%Y-%m-%d'))(pydate_array)
dictionary = {tempt_train_week.index.name: list(date_str_array),
              tempt_train_week.columns[0]: list(tempt_train_week[tempt_train_week.columns[0]])}
tempt_train_week = pd.DataFrame(data=dictionary)


tempt_test = tempt[(tempt.index > tempt[tempt.ds == '2017-12-31'].index[0])
                   & (tempt.index <= tempt[tempt.ds == '2018-06-01'].index[0])]
tempt_test[tempt_test.columns[0]] = pd.to_datetime(tempt_test[tempt_test.columns[0]])
tempt_test = pd.DataFrame(data=list(tempt_test[tempt_test.columns[1]]),
                          index=tempt_test[tempt_test.columns[0]], columns=['Tmean(C)'])
tempt_test_week = tempt_test.resample('W').mean()
pydate_array = tempt_test_week.index.to_pydatetime()
date_str_array = np.vectorize(lambda x: x.strftime('%Y-%m-%d'))(pydate_array)
dictionary = {tempt_test_week.index.name: list(date_str_array),
              tempt_test_week.columns[0]: list(tempt_test_week[tempt_test_week.columns[0]])}
tempt_test_week = pd.DataFrame(data=dictionary)
tempt_test_week = tempt_test_week[tempt_test_week['ds'] <= end]


# 导入并整理湿度变量
temp = pd.read_csv('temp_day.csv', parse_dates=True)
temp.rename(columns={temp.columns[0]: 'ds'}, inplace=True)
RH = temp.loc[:, ['ds', 'RHmean(%)']]
RH = RH.interpolate(method='linear')  # 因为RH中有NaN，进入fit时会报错，所以需填充数值。


RH_train = RH[(RH.index >= RH[RH.ds == '2013-01-01'].index[0])
                    & (RH.index <= RH[RH.ds == '2017-12-31'].index[0])]
RH_train[RH_train.columns[0]] = pd.to_datetime(RH_train[RH_train.columns[0]])
RH_train = pd.DataFrame(data=list(RH_train[RH_train.columns[1]]),
                        index=RH_train[RH_train.columns[0]], columns=['RHmean(%)'])
RH_train_week = RH_train.resample('W').mean()
pydate_array = RH_train_week.index.to_pydatetime()
date_str_array = np.vectorize(lambda x: x.strftime('%Y-%m-%d'))(pydate_array)
dictionary = {RH_train_week.index.name: list(date_str_array),
              RH_train_week.columns[0]: list(RH_train_week[RH_train_week.columns[0]])}
RH_train_week = pd.DataFrame(data=dictionary)


RH_test = RH[(RH.index > RH[RH.ds == '2017-12-31'].index[0])
                   & (RH.index <= RH[RH.ds == '2018-06-01'].index[0])]
RH_test[RH_test.columns[0]] = pd.to_datetime(RH_test[RH_test.columns[0]])
RH_test = pd.DataFrame(data=list(RH_test[RH_test.columns[1]]),
                       index=RH_test[RH_test.columns[0]], columns=['RHmean(%)'])
RH_test_week = RH_test.resample('W').mean()
pydate_array = RH_test_week.index.to_pydatetime()
date_str_array = np.vectorize(lambda x: x.strftime('%Y-%m-%d'))(pydate_array)
dictionary = {RH_test_week.index.name: list(date_str_array),
              RH_test_week.columns[0]: list(RH_test_week[RH_test_week.columns[0]])}
RH_test_week = pd.DataFrame(data=dictionary)
RH_test_week = RH_test_week[RH_test_week['ds'] <= end]


# 导入并整理降雨量变量
rain = pd.read_csv('rain_day.csv', parse_dates=True)
rain.rename(columns={rain.columns[0]: 'ds', rain.columns[1]:'rain'}, inplace=True)


rain_train = rain[(rain.index >= rain[rain.ds == '2013-01-01'].index[0])
                  & (rain.index <= rain[rain.ds == '2017-12-31'].index[0])]
rain_train[rain_train.columns[0]] = pd.to_datetime(rain_train[rain_train.columns[0]])
rain_train = pd.DataFrame(data=list(rain_train[rain_train.columns[1]]),
                          index=rain_train[rain_train.columns[0]], columns=['rain'])
rain_train_week = rain_train.resample('W').mean()
pydate_array = rain_train_week.index.to_pydatetime()
date_str_array = np.vectorize(lambda x: x.strftime('%Y-%m-%d'))(pydate_array)
dictionary = {rain_train_week.index.name: list(date_str_array),
              rain_train_week.columns[0]: list(rain_train_week[rain_train_week.columns[0]])}
rain_train_week = pd.DataFrame(data=dictionary)
rain_train_week.rename(columns={rain_train_week.columns[1]: 'rain'}, inplace=True)


rain_test = rain[(rain.index > rain[rain.ds == '2017-12-31'].index[0])
                   & (rain.index <= rain[rain.ds == '2018-06-01'].index[0])]
rain_test[rain_test.columns[0]] = pd.to_datetime(rain_test[rain_test.columns[0]])
rain_test = pd.DataFrame(data=list(rain_test[rain_test.columns[1]]),
                         index=rain_test[rain_test.columns[0]], columns=['rain'])
rain_test_week = rain_test.resample('W').mean()
pydate_array = rain_test_week.index.to_pydatetime()
date_str_array = np.vectorize(lambda x: x.strftime('%Y-%m-%d'))(pydate_array)
dictionary = {rain_test_week.index.name: list(date_str_array),
              rain_test_week.columns[0]: list(rain_test_week[rain_test_week.columns[0]])}
rain_test_week = pd.DataFrame(data=dictionary)
rain_test_week = rain_test_week[rain_test_week['ds'] <= end]


# 导入并整理日照量变量
sun = pd.read_csv('sun_day.csv', parse_dates=True)
sun.rename(columns={sun.columns[0]: 'ds', sun.columns[1]: 'sun'}, inplace=True)


sun_train = sun[(sun.index >= sun[sun.ds == '2013-01-01'].index[0])
                    & (sun.index <= sun[sun.ds == '2017-12-31'].index[0])]
sun_train[sun_train.columns[0]] = pd.to_datetime(sun_train[sun_train.columns[0]])
sun_train = pd.DataFrame(data=list(sun_train[sun_train.columns[1]]),
                         index=sun_train[sun_train.columns[0]], columns=['sun'])
sun_train_week = sun_train.resample('W').mean()
pydate_array = sun_train_week.index.to_pydatetime()
date_str_array = np.vectorize(lambda x: x.strftime('%Y-%m-%d'))(pydate_array)
dictionary = {sun_train_week.index.name: list(date_str_array),
              sun_train_week.columns[0]: list(sun_train_week[sun_train_week.columns[0]])}
sun_train_week = pd.DataFrame(data=dictionary)


sun_test = sun[(sun.index > sun[sun.ds == '2017-12-31'].index[0])
                   & (sun.index <= sun[sun.ds == '2018-06-01'].index[0])]
sun_test[sun_test.columns[0]] = pd.to_datetime(sun_test[sun_test.columns[0]])
sun_test = pd.DataFrame(data=list(sun_test[sun_test.columns[1]]),
                        index=sun_test[sun_test.columns[0]], columns=['sun'])
sun_test_week = sun_test.resample('W').mean()
pydate_array = sun_test_week.index.to_pydatetime()
date_str_array = np.vectorize(lambda x: x.strftime('%Y-%m-%d'))(pydate_array)
dictionary = {sun_test_week.index.name: list(date_str_array),
              sun_test_week.columns[0]: list(sun_test_week[sun_test_week.columns[0]])}
sun_test_week = pd.DataFrame(data=dictionary)
sun_test_week = sun_test_week[sun_test_week['ds'] <= end]


# 导入并整理风力变量
wind = pd.read_csv('wind_day.csv', parse_dates=True)
wind.rename(columns={wind.columns[0]: 'ds', wind.columns[2]: 'wind'}, inplace=True)
wind.drop(columns=[wind.columns[1]], inplace=True)


wind_train = wind[(wind.index >= wind[wind.ds == '2013-01-01'].index[0])
                  & (wind.index <= wind[wind.ds == '2017-12-31'].index[0])]
wind_train[wind_train.columns[0]] = pd.to_datetime(wind_train[wind_train.columns[0]])
wind_train = pd.DataFrame(data=list(wind_train[wind_train.columns[1]]),
                          index=wind_train[wind_train.columns[0]], columns=['wind'])
wind_train_week = wind_train.resample('W').mean()
pydate_array = wind_train_week.index.to_pydatetime()
date_str_array = np.vectorize(lambda x: x.strftime('%Y-%m-%d'))(pydate_array)
dictionary = {wind_train_week.index.name: list(date_str_array),
              wind_train_week.columns[0]: list(wind_train_week[wind_train_week.columns[0]])}
wind_train_week = pd.DataFrame(data=dictionary)


wind_test = wind[(wind.index > wind[wind.ds == '2017-12-31'].index[0])
                   & (wind.index <= wind[wind.ds == '2018-06-01'].index[0])]
wind_test[wind_test.columns[0]] = pd.to_datetime(wind_test[wind_test.columns[0]])
wind_test = pd.DataFrame(data=list(wind_test[wind_test.columns[1]]),
                         index=wind_test[wind_test.columns[0]], columns=['wind'])
wind_test_week = wind_test.resample('W').mean()
pydate_array = wind_test_week.index.to_pydatetime()
date_str_array = np.vectorize(lambda x: x.strftime('%Y-%m-%d'))(pydate_array)
dictionary = {wind_test_week.index.name: list(date_str_array),
              wind_test_week.columns[0]: list(wind_test_week[wind_test_week.columns[0]])}
wind_test_week = pd.DataFrame(data=dictionary)
wind_test_week = wind_test_week[wind_test_week['ds'] <= end]


# 设置自定义节假日。注意：当时序为周序列时，自定义节假日时间段与周时间段匹配时，节假日项才参与计算。
# 'lower_window': -n，表示从“指定日期减n天”开始计算假日分量；upper_window': n，表示假日分量计算到”指定日期加n天“；0表示指定日期当天。
holidays_df = pd.DataFrame({'holiday': 'holidays_df',
                            'ds': pd.to_datetime(['2013-01-06', '2013-03-08', '2013-06-01',
                                                  '2013-09-10', '2013-11-11', '2013-12-12',
                                                  '2014-01-06', '2014-03-08', '2014-06-01',
                                                  '2014-09-10', '2014-11-11', '2014-12-12',
                                                  '2015-01-06', '2015-03-08', '2015-06-01',
                                                  '2015-09-10', '2015-11-11', '2015-12-12',
                                                  '2016-01-06', '2016-03-08', '2016-06-01',
                                                  '2016-09-10', '2016-11-11', '2016-12-12',
                                                  '2017-01-06', '2017-03-08', '2017-06-01',
                                                  '2017-09-10', '2017-11-11', '2017-12-12',
                                                  '2018-01-06', '2018-03-08', '2018-06-01',
                                                  '2018-09-10', '2018-11-11', '2018-12-12', ]),
                            'lower_window': -1,
                            'upper_window': 1})


def prophet_func_111(pp_date=data_train_week['ds'].max(), original_data=data_train_week, fc=fc, holiday=holidays_df,
                     relative_data_1_train=tempt_train_week, relative_data_2_train=RH_train_week,
                     relative_data_3_train=rain_train_week, relative_data_4_train=sun_train_week,
                     relative_data_5_train=wind_train_week,
                     relative_data_1_test=tempt_test_week, relative_data_2_test=RH_test_week,
                     relative_data_3_test=rain_test_week, relative_data_4_test=sun_test_week,
                     relative_data_5_test=wind_test_week):
    '''
    prophet预测，所有非线性项均以加法方式进入模型。
    '''

    # 设置模型参数。“1”表示对应项以加法方式进入模型，“2”表示对应项以乘法方式进入模型。
    m111 = Prophet(holidays=holiday, holidays_prior_scale=0.25, changepoint_prior_scale=0.01,
                   seasonality_mode='additive', yearly_seasonality=False, weekly_seasonality=False,
                   daily_seasonality=False)
    # 内置固定及移动节假日
    m111.add_country_holidays(country_name='CN')
    # 设置多重季节性
    m111.add_seasonality(name='yearly', period=365 / 7, fourier_order=6, prior_scale=0.1, mode='additive')
    m111.add_seasonality(name='seasonly', period=365 / 4 / 7, fourier_order=5, prior_scale=0.1, mode='additive')
    m111.add_seasonality(name='monthly', period=365 / 12 / 7, fourier_order=4, prior_scale=0.1, mode='additive')
    # 设置add_regressor中相关变量信息但不赋值
    m111.add_regressor(relative_data_1_train.columns[1], prior_scale=0.03, mode='additive')
    m111.add_regressor(relative_data_2_train.columns[1], prior_scale=0.03, mode='additive')
    m111.add_regressor(relative_data_3_train.columns[1], prior_scale=0.03, mode='additive')
    m111.add_regressor(relative_data_4_train.columns[1], prior_scale=0.03, mode='additive')
    m111.add_regressor(relative_data_5_train.columns[1], prior_scale=0.03, mode='additive')
    # 拟合及预测数据
    datestamp = pd.date_range(end=pp_date, periods=len(original_data))
    df_train = pd.DataFrame({'ds': datestamp, 'y': original_data['y'],
                             relative_data_1_train.columns[1]: relative_data_1_train[relative_data_1_train.columns[1]],
                             relative_data_2_train.columns[1]: relative_data_2_train[relative_data_2_train.columns[1]],
                             relative_data_3_train.columns[1]: relative_data_3_train[relative_data_3_train.columns[1]],
                             relative_data_4_train.columns[1]: relative_data_4_train[relative_data_4_train.columns[1]],
                             relative_data_5_train.columns[1]: relative_data_5_train[relative_data_5_train.columns[1]],
                             })  # 构造完整训练集df_train
    m111.fit(df_train)
    future = m111.make_future_dataframe(periods=fc, freq='1W', include_history=False)
    future111 = pd.concat([future, relative_data_1_test[relative_data_1_test.columns[1]],
                           relative_data_2_test[relative_data_2_test.columns[1]],
                           relative_data_3_test[relative_data_3_test.columns[1]],
                           relative_data_4_test[relative_data_4_test.columns[1]],
                           relative_data_5_test[relative_data_5_test.columns[1]]], axis=1)  # 构造完整测试集future111
    forecast111 = m111.predict(future111)
    # 输出预测结果
    prophet_amou = forecast111['yhat'].values
    trend_amou = forecast111['trend'].values
    holiday_amou = forecast111['holidays'].values
    seasonality_amou = forecast111['yearly'].values + forecast111['seasonly'].values + forecast111['monthly'].values
    relative_amou = forecast111['extra_regressors_additive'].values

    if False not in (prophet_amou - trend_amou - holiday_amou - seasonality_amou - relative_amou < 1e-10):
        prophet_amou_1 = prophet_amou
        trend_amou_1 = trend_amou
        holiday_amou_1 = holiday_amou
        seasonality_amou_1 = seasonality_amou
        relative_amou_1 = relative_amou
        prophet_amou_1 = [0 if i < 0 else i for i in prophet_amou_1]  # 只有prophet_amou_1的负值预测需用0代替，其余不能。
        return prophet_amou_1, trend_amou_1, holiday_amou_1, seasonality_amou_1, relative_amou_1
    else:
        return None


def prophet_func_222(pp_date=data_train_week['ds'].max(), original_data=data_train_week, fc=fc, holiday=holidays_df,
                     relative_data_1_train=tempt_train_week, relative_data_2_train=RH_train_week,
                     relative_data_3_train=rain_train_week, relative_data_4_train=sun_train_week,
                     relative_data_5_train=wind_train_week,
                     relative_data_1_test=tempt_test_week, relative_data_2_test=RH_test_week,
                     relative_data_3_test=rain_test_week, relative_data_4_test=sun_test_week,
                     relative_data_5_test=wind_test_week):
    '''
    prophet预测，所有非线性项均以乘法方式进入模型。
    '''

    # 设置模型参数。“1”表示对应项以加法方式进入模型，“2”表示对应项以乘法方式进入模型。
    m222 = Prophet(holidays=holiday, holidays_prior_scale=0.25, changepoint_prior_scale=0.01,
                   seasonality_mode='multiplicative', yearly_seasonality=False, weekly_seasonality=False,
                   daily_seasonality=False)
    # 内置固定及移动节假日
    m222.add_country_holidays(country_name='CN')
    # 设置多重季节性
    m222.add_seasonality(name='yearly', period=365 / 7, fourier_order=6, prior_scale=0.1, mode='multiplicative')
    m222.add_seasonality(name='seasonly', period=365 / 4 / 7, fourier_order=5, prior_scale=0.1, mode='multiplicative')
    m222.add_seasonality(name='monthly', period=365 / 12 / 7, fourier_order=4, prior_scale=0.1, mode='multiplicative')
    # 设置add_regressor中相关变量信息但不赋值
    m222.add_regressor(relative_data_1_train.columns[1], prior_scale=0.03, mode='multiplicative')
    m222.add_regressor(relative_data_2_train.columns[1], prior_scale=0.03, mode='multiplicative')
    m222.add_regressor(relative_data_3_train.columns[1], prior_scale=0.03, mode='multiplicative')
    m222.add_regressor(relative_data_4_train.columns[1], prior_scale=0.03, mode='multiplicative')
    m222.add_regressor(relative_data_5_train.columns[1], prior_scale=0.03, mode='multiplicative')
    # 拟合及预测数据
    datestamp = pd.date_range(end=pp_date, periods=len(original_data))
    df_train = pd.DataFrame({'ds': datestamp, 'y': original_data['y'],
                             relative_data_1_train.columns[1]: relative_data_1_train[relative_data_1_train.columns[1]],
                             relative_data_2_train.columns[1]: relative_data_2_train[relative_data_2_train.columns[1]],
                             relative_data_3_train.columns[1]: relative_data_3_train[relative_data_3_train.columns[1]],
                             relative_data_4_train.columns[1]: relative_data_4_train[relative_data_4_train.columns[1]],
                             relative_data_5_train.columns[1]: relative_data_5_train[relative_data_5_train.columns[1]],
                             })  # 构造完整训练集df_train
    m222.fit(df_train)
    future = m222.make_future_dataframe(periods=fc, freq='1W', include_history=False)
    future222 = pd.concat([future, relative_data_1_test[relative_data_1_test.columns[1]],
                           relative_data_2_test[relative_data_2_test.columns[1]],
                           relative_data_3_test[relative_data_3_test.columns[1]],
                           relative_data_4_test[relative_data_4_test.columns[1]],
                           relative_data_5_test[relative_data_5_test.columns[1]]], axis=1)  # 构造完整测试集future222
    forecast222 = m222.predict(future222)
    # 输出预测结果
    prophet_amou = forecast222['yhat'].values
    trend_amou = forecast222['trend'].values
    holiday_amou = forecast222['holidays'].values
    seasonality_amou = forecast222['yearly'].values + forecast222['seasonly'].values + forecast222['monthly'].values
    relative_amou = forecast222['extra_regressors_multiplicative'].values

    if False not in (prophet_amou - (trend_amou * (1 + holiday_amou + seasonality_amou + relative_amou)) < 1e-10):
        prophet_amou_1 = prophet_amou
        trend_amou_1 = trend_amou
        holiday_amou_1 = holiday_amou
        seasonality_amou_1 = seasonality_amou
        relative_amou_1 = relative_amou
        prophet_amou_1 = [0 if i < 0 else i for i in prophet_amou_1]  # 只有prophet_amou_1的负值预测需用0代替，其余不能。
        return prophet_amou_1, trend_amou_1, holiday_amou_1, seasonality_amou_1, relative_amou_1
    else:
        return None


def prophet_func_112(pp_date=data_train_week['ds'].max(), original_data=data_train_week, fc=fc, holiday=holidays_df,
                     relative_data_1_train=tempt_train_week, relative_data_2_train=RH_train_week,
                     relative_data_3_train=rain_train_week, relative_data_4_train=sun_train_week,
                     relative_data_5_train=wind_train_week,
                     relative_data_1_test=tempt_test_week, relative_data_2_test=RH_test_week,
                     relative_data_3_test=rain_test_week, relative_data_4_test=sun_test_week,
                     relative_data_5_test=wind_test_week):
    '''
    prophet预测，节假日项、季节项以加法方式进入模型，相关变量以乘法方式进入模型。
    '''

    # 设置模型参数。“1”表示对应项以加法方式进入模型，“2”表示对应项以乘法方式进入模型。
    m112 = Prophet(holidays=holiday, holidays_prior_scale=0.25, changepoint_prior_scale=0.01,
                   seasonality_mode='additive', yearly_seasonality=False, weekly_seasonality=False,
                   daily_seasonality=False)
    # 内置固定及移动节假日
    m112.add_country_holidays(country_name='CN')
    # 设置多重季节性
    m112.add_seasonality(name='yearly', period=365 / 7, fourier_order=6, prior_scale=0.1, mode='additive')
    m112.add_seasonality(name='seasonly', period=365 / 4 / 7, fourier_order=5, prior_scale=0.1, mode='additive')
    m112.add_seasonality(name='monthly', period=365 / 12 / 7, fourier_order=4, prior_scale=0.1, mode='additive')
    # 设置add_regressor中相关变量信息但不赋值
    m112.add_regressor(relative_data_1_train.columns[1], prior_scale=0.03, mode='multiplicative')
    m112.add_regressor(relative_data_2_train.columns[1], prior_scale=0.03, mode='multiplicative')
    m112.add_regressor(relative_data_3_train.columns[1], prior_scale=0.03, mode='multiplicative')
    m112.add_regressor(relative_data_4_train.columns[1], prior_scale=0.03, mode='multiplicative')
    m112.add_regressor(relative_data_5_train.columns[1], prior_scale=0.03, mode='multiplicative')
    # 拟合及预测数据
    datestamp = pd.date_range(end=pp_date, periods=len(original_data))
    df_train = pd.DataFrame({'ds': datestamp, 'y': original_data['y'],
                             relative_data_1_train.columns[1]: relative_data_1_train[relative_data_1_train.columns[1]],
                             relative_data_2_train.columns[1]: relative_data_2_train[relative_data_2_train.columns[1]],
                             relative_data_3_train.columns[1]: relative_data_3_train[relative_data_3_train.columns[1]],
                             relative_data_4_train.columns[1]: relative_data_4_train[relative_data_4_train.columns[1]],
                             relative_data_5_train.columns[1]: relative_data_5_train[relative_data_5_train.columns[1]],
                             })  # 构造完整训练集df_train
    m112.fit(df_train)
    future = m112.make_future_dataframe(periods=fc, freq='1W', include_history=False)
    future112 = pd.concat([future, relative_data_1_test[relative_data_1_test.columns[1]],
                           relative_data_2_test[relative_data_2_test.columns[1]],
                           relative_data_3_test[relative_data_3_test.columns[1]],
                           relative_data_4_test[relative_data_4_test.columns[1]],
                           relative_data_5_test[relative_data_5_test.columns[1]]], axis=1)  # 构造完整测试集future112
    forecast112 = m112.predict(future112)
    # 输出预测结果
    prophet_amou = forecast112['yhat'].values
    trend_amou = forecast112['trend'].values
    holiday_amou = forecast112['holidays'].values
    seasonality_amou = forecast112['yearly'].values + forecast112['seasonly'].values + forecast112['monthly'].values
    relative_amou = forecast112['extra_regressors_multiplicative'].values

    if False not in (prophet_amou - (trend_amou * (1 + relative_amou) + holiday_amou + seasonality_amou) < 1e-10):
        prophet_amou_1 = prophet_amou
        trend_amou_1 = trend_amou
        holiday_amou_1 = holiday_amou
        seasonality_amou_1 = seasonality_amou
        relative_amou_1 = relative_amou
        prophet_amou_1 = [0 if i < 0 else i for i in prophet_amou_1]  # 只有prophet_amou_1的负值预测需用0代替，其余不能。
        return prophet_amou_1, trend_amou_1, holiday_amou_1, seasonality_amou_1, relative_amou_1
    else:
        return None


def prophet_func_121(pp_date=data_train_week['ds'].max(), original_data=data_train_week, fc=fc, holiday=holidays_df,
                     relative_data_1_train=tempt_train_week, relative_data_2_train=RH_train_week,
                     relative_data_3_train=rain_train_week, relative_data_4_train=sun_train_week,
                     relative_data_5_train=wind_train_week,
                     relative_data_1_test=tempt_test_week, relative_data_2_test=RH_test_week,
                     relative_data_3_test=rain_test_week, relative_data_4_test=sun_test_week,
                     relative_data_5_test=wind_test_week):
    '''
    prophet预测，节假日项、相关变量以加法方式进入模型，季节项以乘法方式进入模型。
    '''

    # 设置模型参数。“1”表示对应项以加法方式进入模型，“2”表示对应项以乘法方式进入模型。
    m121 = Prophet(holidays=holiday, holidays_prior_scale=0.25, changepoint_prior_scale=0.01,
                   seasonality_mode='additive', yearly_seasonality=False, weekly_seasonality=False,
                   daily_seasonality=False)
    # 内置固定及移动节假日
    m121.add_country_holidays(country_name='CN')
    # 设置多重季节性
    m121.add_seasonality(name='yearly', period=365 / 7, fourier_order=6, prior_scale=0.1, mode='multiplicative')
    m121.add_seasonality(name='seasonly', period=365 / 4 / 7, fourier_order=5, prior_scale=0.1, mode='multiplicative')
    m121.add_seasonality(name='monthly', period=365 / 12 / 7, fourier_order=4, prior_scale=0.1, mode='multiplicative')
    # 设置add_regressor中相关变量信息但不赋值
    m121.add_regressor(relative_data_1_train.columns[1], prior_scale=0.03, mode='additive')
    m121.add_regressor(relative_data_2_train.columns[1], prior_scale=0.03, mode='additive')
    m121.add_regressor(relative_data_3_train.columns[1], prior_scale=0.03, mode='additive')
    m121.add_regressor(relative_data_4_train.columns[1], prior_scale=0.03, mode='additive')
    m121.add_regressor(relative_data_5_train.columns[1], prior_scale=0.03, mode='additive')
    # 拟合及预测数据
    datestamp = pd.date_range(end=pp_date, periods=len(original_data))
    df_train = pd.DataFrame({'ds': datestamp, 'y': original_data['y'],
                             relative_data_1_train.columns[1]: relative_data_1_train[relative_data_1_train.columns[1]],
                             relative_data_2_train.columns[1]: relative_data_2_train[relative_data_2_train.columns[1]],
                             relative_data_3_train.columns[1]: relative_data_3_train[relative_data_3_train.columns[1]],
                             relative_data_4_train.columns[1]: relative_data_4_train[relative_data_4_train.columns[1]],
                             relative_data_5_train.columns[1]: relative_data_5_train[relative_data_5_train.columns[1]],
                             })  # 构造完整训练集df_train
    m121.fit(df_train)
    future = m121.make_future_dataframe(periods=fc, freq='1W', include_history=False)
    future121 = pd.concat([future, relative_data_1_test[relative_data_1_test.columns[1]],
                           relative_data_2_test[relative_data_2_test.columns[1]],
                           relative_data_3_test[relative_data_3_test.columns[1]],
                           relative_data_4_test[relative_data_4_test.columns[1]],
                           relative_data_5_test[relative_data_5_test.columns[1]]], axis=1)  # 构造完整测试集future121
    forecast121 = m121.predict(future121)
    # 输出预测结果
    prophet_amou = forecast121['yhat'].values
    trend_amou = forecast121['trend'].values
    holiday_amou = forecast121['holidays'].values
    seasonality_amou = forecast121['yearly'].values + forecast121['seasonly'].values + forecast121['monthly'].values
    relative_amou = forecast121['extra_regressors_additive'].values

    if False not in (prophet_amou - (trend_amou * (1 + seasonality_amou) + holiday_amou + relative_amou) < 1e-10):
        prophet_amou_1 = prophet_amou
        trend_amou_1 = trend_amou
        holiday_amou_1 = holiday_amou
        seasonality_amou_1 = seasonality_amou
        relative_amou_1 = relative_amou
        prophet_amou_1 = [0 if i < 0 else i for i in prophet_amou_1]  # 只有prophet_amou_1的负值预测需用0代替，其余不能。
        return prophet_amou_1, trend_amou_1, holiday_amou_1, seasonality_amou_1, relative_amou_1
    else:
        return None


def prophet_func_122(pp_date=data_train_week['ds'].max(), original_data=data_train_week, fc=fc, holiday=holidays_df,
                     relative_data_1_train=tempt_train_week, relative_data_2_train=RH_train_week,
                     relative_data_3_train=rain_train_week, relative_data_4_train=sun_train_week,
                     relative_data_5_train=wind_train_week,
                     relative_data_1_test=tempt_test_week, relative_data_2_test=RH_test_week,
                     relative_data_3_test=rain_test_week, relative_data_4_test=sun_test_week,
                     relative_data_5_test=wind_test_week):
    '''
    prophet预测，节假日项以加法方式进入模型，季节项、相关变量以乘法方式进入模型。
    '''

    # 设置模型参数。“1”表示对应项以加法方式进入模型，“2”表示对应项以乘法方式进入模型。
    m122 = Prophet(holidays=holiday, holidays_prior_scale=0.25, changepoint_prior_scale=0.01,
                   seasonality_mode='additive', yearly_seasonality=False, weekly_seasonality=False,
                   daily_seasonality=False)
    # 内置固定及移动节假日
    m122.add_country_holidays(country_name='CN')
    # 设置多重季节性
    m122.add_seasonality(name='yearly', period=365 / 7, fourier_order=6, prior_scale=0.1, mode='multiplicative')
    m122.add_seasonality(name='seasonly', period=365 / 4 / 7, fourier_order=5, prior_scale=0.1, mode='multiplicative')
    m122.add_seasonality(name='monthly', period=365 / 12 / 7, fourier_order=4, prior_scale=0.1, mode='multiplicative')
    # 设置add_regressor中相关变量信息但不赋值
    m122.add_regressor(relative_data_1_train.columns[1], prior_scale=0.03, mode='multiplicative')
    m122.add_regressor(relative_data_2_train.columns[1], prior_scale=0.03, mode='multiplicative')
    m122.add_regressor(relative_data_3_train.columns[1], prior_scale=0.03, mode='multiplicative')
    m122.add_regressor(relative_data_4_train.columns[1], prior_scale=0.03, mode='multiplicative')
    m122.add_regressor(relative_data_5_train.columns[1], prior_scale=0.03, mode='multiplicative')
    # 拟合及预测数据
    datestamp = pd.date_range(end=pp_date, periods=len(original_data))
    df_train = pd.DataFrame({'ds': datestamp, 'y': original_data['y'],
                             relative_data_1_train.columns[1]: relative_data_1_train[relative_data_1_train.columns[1]],
                             relative_data_2_train.columns[1]: relative_data_2_train[relative_data_2_train.columns[1]],
                             relative_data_3_train.columns[1]: relative_data_3_train[relative_data_3_train.columns[1]],
                             relative_data_4_train.columns[1]: relative_data_4_train[relative_data_4_train.columns[1]],
                             relative_data_5_train.columns[1]: relative_data_5_train[relative_data_5_train.columns[1]],
                             })  # 构造完整训练集df_train
    m122.fit(df_train)
    future = m122.make_future_dataframe(periods=fc, freq='1W', include_history=False)
    future122 = pd.concat([future, relative_data_1_test[relative_data_1_test.columns[1]],
                           relative_data_2_test[relative_data_2_test.columns[1]],
                           relative_data_3_test[relative_data_3_test.columns[1]],
                           relative_data_4_test[relative_data_4_test.columns[1]],
                           relative_data_5_test[relative_data_5_test.columns[1]]], axis=1)  # 构造完整测试集future122
    forecast122 = m122.predict(future122)
    # 输出预测结果
    prophet_amou = forecast122['yhat'].values
    trend_amou = forecast122['trend'].values
    holiday_amou = forecast122['holidays'].values
    seasonality_amou = forecast122['yearly'].values + forecast122['seasonly'].values + forecast122['monthly'].values
    relative_amou = forecast122['extra_regressors_multiplicative'].values

    if False not in (prophet_amou - (trend_amou * (1 + seasonality_amou + relative_amou) + holiday_amou) < 1e-10):
        prophet_amou_1 = prophet_amou
        trend_amou_1 = trend_amou
        holiday_amou_1 = holiday_amou
        seasonality_amou_1 = seasonality_amou
        relative_amou_1 = relative_amou
        prophet_amou_1 = [0 if i < 0 else i for i in prophet_amou_1]  # 只有prophet_amou_1的负值预测需用0代替，其余不能。
        return prophet_amou_1, trend_amou_1, holiday_amou_1, seasonality_amou_1, relative_amou_1
    else:
        return None


def prophet_func_211(pp_date=data_train_week['ds'].max(), original_data=data_train_week, fc=fc, holiday=holidays_df,
                     relative_data_1_train=tempt_train_week, relative_data_2_train=RH_train_week,
                     relative_data_3_train=rain_train_week, relative_data_4_train=sun_train_week,
                     relative_data_5_train=wind_train_week,
                     relative_data_1_test=tempt_test_week, relative_data_2_test=RH_test_week,
                     relative_data_3_test=rain_test_week, relative_data_4_test=sun_test_week,
                     relative_data_5_test=wind_test_week):
    '''
    prophet预测，节假日项以乘法方式进入模型，季节项、相关变量以加法方式进入模型。
    '''

    # 设置模型参数。“1”表示对应项以加法方式进入模型，“2”表示对应项以乘法方式进入模型。
    m211 = Prophet(holidays=holiday, holidays_prior_scale=0.25, changepoint_prior_scale=0.01,
                   seasonality_mode='multiplicative', yearly_seasonality=False, weekly_seasonality=False,
                   daily_seasonality=False)
    # 内置固定及移动节假日
    m211.add_country_holidays(country_name='CN')
    # 设置多重季节性
    m211.add_seasonality(name='yearly', period=365 / 7, fourier_order=6, prior_scale=0.1, mode='additive')
    m211.add_seasonality(name='seasonly', period=365 / 4 / 7, fourier_order=5, prior_scale=0.1, mode='additive')
    m211.add_seasonality(name='monthly', period=365 / 12 / 7, fourier_order=4, prior_scale=0.1, mode='additive')
    # 设置add_regressor中相关变量信息但不赋值
    m211.add_regressor(relative_data_1_train.columns[1], prior_scale=0.03, mode='additive')
    m211.add_regressor(relative_data_2_train.columns[1], prior_scale=0.03, mode='additive')
    m211.add_regressor(relative_data_3_train.columns[1], prior_scale=0.03, mode='additive')
    m211.add_regressor(relative_data_4_train.columns[1], prior_scale=0.03, mode='additive')
    m211.add_regressor(relative_data_5_train.columns[1], prior_scale=0.03, mode='additive')
    # 拟合及预测数据
    datestamp = pd.date_range(end=pp_date, periods=len(original_data))
    df_train = pd.DataFrame({'ds': datestamp, 'y': original_data['y'],
                             relative_data_1_train.columns[1]: relative_data_1_train[relative_data_1_train.columns[1]],
                             relative_data_2_train.columns[1]: relative_data_2_train[relative_data_2_train.columns[1]],
                             relative_data_3_train.columns[1]: relative_data_3_train[relative_data_3_train.columns[1]],
                             relative_data_4_train.columns[1]: relative_data_4_train[relative_data_4_train.columns[1]],
                             relative_data_5_train.columns[1]: relative_data_5_train[relative_data_5_train.columns[1]],
                             })  # 构造完整训练集df_train
    m211.fit(df_train)
    future = m211.make_future_dataframe(periods=fc, freq='1W', include_history=False)
    future211 = pd.concat([future, relative_data_1_test[relative_data_1_test.columns[1]],
                           relative_data_2_test[relative_data_2_test.columns[1]],
                           relative_data_3_test[relative_data_3_test.columns[1]],
                           relative_data_4_test[relative_data_4_test.columns[1]],
                           relative_data_5_test[relative_data_5_test.columns[1]]], axis=1)  # 构造完整测试集future211
    forecast211 = m211.predict(future211)
    # 输出预测结果
    prophet_amou = forecast211['yhat'].values
    trend_amou = forecast211['trend'].values
    holiday_amou = forecast211['holidays'].values
    seasonality_amou = forecast211['yearly'].values + forecast211['seasonly'].values + forecast211['monthly'].values
    relative_amou = forecast211['extra_regressors_additive'].values

    if False not in (prophet_amou - (trend_amou * (1 + holiday_amou) + seasonality_amou + relative_amou) < 1e-10):
        prophet_amou_1 = prophet_amou
        trend_amou_1 = trend_amou
        holiday_amou_1 = holiday_amou
        seasonality_amou_1 = seasonality_amou
        relative_amou_1 = relative_amou
        prophet_amou_1 = [0 if i < 0 else i for i in prophet_amou_1]  # 只有prophet_amou_1的负值预测需用0代替，其余不能。
        return prophet_amou_1, trend_amou_1, holiday_amou_1, seasonality_amou_1, relative_amou_1
    else:
        return None


def prophet_func_212(pp_date=data_train_week['ds'].max(), original_data=data_train_week, fc=fc, holiday=holidays_df,
                     relative_data_1_train=tempt_train_week, relative_data_2_train=RH_train_week,
                     relative_data_3_train=rain_train_week, relative_data_4_train=sun_train_week,
                     relative_data_5_train=wind_train_week,
                     relative_data_1_test=tempt_test_week, relative_data_2_test=RH_test_week,
                     relative_data_3_test=rain_test_week, relative_data_4_test=sun_test_week,
                     relative_data_5_test=wind_test_week):
    '''
    prophet预测，节假日项、相关变量以乘法方式进入模型，季节项以加法方式进入模型。
    '''

    # 设置模型参数。“1”表示对应项以加法方式进入模型，“2”表示对应项以乘法方式进入模型。
    m212 = Prophet(holidays=holiday, holidays_prior_scale=0.25, changepoint_prior_scale=0.01,
                   seasonality_mode='multiplicative', yearly_seasonality=False, weekly_seasonality=False,
                   daily_seasonality=False)
    # 内置固定及移动节假日
    m212.add_country_holidays(country_name='CN')
    # 设置多重季节性
    m212.add_seasonality(name='yearly', period=365 / 7, fourier_order=6, prior_scale=0.1, mode='additive')
    m212.add_seasonality(name='seasonly', period=365 / 4 / 7, fourier_order=5, prior_scale=0.1, mode='additive')
    m212.add_seasonality(name='monthly', period=365 / 12 / 7, fourier_order=4, prior_scale=0.1, mode='additive')
    # 设置add_regressor中相关变量信息但不赋值
    m212.add_regressor(relative_data_1_train.columns[1], prior_scale=0.03, mode='multiplicative')
    m212.add_regressor(relative_data_2_train.columns[1], prior_scale=0.03, mode='multiplicative')
    m212.add_regressor(relative_data_3_train.columns[1], prior_scale=0.03, mode='multiplicative')
    m212.add_regressor(relative_data_4_train.columns[1], prior_scale=0.03, mode='multiplicative')
    m212.add_regressor(relative_data_5_train.columns[1], prior_scale=0.03, mode='multiplicative')
    # 拟合及预测数据
    datestamp = pd.date_range(end=pp_date, periods=len(original_data))
    df_train = pd.DataFrame({'ds': datestamp, 'y': original_data['y'],
                             relative_data_1_train.columns[1]: relative_data_1_train[relative_data_1_train.columns[1]],
                             relative_data_2_train.columns[1]: relative_data_2_train[relative_data_2_train.columns[1]],
                             relative_data_3_train.columns[1]: relative_data_3_train[relative_data_3_train.columns[1]],
                             relative_data_4_train.columns[1]: relative_data_4_train[relative_data_4_train.columns[1]],
                             relative_data_5_train.columns[1]: relative_data_5_train[relative_data_5_train.columns[1]],
                             })  # 构造完整训练集df_train
    m212.fit(df_train)
    future = m212.make_future_dataframe(periods=fc, freq='1W', include_history=False)
    future212 = pd.concat([future, relative_data_1_test[relative_data_1_test.columns[1]],
                           relative_data_2_test[relative_data_2_test.columns[1]],
                           relative_data_3_test[relative_data_3_test.columns[1]],
                           relative_data_4_test[relative_data_4_test.columns[1]],
                           relative_data_5_test[relative_data_5_test.columns[1]]], axis=1)  # 构造完整测试集future212
    forecast212 = m212.predict(future212)
    # 输出预测结果
    prophet_amou = forecast212['yhat'].values
    trend_amou = forecast212['trend'].values
    holiday_amou = forecast212['holidays'].values
    seasonality_amou = forecast212['yearly'].values + forecast212['seasonly'].values + forecast212['monthly'].values
    relative_amou = forecast212['extra_regressors_multiplicative'].values

    if False not in (prophet_amou - (trend_amou * (1 + holiday_amou + relative_amou) + seasonality_amou) < 1e-10):
        prophet_amou_1 = prophet_amou
        trend_amou_1 = trend_amou
        holiday_amou_1 = holiday_amou
        seasonality_amou_1 = seasonality_amou
        relative_amou_1 = relative_amou
        prophet_amou_1 = [0 if i < 0 else i for i in prophet_amou_1]  # 只有prophet_amou_1的负值预测需用0代替，其余不能。
        return prophet_amou_1, trend_amou_1, holiday_amou_1, seasonality_amou_1, relative_amou_1
    else:
        return None


def prophet_func_221(pp_date=data_train_week['ds'].max(), original_data=data_train_week, fc=fc, holiday=holidays_df,
                     relative_data_1_train=tempt_train_week, relative_data_2_train=RH_train_week,
                     relative_data_3_train=rain_train_week, relative_data_4_train=sun_train_week,
                     relative_data_5_train=wind_train_week,
                     relative_data_1_test=tempt_test_week, relative_data_2_test=RH_test_week,
                     relative_data_3_test=rain_test_week, relative_data_4_test=sun_test_week,
                     relative_data_5_test=wind_test_week):
    '''
    prophet预测，节假日项、季节项以乘法方式进入模型，相关变量以加法方式进入模型。
    '''

    # 设置模型参数。“1”表示对应项以加法方式进入模型，“2”表示对应项以乘法方式进入模型。
    m221 = Prophet(holidays=holiday, holidays_prior_scale=0.25, changepoint_prior_scale=0.01,
                   seasonality_mode='multiplicative', yearly_seasonality=False, weekly_seasonality=False,
                   daily_seasonality=False)
    # 内置固定及移动节假日
    m221.add_country_holidays(country_name='CN')
    # 设置多重季节性
    m221.add_seasonality(name='yearly', period=365 / 7, fourier_order=6, prior_scale=0.1, mode='multiplicative')
    m221.add_seasonality(name='seasonly', period=365 / 4 / 7, fourier_order=5, prior_scale=0.1, mode='multiplicative')
    m221.add_seasonality(name='monthly', period=365 / 12 / 7, fourier_order=4, prior_scale=0.1, mode='multiplicative')
    # 设置add_regressor中相关变量信息但不赋值
    m221.add_regressor(relative_data_1_train.columns[1], prior_scale=0.03, mode='additive')
    m221.add_regressor(relative_data_2_train.columns[1], prior_scale=0.03, mode='additive')
    m221.add_regressor(relative_data_3_train.columns[1], prior_scale=0.03, mode='additive')
    m221.add_regressor(relative_data_4_train.columns[1], prior_scale=0.03, mode='additive')
    m221.add_regressor(relative_data_5_train.columns[1], prior_scale=0.03, mode='additive')
    # 拟合及预测数据
    datestamp = pd.date_range(end=pp_date, periods=len(original_data))
    df_train = pd.DataFrame({'ds': datestamp, 'y': original_data['y'],
                             relative_data_1_train.columns[1]: relative_data_1_train[relative_data_1_train.columns[1]],
                             relative_data_2_train.columns[1]: relative_data_2_train[relative_data_2_train.columns[1]],
                             relative_data_3_train.columns[1]: relative_data_3_train[relative_data_3_train.columns[1]],
                             relative_data_4_train.columns[1]: relative_data_4_train[relative_data_4_train.columns[1]],
                             relative_data_5_train.columns[1]: relative_data_5_train[relative_data_5_train.columns[1]],
                             })  # 构造完整训练集df_train
    m221.fit(df_train)
    future = m221.make_future_dataframe(periods=fc, freq='1W', include_history=False)
    future221 = pd.concat([future, relative_data_1_test[relative_data_1_test.columns[1]],
                           relative_data_2_test[relative_data_2_test.columns[1]],
                           relative_data_3_test[relative_data_3_test.columns[1]],
                           relative_data_4_test[relative_data_4_test.columns[1]],
                           relative_data_5_test[relative_data_5_test.columns[1]]], axis=1)  # 构造完整测试集future221
    forecast221 = m221.predict(future221)
    # 输出预测结果
    prophet_amou = forecast221['yhat'].values
    trend_amou = forecast221['trend'].values
    holiday_amou = forecast221['holidays'].values
    seasonality_amou = forecast221['yearly'].values + forecast221['seasonly'].values + forecast221['monthly'].values
    relative_amou = forecast221['extra_regressors_additive'].values

    if False not in (prophet_amou - (trend_amou * (1 + holiday_amou + seasonality_amou) + relative_amou) < 1e-10):
        prophet_amou_1 = prophet_amou
        trend_amou_1 = trend_amou
        holiday_amou_1 = holiday_amou
        seasonality_amou_1 = seasonality_amou
        relative_amou_1 = relative_amou
        prophet_amou_1 = [0 if i < 0 else i for i in prophet_amou_1]  # 只有prophet_amou_1的负值预测需用0代替，其余不能。
        return prophet_amou_1, trend_amou_1, holiday_amou_1, seasonality_amou_1, relative_amou_1
    else:
        return None
