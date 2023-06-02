import logging
logging.getLogger('fbprophet').setLevel(logging.ERROR)
import warnings
warnings.simplefilter("ignore", DeprecationWarning)
warnings.simplefilter("ignore", FutureWarning)
import pandas as pd
import fbprophet
Prophet = fbprophet.Prophet


'''
# 可以准备的预测相关变量：
# 一. 时序类：
# 1.农历日期或农历假日标签，2.当地的天气、平均温度、平均相对湿度，3.促销分类标签，4.陈列位置分类标签，5.除促销外的其他异常事件标签，
# 6.商店人流量，7.单品价格，8.单品销售额，9.采购订单历史记录（采购量、采购额），10.缺货历史记录（数量），
# 11.竞争对手价格、销售额、销量，12.与该单品相似商品的各种历史数据，13.门店营业天数及单品在销天数，14.地区相关门店的其他信息，
# 15.星期（对日序列）、月份、季度，16.距保质期末剩余天数，17.降雨量，18.日照时长、光照量，19.风力及风向，
# 二. 非时序类：
# 1.地区总人口、GDP、GNP、CPI、地区失业率、人均可支配收入等在统计局网站所能获取的经济指标，2.门店位置标签或门店编号、门店类型，
# 3.地区房屋均价、租金，4.单品所属部门类型或编号，5.地区相关门店的其他信息，6.商品相关属性，如生产商、品牌、材料、颜色、形状、尺寸，
# 7.某一时期地区能源价格如电价、油价等，
'''

# test

# 导入目标变量及相关变量
data = pd.read_csv('test.csv')
data = data.rename(columns={data.columns[0]: 'ds'})
data_train = data[data.index <= data[data.ds == '2017-12-31'].index[0]]
data_test = data[data.index > data[data.ds == '2017-12-31'].index[0]]
data_test.reset_index(drop=True, inplace=True)

temp = pd.read_csv('temp_day.csv', parse_dates=True)
temp = temp.rename(columns={temp.columns[0]: 'ds'})
tempt = temp.loc[:, ['ds', 'Tmean(C)']]

RH = temp.loc[:, ['ds', 'RHmean(%)']]
RH = RH.interpolate(method='linear')  # 因为RH中有NaN，进入fit时会报错，所以需填充数值。

rain = pd.read_csv('rain_day.csv', parse_dates=True)
rain = rain.rename(columns={rain.columns[0]: 'ds', rain.columns[1]: 'rain'})

sun = pd.read_csv('sun_day.csv', parse_dates=True)
sun = sun.rename(columns={sun.columns[0]: 'ds', sun.columns[1]: 'sun'})

wind = pd.read_csv('wind_day.csv', parse_dates=True)
wind = wind.rename(columns={wind.columns[0]: 'ds', wind.columns[2]: 'wind'})
wind.drop(columns=[wind.columns[1]], inplace=True)


# 设置自定义节假日
# 'lower_window': -n，表示从“指定日期减n天”开始计算假日分量；upper_window': n，表示假日分量计算到”指定日期加n天“；0表示指定日期当天。
holidays_df = pd.DataFrame({'holiday': 'holidays_df',
                            'ds': pd.to_datetime(['2013-02-14', '2013-03-08', '2013-06-01',
                                                  '2013-09-10', '2013-11-11', '2013-12-12',
                                                  '2014-02-14', '2014-03-08', '2014-06-01',
                                                  '2014-09-10', '2014-11-11', '2014-12-12',
                                                  '2015-02-14', '2015-03-08', '2015-06-01',
                                                  '2015-09-10', '2015-11-11', '2015-12-12',
                                                  '2016-02-14', '2016-03-08', '2016-06-01',
                                                  '2016-09-10', '2016-11-11', '2016-12-12',
                                                  '2017-02-14', '2017-03-08', '2017-06-01',
                                                  '2017-09-10', '2017-11-11', '2017-12-12',
                                                  '2018-02-14', '2018-03-08', '2018-06-01',
                                                  '2018-09-10', '2018-11-11', '2018-12-12', ]),
                            'lower_window': -1,
                            'upper_window': 1})


def prophet_func_111(pp_date=data_train['ds'].max(), original_data=data_train, fc=150,
                     relative_data_1=tempt, relative_data_2=RH, relative_data_3=rain,
                     relative_data_4=sun, relative_data_5=wind, holiday=holidays_df):
    '''
    prophet预测,所有非线性项均以加法方式进入模型
    '''

    # 将相关变量切分为训练集和测试集，训练集用在Prophet.fit()中训练模型，测试集用在Prophet.predict()中生成目标变量的预测数据。
    end = str(pd.date_range(start=original_data['ds'].max(), periods=fc + 1).max())

    tempt_train = relative_data_1[
        (relative_data_1['ds'] >= original_data['ds'].min()) & (relative_data_1['ds'] <= pp_date)]
    tempt_test = relative_data_1[(relative_data_1['ds'] > pp_date) & (relative_data_1['ds'] <= end)]
    tempt_train.reset_index(drop=True, inplace=True)
    tempt_test.reset_index(drop=True, inplace=True)

    RH_train = relative_data_2[
        (relative_data_2['ds'] >= original_data['ds'].min()) & (relative_data_2['ds'] <= pp_date)]
    RH_test = relative_data_2[(relative_data_2['ds'] > pp_date) & (relative_data_2['ds'] <= end)]
    RH_train.reset_index(drop=True, inplace=True)
    RH_test.reset_index(drop=True, inplace=True)

    rain_train = relative_data_3[
        (relative_data_3['ds'] >= original_data['ds'].min()) & (relative_data_3['ds'] <= pp_date)]
    rain_test = relative_data_3[(relative_data_3['ds'] > pp_date) & (relative_data_3['ds'] <= end)]
    rain_train.reset_index(drop=True, inplace=True)
    rain_test.reset_index(drop=True, inplace=True)

    sun_train = relative_data_4[
        (relative_data_4['ds'] >= original_data['ds'].min()) & (relative_data_4['ds'] <= pp_date)]
    sun_test = relative_data_4[(relative_data_4['ds'] > pp_date) & (relative_data_4['ds'] <= end)]
    sun_train.reset_index(drop=True, inplace=True)
    sun_test.reset_index(drop=True, inplace=True)

    wind_train = relative_data_5[
        (relative_data_5['ds'] >= original_data['ds'].min()) & (relative_data_5['ds'] <= pp_date)]
    wind_test = relative_data_5[(relative_data_5['ds'] > pp_date) & (relative_data_5['ds'] <= end)]
    wind_train.reset_index(drop=True, inplace=True)
    wind_test.reset_index(drop=True, inplace=True)

    # 设置模型参数。“1”表示对应项以加法方式进入模型，“2”表示对应项以乘法方式进入模型。
    m111 = Prophet(holidays=holiday, holidays_prior_scale=0.25, changepoint_prior_scale=0.01,
                   seasonality_mode='additive', yearly_seasonality=False, weekly_seasonality=False,
                   daily_seasonality=False)
    # 内置固定及移动节假日
    m111.add_country_holidays(country_name='CN')
    # 设置多重季节性
    m111.add_seasonality(name='yearly', period=365, fourier_order=12, prior_scale=0.1, mode='additive')
    m111.add_seasonality(name='seasonly', period=365 / 4, fourier_order=10, prior_scale=0.1, mode='additive')
    m111.add_seasonality(name='monthly', period=365 / 12, fourier_order=8, prior_scale=0.1, mode='additive')
    m111.add_seasonality(name='weekly', period=7, fourier_order=6, prior_scale=0.1, mode='additive')
    # 设置相关变量
    m111.add_regressor(tempt_train.columns[1], prior_scale=0.03, mode='additive')
    m111.add_regressor(RH_train.columns[1], prior_scale=0.03, mode='additive')
    m111.add_regressor(rain_train.columns[1], prior_scale=0.03, mode='additive')
    m111.add_regressor(sun_train.columns[1], prior_scale=0.03, mode='additive')
    m111.add_regressor(wind_train.columns[1], prior_scale=0.03, mode='additive')
    # 拟合及预测数据
    datestamp = pd.date_range(end=pp_date, periods=len(original_data))
    df_train = pd.DataFrame({'ds': datestamp, 'y': original_data['y'],
                             tempt_train.columns[1]: tempt_train[tempt_train.columns[1]],
                             RH_train.columns[1]: RH_train[RH_train.columns[1]],
                             rain_train.columns[1]: rain_train[rain_train.columns[1]],
                             sun_train.columns[1]: sun_train[sun_train.columns[1]],
                             wind_train.columns[1]: wind_train[wind_train.columns[1]],
                             })
    m111.fit(df_train)
    future = m111.make_future_dataframe(periods=fc, freq='1D', include_history=False)
    future111 = pd.concat([future, tempt_test[tempt_test.columns[1]], RH_test[RH_test.columns[1]],
                           rain_test[rain_test.columns[1]], sun_test[sun_test.columns[1]],
                           wind_test[wind_test.columns[1]]], axis=1)
    forecast111 = m111.predict(future111)
    # 输出数值预测结果
    prophet_amou = forecast111['yhat'].values
    trend_amou = forecast111['trend'].values
    holiday_amou = forecast111['holidays'].values
    seasonality_amou = forecast111['yearly'].values + forecast111['seasonly'].values \
                       + forecast111['monthly'].values + forecast111['weekly'].values
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


def prophet_func_222(pp_date=data_train['ds'].max(), original_data=data_train, fc=150,
                     relative_data_1=tempt, relative_data_2=RH, relative_data_3=rain,
                     relative_data_4=sun, relative_data_5=wind, holiday=holidays_df):
    '''
    prophet预测,所有非线性项均以乘法方式进入模型
    '''

    # 将相关变量切分为训练集和测试集，训练集用在Prophet.fit()中训练模型，测试集用在Prophet.predict()中生成目标变量的预测数据。
    end = str(pd.date_range(start=original_data['ds'].max(), periods=fc + 1).max())

    tempt_train = relative_data_1[
        (relative_data_1['ds'] >= original_data['ds'].min()) & (relative_data_1['ds'] <= pp_date)]
    tempt_test = relative_data_1[(relative_data_1['ds'] > pp_date) & (relative_data_1['ds'] <= end)]
    tempt_train.reset_index(drop=True, inplace=True)
    tempt_test.reset_index(drop=True, inplace=True)

    RH_train = relative_data_2[
        (relative_data_2['ds'] >= original_data['ds'].min()) & (relative_data_2['ds'] <= pp_date)]
    RH_test = relative_data_2[(relative_data_2['ds'] > pp_date) & (relative_data_2['ds'] <= end)]
    RH_train.reset_index(drop=True, inplace=True)
    RH_test.reset_index(drop=True, inplace=True)

    rain_train = relative_data_3[
        (relative_data_3['ds'] >= original_data['ds'].min()) & (relative_data_3['ds'] <= pp_date)]
    rain_test = relative_data_3[(relative_data_3['ds'] > pp_date) & (relative_data_3['ds'] <= end)]
    rain_train.reset_index(drop=True, inplace=True)
    rain_test.reset_index(drop=True, inplace=True)

    sun_train = relative_data_4[
        (relative_data_4['ds'] >= original_data['ds'].min()) & (relative_data_4['ds'] <= pp_date)]
    sun_test = relative_data_4[(relative_data_4['ds'] > pp_date) & (relative_data_4['ds'] <= end)]
    sun_train.reset_index(drop=True, inplace=True)
    sun_test.reset_index(drop=True, inplace=True)

    wind_train = relative_data_5[
        (relative_data_5['ds'] >= original_data['ds'].min()) & (relative_data_5['ds'] <= pp_date)]
    wind_test = relative_data_5[(relative_data_5['ds'] > pp_date) & (relative_data_5['ds'] <= end)]
    wind_train.reset_index(drop=True, inplace=True)
    wind_test.reset_index(drop=True, inplace=True)

    # 设置模型参数。“1”表示对应项以加法方式进入模型，“2”表示对应项以乘法方式进入模型。
    m222 = Prophet(holidays=holiday, holidays_prior_scale=0.25, changepoint_prior_scale=0.01,
                   seasonality_mode='multiplicative', yearly_seasonality=False, weekly_seasonality=False,
                   daily_seasonality=False)
    # 内置固定及移动节假日
    m222.add_country_holidays(country_name='CN')
    # 设置多重季节性
    m222.add_seasonality(name='yearly', period=365, fourier_order=12, prior_scale=0.1, mode='multiplicative')
    m222.add_seasonality(name='seasonly', period=365 / 4, fourier_order=10, prior_scale=0.1, mode='multiplicative')
    m222.add_seasonality(name='monthly', period=365 / 12, fourier_order=8, prior_scale=0.1, mode='multiplicative')
    m222.add_seasonality(name='weekly', period=7, fourier_order=6, prior_scale=0.1, mode='multiplicative')
    # 设置相关变量
    m222.add_regressor(tempt_train.columns[1], prior_scale=0.03, mode='multiplicative')
    m222.add_regressor(RH_train.columns[1], prior_scale=0.03, mode='multiplicative')
    m222.add_regressor(rain_train.columns[1], prior_scale=0.03, mode='multiplicative')
    m222.add_regressor(sun_train.columns[1], prior_scale=0.03, mode='multiplicative')
    m222.add_regressor(wind_train.columns[1], prior_scale=0.03, mode='multiplicative')
    # 拟合及预测数据
    datestamp = pd.date_range(end=pp_date, periods=len(original_data))
    df_train = pd.DataFrame({'ds': datestamp, 'y': original_data['y'],
                             tempt_train.columns[1]: tempt_train[tempt_train.columns[1]],
                             RH_train.columns[1]: RH_train[RH_train.columns[1]],
                             rain_train.columns[1]: rain_train[rain_train.columns[1]],
                             sun_train.columns[1]: sun_train[sun_train.columns[1]],
                             wind_train.columns[1]: wind_train[wind_train.columns[1]],
                             })
    m222.fit(df_train)
    future = m222.make_future_dataframe(periods=fc, freq='1D', include_history=False)
    future222 = pd.concat([future, tempt_test[tempt_test.columns[1]], RH_test[RH_test.columns[1]],
                           rain_test[rain_test.columns[1]], sun_test[sun_test.columns[1]],
                           wind_test[wind_test.columns[1]]], axis=1)
    forecast222 = m222.predict(future222)
    # 输出预测结果
    prophet_amou = forecast222['yhat'].values
    trend_amou = forecast222['trend'].values
    holiday_amou = forecast222['holidays'].values
    seasonality_amou = forecast222['yearly'].values + forecast222['seasonly'].values \
                       + forecast222['monthly'].values + forecast222['weekly'].values
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


def prophet_func_112(pp_date=data_train['ds'].max(), original_data=data_train, fc=150,
                     relative_data_1=tempt, relative_data_2=RH, relative_data_3=rain,
                     relative_data_4=sun, relative_data_5=wind, holiday=holidays_df):
    '''
    prophet预测,节假日项、季节项以加法方式进入模型，相关变量以乘法方式进入模型。
    '''

    # 将相关变量切分为训练集和测试集，训练集用在Prophet.fit()中训练模型，测试集用在Prophet.predict()中生成目标变量的预测数据。
    end = str(pd.date_range(start=original_data['ds'].max(), periods=fc + 1).max())

    tempt_train = relative_data_1[
        (relative_data_1['ds'] >= original_data['ds'].min()) & (relative_data_1['ds'] <= pp_date)]
    tempt_test = relative_data_1[(relative_data_1['ds'] > pp_date) & (relative_data_1['ds'] <= end)]
    tempt_train.reset_index(drop=True, inplace=True)
    tempt_test.reset_index(drop=True, inplace=True)

    RH_train = relative_data_2[
        (relative_data_2['ds'] >= original_data['ds'].min()) & (relative_data_2['ds'] <= pp_date)]
    RH_test = relative_data_2[(relative_data_2['ds'] > pp_date) & (relative_data_2['ds'] <= end)]
    RH_train.reset_index(drop=True, inplace=True)
    RH_test.reset_index(drop=True, inplace=True)

    rain_train = relative_data_3[
        (relative_data_3['ds'] >= original_data['ds'].min()) & (relative_data_3['ds'] <= pp_date)]
    rain_test = relative_data_3[(relative_data_3['ds'] > pp_date) & (relative_data_3['ds'] <= end)]
    rain_train.reset_index(drop=True, inplace=True)
    rain_test.reset_index(drop=True, inplace=True)

    sun_train = relative_data_4[
        (relative_data_4['ds'] >= original_data['ds'].min()) & (relative_data_4['ds'] <= pp_date)]
    sun_test = relative_data_4[(relative_data_4['ds'] > pp_date) & (relative_data_4['ds'] <= end)]
    sun_train.reset_index(drop=True, inplace=True)
    sun_test.reset_index(drop=True, inplace=True)

    wind_train = relative_data_5[
        (relative_data_5['ds'] >= original_data['ds'].min()) & (relative_data_5['ds'] <= pp_date)]
    wind_test = relative_data_5[(relative_data_5['ds'] > pp_date) & (relative_data_5['ds'] <= end)]
    wind_train.reset_index(drop=True, inplace=True)
    wind_test.reset_index(drop=True, inplace=True)

    # 设置模型参数。“1”表示对应项以加法方式进入模型，“2”表示对应项以乘法方式进入模型。
    m112 = Prophet(holidays=holiday, holidays_prior_scale=0.25, changepoint_prior_scale=0.01,
                   seasonality_mode='additive', yearly_seasonality=False, weekly_seasonality=False,
                   daily_seasonality=False)
    # 内置固定及移动节假日
    m112.add_country_holidays(country_name='CN')
    # 设置多重季节性
    m112.add_seasonality(name='yearly', period=365, fourier_order=12, prior_scale=0.1, mode='additive')
    m112.add_seasonality(name='seasonly', period=365 / 4, fourier_order=10, prior_scale=0.1, mode='additive')
    m112.add_seasonality(name='monthly', period=365 / 12, fourier_order=8, prior_scale=0.1, mode='additive')
    m112.add_seasonality(name='weekly', period=7, fourier_order=6, prior_scale=0.1, mode='additive')
    # 设置相关变量
    m112.add_regressor(tempt_train.columns[1], prior_scale=0.03, mode='multiplicative')
    m112.add_regressor(RH_train.columns[1], prior_scale=0.03, mode='multiplicative')
    m112.add_regressor(rain_train.columns[1], prior_scale=0.03, mode='multiplicative')
    m112.add_regressor(sun_train.columns[1], prior_scale=0.03, mode='multiplicative')
    m112.add_regressor(wind_train.columns[1], prior_scale=0.03, mode='multiplicative')
    # 拟合及预测数据
    datestamp = pd.date_range(end=pp_date, periods=len(original_data))
    df_train = pd.DataFrame({'ds': datestamp, 'y': original_data['y'],
                             tempt_train.columns[1]: tempt_train[tempt_train.columns[1]],
                             RH_train.columns[1]: RH_train[RH_train.columns[1]],
                             rain_train.columns[1]: rain_train[rain_train.columns[1]],
                             sun_train.columns[1]: sun_train[sun_train.columns[1]],
                             wind_train.columns[1]: wind_train[wind_train.columns[1]],
                             })
    m112.fit(df_train)
    future = m112.make_future_dataframe(periods=fc, freq='1D', include_history=False)
    future112 = pd.concat([future, tempt_test[tempt_test.columns[1]], RH_test[RH_test.columns[1]],
                           rain_test[rain_test.columns[1]], sun_test[sun_test.columns[1]],
                           wind_test[wind_test.columns[1]]], axis=1)
    forecast112 = m112.predict(future112)
    # 输出预测结果
    prophet_amou = forecast112['yhat'].values
    trend_amou = forecast112['trend'].values
    holiday_amou = forecast112['holidays'].values
    seasonality_amou = forecast112['yearly'].values + forecast112['seasonly'].values \
                       + forecast112['monthly'].values + forecast112['weekly'].values
    relative_amou = forecast112['extra_regressors_multiplicative'].values

    if False not in (prophet_amou - (trend_amou * (1+relative_amou) + holiday_amou + seasonality_amou) < 1e-10):
        prophet_amou_1 = prophet_amou
        trend_amou_1 = trend_amou
        holiday_amou_1 = holiday_amou
        seasonality_amou_1 = seasonality_amou
        relative_amou_1 = relative_amou
        prophet_amou_1 = [0 if i < 0 else i for i in prophet_amou_1]  # 只有prophet_amou_1的负值预测需用0代替，其余不能。
        return prophet_amou_1, trend_amou_1, holiday_amou_1, seasonality_amou_1, relative_amou_1
    else:
        return None


def prophet_func_121(pp_date=data_train['ds'].max(), original_data=data_train, fc=150,
                     relative_data_1=tempt, relative_data_2=RH, relative_data_3=rain,
                     relative_data_4=sun, relative_data_5=wind, holiday=holidays_df):
    '''
    prophet预测,节假日项、相关变量以加法方式进入模型，季节项以乘法方式进入模型。
    '''

    # 将相关变量切分为训练集和测试集，训练集用在Prophet.fit()中训练模型，测试集用在Prophet.predict()中生成目标变量的预测数据。
    end = str(pd.date_range(start=original_data['ds'].max(), periods=fc + 1).max())

    tempt_train = relative_data_1[
        (relative_data_1['ds'] >= original_data['ds'].min()) & (relative_data_1['ds'] <= pp_date)]
    tempt_test = relative_data_1[(relative_data_1['ds'] > pp_date) & (relative_data_1['ds'] <= end)]
    tempt_train.reset_index(drop=True, inplace=True)
    tempt_test.reset_index(drop=True, inplace=True)

    RH_train = relative_data_2[
        (relative_data_2['ds'] >= original_data['ds'].min()) & (relative_data_2['ds'] <= pp_date)]
    RH_test = relative_data_2[(relative_data_2['ds'] > pp_date) & (relative_data_2['ds'] <= end)]
    RH_train.reset_index(drop=True, inplace=True)
    RH_test.reset_index(drop=True, inplace=True)

    rain_train = relative_data_3[
        (relative_data_3['ds'] >= original_data['ds'].min()) & (relative_data_3['ds'] <= pp_date)]
    rain_test = relative_data_3[(relative_data_3['ds'] > pp_date) & (relative_data_3['ds'] <= end)]
    rain_train.reset_index(drop=True, inplace=True)
    rain_test.reset_index(drop=True, inplace=True)

    sun_train = relative_data_4[
        (relative_data_4['ds'] >= original_data['ds'].min()) & (relative_data_4['ds'] <= pp_date)]
    sun_test = relative_data_4[(relative_data_4['ds'] > pp_date) & (relative_data_4['ds'] <= end)]
    sun_train.reset_index(drop=True, inplace=True)
    sun_test.reset_index(drop=True, inplace=True)

    wind_train = relative_data_5[
        (relative_data_5['ds'] >= original_data['ds'].min()) & (relative_data_5['ds'] <= pp_date)]
    wind_test = relative_data_5[(relative_data_5['ds'] > pp_date) & (relative_data_5['ds'] <= end)]
    wind_train.reset_index(drop=True, inplace=True)
    wind_test.reset_index(drop=True, inplace=True)

    # 设置模型参数。“1”表示对应项以加法方式进入模型，“2”表示对应项以乘法方式进入模型。
    m121 = Prophet(holidays=holiday, holidays_prior_scale=0.25, changepoint_prior_scale=0.01,
                   seasonality_mode='additive', yearly_seasonality=False, weekly_seasonality=False,
                   daily_seasonality=False)
    # 内置固定及移动节假日
    m121.add_country_holidays(country_name='CN')
    # 设置多重季节性
    m121.add_seasonality(name='yearly', period=365, fourier_order=12, prior_scale=0.1, mode='multiplicative')
    m121.add_seasonality(name='seasonly', period=365 / 4, fourier_order=10, prior_scale=0.1, mode='multiplicative')
    m121.add_seasonality(name='monthly', period=365 / 12, fourier_order=8, prior_scale=0.1, mode='multiplicative')
    m121.add_seasonality(name='weekly', period=7, fourier_order=6, prior_scale=0.1, mode='multiplicative')
    # 设置相关变量
    m121.add_regressor(tempt_train.columns[1], prior_scale=0.03, mode='additive')
    m121.add_regressor(RH_train.columns[1], prior_scale=0.03, mode='additive')
    m121.add_regressor(rain_train.columns[1], prior_scale=0.03, mode='additive')
    m121.add_regressor(sun_train.columns[1], prior_scale=0.03, mode='additive')
    m121.add_regressor(wind_train.columns[1], prior_scale=0.03, mode='additive')
    # 拟合及预测数据
    datestamp = pd.date_range(end=pp_date, periods=len(original_data))
    df_train = pd.DataFrame({'ds': datestamp, 'y': original_data['y'],
                             tempt_train.columns[1]: tempt_train[tempt_train.columns[1]],
                             RH_train.columns[1]: RH_train[RH_train.columns[1]],
                             rain_train.columns[1]: rain_train[rain_train.columns[1]],
                             sun_train.columns[1]: sun_train[sun_train.columns[1]],
                             wind_train.columns[1]: wind_train[wind_train.columns[1]],
                             })
    m121.fit(df_train)
    future = m121.make_future_dataframe(periods=fc, freq='1D', include_history=False)
    future121 = pd.concat([future, tempt_test[tempt_test.columns[1]], RH_test[RH_test.columns[1]],
                           rain_test[rain_test.columns[1]], sun_test[sun_test.columns[1]],
                           wind_test[wind_test.columns[1]]], axis=1)
    forecast121 = m121.predict(future121)
    # 输出预测结果
    prophet_amou = forecast121['yhat'].values
    trend_amou = forecast121['trend'].values
    holiday_amou = forecast121['holidays'].values
    seasonality_amou = forecast121['yearly'].values + forecast121['seasonly'].values \
                       + forecast121['monthly'].values + forecast121['weekly'].values
    relative_amou = forecast121['extra_regressors_additive'].values

    if False not in (prophet_amou - (trend_amou * (1+seasonality_amou) + holiday_amou + relative_amou) < 1e-10):
        prophet_amou_1 = prophet_amou
        trend_amou_1 = trend_amou
        holiday_amou_1 = holiday_amou
        seasonality_amou_1 = seasonality_amou
        relative_amou_1 = relative_amou
        prophet_amou_1 = [0 if i < 0 else i for i in prophet_amou_1]  # 只有prophet_amou_1的负值预测需用0代替，其余不能。
        return prophet_amou_1, trend_amou_1, holiday_amou_1, seasonality_amou_1, relative_amou_1
    else:
        return None


def prophet_func_122(pp_date=data_train['ds'].max(), original_data=data_train, fc=150,
                     relative_data_1=tempt, relative_data_2=RH, relative_data_3=rain,
                     relative_data_4=sun, relative_data_5=wind, holiday=holidays_df):
    '''
    prophet预测,节假日项以加法方式进入模型，季节项相、关变量以乘法方式进入模型。
    '''

    # 将相关变量切分为训练集和测试集，训练集用在Prophet.fit()中训练模型，测试集用在Prophet.predict()中生成目标变量的预测数据。
    end = str(pd.date_range(start=original_data['ds'].max(), periods=fc + 1).max())

    tempt_train = relative_data_1[
        (relative_data_1['ds'] >= original_data['ds'].min()) & (relative_data_1['ds'] <= pp_date)]
    tempt_test = relative_data_1[(relative_data_1['ds'] > pp_date) & (relative_data_1['ds'] <= end)]
    tempt_train.reset_index(drop=True, inplace=True)
    tempt_test.reset_index(drop=True, inplace=True)

    RH_train = relative_data_2[
        (relative_data_2['ds'] >= original_data['ds'].min()) & (relative_data_2['ds'] <= pp_date)]
    RH_test = relative_data_2[(relative_data_2['ds'] > pp_date) & (relative_data_2['ds'] <= end)]
    RH_train.reset_index(drop=True, inplace=True)
    RH_test.reset_index(drop=True, inplace=True)

    rain_train = relative_data_3[
        (relative_data_3['ds'] >= original_data['ds'].min()) & (relative_data_3['ds'] <= pp_date)]
    rain_test = relative_data_3[(relative_data_3['ds'] > pp_date) & (relative_data_3['ds'] <= end)]
    rain_train.reset_index(drop=True, inplace=True)
    rain_test.reset_index(drop=True, inplace=True)

    sun_train = relative_data_4[
        (relative_data_4['ds'] >= original_data['ds'].min()) & (relative_data_4['ds'] <= pp_date)]
    sun_test = relative_data_4[(relative_data_4['ds'] > pp_date) & (relative_data_4['ds'] <= end)]
    sun_train.reset_index(drop=True, inplace=True)
    sun_test.reset_index(drop=True, inplace=True)

    wind_train = relative_data_5[
        (relative_data_5['ds'] >= original_data['ds'].min()) & (relative_data_5['ds'] <= pp_date)]
    wind_test = relative_data_5[(relative_data_5['ds'] > pp_date) & (relative_data_5['ds'] <= end)]
    wind_train.reset_index(drop=True, inplace=True)
    wind_test.reset_index(drop=True, inplace=True)

    # 设置模型参数。“1”表示对应项以加法方式进入模型，“2”表示对应项以乘法方式进入模型。
    m122 = Prophet(holidays=holiday, holidays_prior_scale=0.25, changepoint_prior_scale=0.01,
                   seasonality_mode='additive', yearly_seasonality=False, weekly_seasonality=False,
                   daily_seasonality=False)
    # 内置固定及移动节假日
    m122.add_country_holidays(country_name='CN')
    # 设置多重季节性
    m122.add_seasonality(name='yearly', period=365, fourier_order=12, prior_scale=0.1, mode='multiplicative')
    m122.add_seasonality(name='seasonly', period=365 / 4, fourier_order=10, prior_scale=0.1, mode='multiplicative')
    m122.add_seasonality(name='monthly', period=365 / 12, fourier_order=8, prior_scale=0.1, mode='multiplicative')
    m122.add_seasonality(name='weekly', period=7, fourier_order=6, prior_scale=0.1, mode='multiplicative')
    # 设置相关变量
    m122.add_regressor(tempt_train.columns[1], prior_scale=0.03, mode='multiplicative')
    m122.add_regressor(RH_train.columns[1], prior_scale=0.03, mode='multiplicative')
    m122.add_regressor(rain_train.columns[1], prior_scale=0.03, mode='multiplicative')
    m122.add_regressor(sun_train.columns[1], prior_scale=0.03, mode='multiplicative')
    m122.add_regressor(wind_train.columns[1], prior_scale=0.03, mode='multiplicative')
    # 拟合及预测数据
    datestamp = pd.date_range(end=pp_date, periods=len(original_data))
    df_train = pd.DataFrame({'ds': datestamp, 'y': original_data['y'],
                             tempt_train.columns[1]: tempt_train[tempt_train.columns[1]],
                             RH_train.columns[1]: RH_train[RH_train.columns[1]],
                             rain_train.columns[1]: rain_train[rain_train.columns[1]],
                             sun_train.columns[1]: sun_train[sun_train.columns[1]],
                             wind_train.columns[1]: wind_train[wind_train.columns[1]],
                             })
    m122.fit(df_train)
    future = m122.make_future_dataframe(periods=fc, freq='1D', include_history=False)
    future122 = pd.concat([future, tempt_test[tempt_test.columns[1]], RH_test[RH_test.columns[1]],
                           rain_test[rain_test.columns[1]], sun_test[sun_test.columns[1]],
                           wind_test[wind_test.columns[1]]], axis=1)
    forecast122 = m122.predict(future122)
    # 输出预测结果
    prophet_amou = forecast122['yhat'].values
    trend_amou = forecast122['trend'].values
    holiday_amou = forecast122['holidays'].values
    seasonality_amou = forecast122['yearly'].values + forecast122['seasonly'].values \
                       + forecast122['monthly'].values + forecast122['weekly'].values
    relative_amou = forecast122['extra_regressors_multiplicative'].values

    if False not in (prophet_amou - (trend_amou * (1+seasonality_amou+relative_amou) + holiday_amou) < 1e-10):
        prophet_amou_1 = prophet_amou
        trend_amou_1 = trend_amou
        holiday_amou_1 = holiday_amou
        seasonality_amou_1 = seasonality_amou
        relative_amou_1 = relative_amou
        prophet_amou_1 = [0 if i < 0 else i for i in prophet_amou_1]  # 只有prophet_amou_1的负值预测需用0代替，其余不能。
        return prophet_amou_1, trend_amou_1, holiday_amou_1, seasonality_amou_1, relative_amou_1
    else:
        return None


def prophet_func_211(pp_date=data_train['ds'].max(), original_data=data_train, fc=150,
                     relative_data_1=tempt, relative_data_2=RH, relative_data_3=rain,
                     relative_data_4=sun, relative_data_5=wind, holiday=holidays_df):
    '''
    prophet预测，节假日项以乘法方式进入模型，季节项相、关变量以加法方式进入模型。
    '''

    # 将相关变量切分为训练集和测试集，训练集用在Prophet.fit()中训练模型，测试集用在Prophet.predict()中生成目标变量的预测数据。
    end = str(pd.date_range(start=original_data['ds'].max(), periods=fc + 1).max())

    tempt_train = relative_data_1[
        (relative_data_1['ds'] >= original_data['ds'].min()) & (relative_data_1['ds'] <= pp_date)]
    tempt_test = relative_data_1[(relative_data_1['ds'] > pp_date) & (relative_data_1['ds'] <= end)]
    tempt_train.reset_index(drop=True, inplace=True)
    tempt_test.reset_index(drop=True, inplace=True)

    RH_train = relative_data_2[
        (relative_data_2['ds'] >= original_data['ds'].min()) & (relative_data_2['ds'] <= pp_date)]
    RH_test = relative_data_2[(relative_data_2['ds'] > pp_date) & (relative_data_2['ds'] <= end)]
    RH_train.reset_index(drop=True, inplace=True)
    RH_test.reset_index(drop=True, inplace=True)

    rain_train = relative_data_3[
        (relative_data_3['ds'] >= original_data['ds'].min()) & (relative_data_3['ds'] <= pp_date)]
    rain_test = relative_data_3[(relative_data_3['ds'] > pp_date) & (relative_data_3['ds'] <= end)]
    rain_train.reset_index(drop=True, inplace=True)
    rain_test.reset_index(drop=True, inplace=True)

    sun_train = relative_data_4[
        (relative_data_4['ds'] >= original_data['ds'].min()) & (relative_data_4['ds'] <= pp_date)]
    sun_test = relative_data_4[(relative_data_4['ds'] > pp_date) & (relative_data_4['ds'] <= end)]
    sun_train.reset_index(drop=True, inplace=True)
    sun_test.reset_index(drop=True, inplace=True)

    wind_train = relative_data_5[
        (relative_data_5['ds'] >= original_data['ds'].min()) & (relative_data_5['ds'] <= pp_date)]
    wind_test = relative_data_5[(relative_data_5['ds'] > pp_date) & (relative_data_5['ds'] <= end)]
    wind_train.reset_index(drop=True, inplace=True)
    wind_test.reset_index(drop=True, inplace=True)

    # 设置模型参数。“1”表示对应项以加法方式进入模型，“2”表示对应项以乘法方式进入模型。
    m211 = Prophet(holidays=holiday, holidays_prior_scale=0.25, changepoint_prior_scale=0.01,
                   seasonality_mode='multiplicative', yearly_seasonality=False, weekly_seasonality=False,
                   daily_seasonality=False)
    # 内置固定及移动节假日
    m211.add_country_holidays(country_name='CN')
    # 设置多重季节性
    m211.add_seasonality(name='yearly', period=365, fourier_order=12, prior_scale=0.1, mode='additive')
    m211.add_seasonality(name='seasonly', period=365 / 4, fourier_order=10, prior_scale=0.1, mode='additive')
    m211.add_seasonality(name='monthly', period=365 / 12, fourier_order=8, prior_scale=0.1, mode='additive')
    m211.add_seasonality(name='weekly', period=7, fourier_order=6, prior_scale=0.1, mode='additive')
    # 设置相关变量
    m211.add_regressor(tempt_train.columns[1], prior_scale=0.03, mode='additive')
    m211.add_regressor(RH_train.columns[1], prior_scale=0.03, mode='additive')
    m211.add_regressor(rain_train.columns[1], prior_scale=0.03, mode='additive')
    m211.add_regressor(sun_train.columns[1], prior_scale=0.03, mode='additive')
    m211.add_regressor(wind_train.columns[1], prior_scale=0.03, mode='additive')
    # 拟合及预测数据
    datestamp = pd.date_range(end=pp_date, periods=len(original_data))
    df_train = pd.DataFrame({'ds': datestamp, 'y': original_data['y'],
                             tempt_train.columns[1]: tempt_train[tempt_train.columns[1]],
                             RH_train.columns[1]: RH_train[RH_train.columns[1]],
                             rain_train.columns[1]: rain_train[rain_train.columns[1]],
                             sun_train.columns[1]: sun_train[sun_train.columns[1]],
                             wind_train.columns[1]: wind_train[wind_train.columns[1]],
                             })
    m211.fit(df_train)
    future = m211.make_future_dataframe(periods=fc, freq='1D', include_history=False)
    future211 = pd.concat([future, tempt_test[tempt_test.columns[1]], RH_test[RH_test.columns[1]],
                           rain_test[rain_test.columns[1]], sun_test[sun_test.columns[1]],
                           wind_test[wind_test.columns[1]]], axis=1)
    forecast211 = m211.predict(future211)
    # 输出图形及数值预测结果
    prophet_amou = forecast211['yhat'].values
    trend_amou = forecast211['trend'].values
    holiday_amou = forecast211['holidays'].values
    seasonality_amou = forecast211['yearly'].values + forecast211['seasonly'].values \
                       + forecast211['monthly'].values + forecast211['weekly'].values
    relative_amou = forecast211['extra_regressors_additive'].values

    if False not in (prophet_amou - (trend_amou * (1+holiday_amou) + seasonality_amou + relative_amou) < 1e-10):
        prophet_amou_1 = prophet_amou
        trend_amou_1 = trend_amou
        holiday_amou_1 = holiday_amou
        seasonality_amou_1 = seasonality_amou
        relative_amou_1 = relative_amou
        prophet_amou_1 = [0 if i < 0 else i for i in prophet_amou_1]  # 只有prophet_amou_1的负值预测需用0代替，其余不能。
        return prophet_amou_1, trend_amou_1, holiday_amou_1, seasonality_amou_1, relative_amou_1
    else:
        return None


def prophet_func_212(pp_date=data_train['ds'].max(), original_data=data_train, fc=150,
                     relative_data_1=tempt, relative_data_2=RH, relative_data_3=rain,
                     relative_data_4=sun, relative_data_5=wind, holiday=holidays_df):
    '''
    prophet预测，节假日项、相关变量以乘法方式进入模型，季节项以加法方式进入模型。
    '''

    # 将相关变量切分为训练集和测试集，训练集用在Prophet.fit()中训练模型，测试集用在Prophet.predict()中生成目标变量的预测数据。
    end = str(pd.date_range(start=original_data['ds'].max(), periods=fc + 1).max())

    tempt_train = relative_data_1[
        (relative_data_1['ds'] >= original_data['ds'].min()) & (relative_data_1['ds'] <= pp_date)]
    tempt_test = relative_data_1[(relative_data_1['ds'] > pp_date) & (relative_data_1['ds'] <= end)]
    tempt_train.reset_index(drop=True, inplace=True)
    tempt_test.reset_index(drop=True, inplace=True)

    RH_train = relative_data_2[
        (relative_data_2['ds'] >= original_data['ds'].min()) & (relative_data_2['ds'] <= pp_date)]
    RH_test = relative_data_2[(relative_data_2['ds'] > pp_date) & (relative_data_2['ds'] <= end)]
    RH_train.reset_index(drop=True, inplace=True)
    RH_test.reset_index(drop=True, inplace=True)

    rain_train = relative_data_3[
        (relative_data_3['ds'] >= original_data['ds'].min()) & (relative_data_3['ds'] <= pp_date)]
    rain_test = relative_data_3[(relative_data_3['ds'] > pp_date) & (relative_data_3['ds'] <= end)]
    rain_train.reset_index(drop=True, inplace=True)
    rain_test.reset_index(drop=True, inplace=True)

    sun_train = relative_data_4[
        (relative_data_4['ds'] >= original_data['ds'].min()) & (relative_data_4['ds'] <= pp_date)]
    sun_test = relative_data_4[(relative_data_4['ds'] > pp_date) & (relative_data_4['ds'] <= end)]
    sun_train.reset_index(drop=True, inplace=True)
    sun_test.reset_index(drop=True, inplace=True)

    wind_train = relative_data_5[
        (relative_data_5['ds'] >= original_data['ds'].min()) & (relative_data_5['ds'] <= pp_date)]
    wind_test = relative_data_5[(relative_data_5['ds'] > pp_date) & (relative_data_5['ds'] <= end)]
    wind_train.reset_index(drop=True, inplace=True)
    wind_test.reset_index(drop=True, inplace=True)

    # 设置模型参数。“1”表示对应项以加法方式进入模型，“2”表示对应项以乘法方式进入模型。
    m212 = Prophet(holidays=holiday, holidays_prior_scale=0.25, changepoint_prior_scale=0.01,
                   seasonality_mode='multiplicative', yearly_seasonality=False, weekly_seasonality=False,
                   daily_seasonality=False)
    # 内置固定及移动节假日
    m212.add_country_holidays(country_name='CN')
    # 设置多重季节性
    m212.add_seasonality(name='yearly', period=365, fourier_order=12, prior_scale=0.1, mode='additive')
    m212.add_seasonality(name='seasonly', period=365 / 4, fourier_order=10, prior_scale=0.1, mode='additive')
    m212.add_seasonality(name='monthly', period=365 / 12, fourier_order=8, prior_scale=0.1, mode='additive')
    m212.add_seasonality(name='weekly', period=7, fourier_order=6, prior_scale=0.1, mode='additive')
    # 设置相关变量
    m212.add_regressor(tempt_train.columns[1], prior_scale=0.03, mode='multiplicative')
    m212.add_regressor(RH_train.columns[1], prior_scale=0.03, mode='multiplicative')
    m212.add_regressor(rain_train.columns[1], prior_scale=0.03, mode='multiplicative')
    m212.add_regressor(sun_train.columns[1], prior_scale=0.03, mode='multiplicative')
    m212.add_regressor(wind_train.columns[1], prior_scale=0.03, mode='multiplicative')
    # 拟合及预测数据
    datestamp = pd.date_range(end=pp_date, periods=len(original_data))
    df_train = pd.DataFrame({'ds': datestamp, 'y': original_data['y'],
                             tempt_train.columns[1]: tempt_train[tempt_train.columns[1]],
                             RH_train.columns[1]: RH_train[RH_train.columns[1]],
                             rain_train.columns[1]: rain_train[rain_train.columns[1]],
                             sun_train.columns[1]: sun_train[sun_train.columns[1]],
                             wind_train.columns[1]: wind_train[wind_train.columns[1]],
                             })
    m212.fit(df_train)
    future = m212.make_future_dataframe(periods=fc, freq='1D', include_history=False)
    future212 = pd.concat([future, tempt_test[tempt_test.columns[1]], RH_test[RH_test.columns[1]],
                           rain_test[rain_test.columns[1]], sun_test[sun_test.columns[1]],
                           wind_test[wind_test.columns[1]]], axis=1)
    forecast212 = m212.predict(future212)
    # 输出预测结果
    prophet_amou = forecast212['yhat'].values
    trend_amou = forecast212['trend'].values
    holiday_amou = forecast212['holidays'].values
    seasonality_amou = forecast212['yearly'].values + forecast212['seasonly'].values \
                       + forecast212['monthly'].values + forecast212['weekly'].values
    relative_amou = forecast212['extra_regressors_multiplicative'].values

    if False not in (prophet_amou - (trend_amou * (1+holiday_amou+relative_amou) + seasonality_amou) < 1e-10):
        prophet_amou_1 = prophet_amou
        trend_amou_1 = trend_amou
        holiday_amou_1 = holiday_amou
        seasonality_amou_1 = seasonality_amou
        relative_amou_1 = relative_amou
        prophet_amou_1 = [0 if i < 0 else i for i in prophet_amou_1]  # 只有prophet_amou_1的负值预测需用0代替，其余不能。
        return prophet_amou_1, trend_amou_1, holiday_amou_1, seasonality_amou_1, relative_amou_1
    else:
        return None


def prophet_func_221(pp_date=data_train['ds'].max(), original_data=data_train, fc=150,
                     relative_data_1=tempt, relative_data_2=RH, relative_data_3=rain,
                     relative_data_4=sun, relative_data_5=wind, holiday=holidays_df):
    '''
    prophet预测，节假日项、季节项以乘法方式进入模型，相关变量以加法方式进入模型。
    '''

    # 将相关变量切分为训练集和测试集，训练集用在Prophet.fit()中训练模型，测试集用在Prophet.predict()中生成目标变量的预测数据。
    end = str(pd.date_range(start=original_data['ds'].max(), periods=fc + 1).max())

    tempt_train = relative_data_1[
        (relative_data_1['ds'] >= original_data['ds'].min()) & (relative_data_1['ds'] <= pp_date)]
    tempt_test = relative_data_1[(relative_data_1['ds'] > pp_date) & (relative_data_1['ds'] <= end)]
    tempt_train.reset_index(drop=True, inplace=True)
    tempt_test.reset_index(drop=True, inplace=True)

    RH_train = relative_data_2[
        (relative_data_2['ds'] >= original_data['ds'].min()) & (relative_data_2['ds'] <= pp_date)]
    RH_test = relative_data_2[(relative_data_2['ds'] > pp_date) & (relative_data_2['ds'] <= end)]
    RH_train.reset_index(drop=True, inplace=True)
    RH_test.reset_index(drop=True, inplace=True)

    rain_train = relative_data_3[
        (relative_data_3['ds'] >= original_data['ds'].min()) & (relative_data_3['ds'] <= pp_date)]
    rain_test = relative_data_3[(relative_data_3['ds'] > pp_date) & (relative_data_3['ds'] <= end)]
    rain_train.reset_index(drop=True, inplace=True)
    rain_test.reset_index(drop=True, inplace=True)

    sun_train = relative_data_4[
        (relative_data_4['ds'] >= original_data['ds'].min()) & (relative_data_4['ds'] <= pp_date)]
    sun_test = relative_data_4[(relative_data_4['ds'] > pp_date) & (relative_data_4['ds'] <= end)]
    sun_train.reset_index(drop=True, inplace=True)
    sun_test.reset_index(drop=True, inplace=True)

    wind_train = relative_data_5[
        (relative_data_5['ds'] >= original_data['ds'].min()) & (relative_data_5['ds'] <= pp_date)]
    wind_test = relative_data_5[(relative_data_5['ds'] > pp_date) & (relative_data_5['ds'] <= end)]
    wind_train.reset_index(drop=True, inplace=True)
    wind_test.reset_index(drop=True, inplace=True)

    # 设置模型参数。“1”表示对应项以加法方式进入模型，“2”表示对应项以乘法方式进入模型。
    m221 = Prophet(holidays=holiday, holidays_prior_scale=0.25, changepoint_prior_scale=0.01,
                   seasonality_mode='multiplicative', yearly_seasonality=False, weekly_seasonality=False,
                   daily_seasonality=False)
    # 内置固定及移动节假日
    m221.add_country_holidays(country_name='CN')
    # 设置多重季节性
    m221.add_seasonality(name='yearly', period=365, fourier_order=12, prior_scale=0.1, mode='multiplicative')
    m221.add_seasonality(name='seasonly', period=365 / 4, fourier_order=10, prior_scale=0.1, mode='multiplicative')
    m221.add_seasonality(name='monthly', period=365 / 12, fourier_order=8, prior_scale=0.1, mode='multiplicative')
    m221.add_seasonality(name='weekly', period=7, fourier_order=6, prior_scale=0.1, mode='multiplicative')
    # 设置相关变量
    m221.add_regressor(tempt_train.columns[1], prior_scale=0.03, mode='additive')
    m221.add_regressor(RH_train.columns[1], prior_scale=0.03, mode='additive')
    m221.add_regressor(rain_train.columns[1], prior_scale=0.03, mode='additive')
    m221.add_regressor(sun_train.columns[1], prior_scale=0.03, mode='additive')
    m221.add_regressor(wind_train.columns[1], prior_scale=0.03, mode='additive')
    # 拟合及预测数据
    datestamp = pd.date_range(end=pp_date, periods=len(original_data))
    df_train = pd.DataFrame({'ds': datestamp, 'y': original_data['y'],
                             tempt_train.columns[1]: tempt_train[tempt_train.columns[1]],
                             RH_train.columns[1]: RH_train[RH_train.columns[1]],
                             rain_train.columns[1]: rain_train[rain_train.columns[1]],
                             sun_train.columns[1]: sun_train[sun_train.columns[1]],
                             wind_train.columns[1]: wind_train[wind_train.columns[1]],
                             })
    m221.fit(df_train)
    future = m221.make_future_dataframe(periods=fc, freq='1D', include_history=False)
    future221 = pd.concat([future, tempt_test[tempt_test.columns[1]], RH_test[RH_test.columns[1]],
                           rain_test[rain_test.columns[1]], sun_test[sun_test.columns[1]],
                           wind_test[wind_test.columns[1]]], axis=1)
    forecast221 = m221.predict(future221)
    # 输出图形及数值预测结果
    prophet_amou = forecast221['yhat'].values
    trend_amou = forecast221['trend'].values
    holiday_amou = forecast221['holidays'].values
    seasonality_amou = forecast221['yearly'].values + forecast221['seasonly'].values \
                       + forecast221['monthly'].values + forecast221['weekly'].values
    relative_amou = forecast221['extra_regressors_additive'].values

    if False not in (prophet_amou - (trend_amou * (1+holiday_amou+seasonality_amou) + relative_amou) < 1e-10):
        prophet_amou_1 = prophet_amou
        trend_amou_1 = trend_amou
        holiday_amou_1 = holiday_amou
        seasonality_amou_1 = seasonality_amou
        relative_amou_1 = relative_amou
        prophet_amou_1 = [0 if i < 0 else i for i in prophet_amou_1]  # 只有prophet_amou_1的负值预测需用0代替，其余不能。
        return prophet_amou_1, trend_amou_1, holiday_amou_1, seasonality_amou_1, relative_amou_1
    else:
        return None
