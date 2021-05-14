import pandas as pd
import numpy as np
from prophet import Prophet
import datetime as dt


df = pd.read_excel('/Users/zc/PycharmProjects/fbprophet/gas_2015-2017.xlsx')
df = df[['日期', '修正后用气量']]
df.interpolate(method='spline', order=3, inplace=True)  # 如果用dropna()不带参数，则行数可能会减少
df = df.rename(columns={"日期": "ds", "修正后用气量": "y"})

n = 365
df_train, df_test = df[:-n], df[-n:]
df_test = df_test[['ds']]

lower_window = -4
upper_window = 9
holidays_custom = pd.DataFrame({
  'holiday': 'spring festival',
  'ds': pd.to_datetime(['2015-02-19', '2016-02-08', '2017-01-28']),
  'lower_window': lower_window,
  'upper_window': upper_window})

dates, lower_date, upper_date = [], [], []
i = 0
for i in range(len(str(holidays_custom['ds'][i]).split()[0].split('-'))):  # 使i的长度包含所需要年月日数值，这里是range(0, 3)
    for j in range(len(holidays_custom['ds'])):  # 使j的长度为节假日个数，这里只有3个春节，这里是range(0, 3)
        # print(i, j, '\n')
        dates.append(int(str(holidays_custom['ds'][j]).split()[0].split('-')[i]))

df_dates = pd.DataFrame(np.array(dates).reshape(len(holidays_custom['ds']), len(str(holidays_custom['ds'][i]).split()[0].split('-'))).T)
for i in range(len(df_dates)):
    lower_date.append(dt.date(df_dates[0][i], df_dates[1][i], df_dates[2][i]) + dt.timedelta(days=lower_window))
    upper_date.append(dt.date(df_dates[0][i], df_dates[1][i], df_dates[2][i]) + dt.timedelta(days=upper_window))

print(pd.date_range(lower_date[i], upper_date[i]), '\n', len(pd.date_range(lower_date[i], upper_date[i])), '\n')

date_ranges = []
for _ in range(len(df_dates)):
    date_ranges.append(pd.date_range(lower_date[_], upper_date[_]))
date_ranges = date_ranges[0].union_many([date_ranges[1], date_ranges[2]])


def is_sf_season(ds, date_ranges):
    ds = pd.to_datetime(ds)
    date_ranges = pd.to_datetime(date_ranges)
    if sum(ds == date_ranges) == 1:
        return True
    else:
        return False


df_train['on_season'] = df_train['ds'].apply(is_sf_season, args=(date_ranges, ))  # args必须加(,)否则报错
df_train['off_season'] = ~df_train['ds'].apply(is_sf_season, args=(date_ranges, ))
print(sum(df_train['on_season']), '\n', df_train['on_season'], '\n', sum(df_train['off_season']))

m = Prophet(holidays=holidays_custom, holidays_prior_scale=100,
            yearly_seasonality=False, weekly_seasonality=False, daily_seasonality=False)
# m.add_seasonality(name='monthly', period=30.5, fourier_order=5)
m.add_country_holidays(country_name='China')
# m.add_seasonality(name='weekly', period=7, fourier_order=5, prior_scale=30)
m.add_seasonality(name='yearly', period=365, fourier_order=10, prior_scale=30)
m.add_seasonality(name='weekly_on_sf', period=7, fourier_order=5, prior_scale=30, condition_name='on_season')
m.add_seasonality(name='weekly_off_sf', period=7, fourier_order=10, prior_scale=30, condition_name='off_season')

df_test['on_season'] = df_test['ds'].apply(is_sf_season, args=(date_ranges, ))
df_test['off_season'] = ~df_test['ds'].apply(is_sf_season, args=(date_ranges, ))
forecast = m.fit(df_train).predict(df_test)
print(m.train_holiday_names)
fig1 = m.plot_components(forecast, figsize=(12, 9.5))
fig2 = m.plot(forecast)
