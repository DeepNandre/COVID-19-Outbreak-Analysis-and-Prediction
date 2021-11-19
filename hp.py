#!/usr/bin/env python
# coding: utf-8

# In[98]:
# import schedule
import matplotlib.pyplot as plt
import os
import requests
import time

# Aditya
def sendPlotTelegram(plt):
    plt.savefig('image_to_be_sent.jpg')
    token = "2141823967:AAGTMO6GxFaDZ1q42siBcvNtKSyV_TUt0xQ"
    chat_id = -1001581683297  # chat id
    file = "image_to_be_sent.jpg"
    url = f"https://api.telegram.org/bot{token}/sendPhoto"
    files = {}
    files["photo"] = open(file, "rb")
    requests.get(url, params={"chat_id": chat_id}, files=files)

# sendPlotTelegram(plt)
# os.remove("image_to_be_sent.jpg")

# Aditya
def sendTextTelegram(txt):
    main_string = 'https://api.telegram.org/bot2141823967:AAGTMO6GxFaDZ1q42siBcvNtKSyV_TUt0xQ/sendMessage?chat_id=@nCoV2&text='+txt
    requests.get(main_string)

# st = 'Predictions for next 5 days are : '
#
# sendTextTelegram(st)


import sys


from colorama import init
init(strip=not sys.stdout.isatty()) # strip colors if stdout is redirected
from termcolor import cprint
from termcolor import colored
from pyfiglet import figlet_format

cprint(figlet_format('COVID & FRIENDS', font='starwars'),
       'yellow', 'on_red', attrs=['bold'])

print(colored('importing modules......', 'green'))

print()
import os
from bob_telegram_tools.bot import TelegramBot
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import random
import math
import time
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error
import datetime
import operator
plt.style.use('fivethirtyeight')
# %matplotlib inline
# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

from datetime import date
from datetime import timedelta

# Vedant
# Get today's date
print(colored('finding date......', 'green'))
today = date.today()
# print("Today is: ", today)

# Vedant
# Yesterday date
yesterday = today - timedelta(days = 2)
yesterday = str(yesterday.month)+"-"+str(yesterday.day)+"-"+str(yesterday.year)
# print(yesterday)

# Vedamt
# In[100]:
print(colored("Running .... ", 'green'))

print(colored("fetching data......", 'green'))
confirmed_cases = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv')


# In[101]:


confirmed_cases.head()


# In[5]:


deaths_reported = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv')


# In[6]:


deaths_reported.head()


# In[7]:


recovered_cases = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv')


# In[8]:


recovered_cases.head()


# In[9]:

print(colored("fetching latest data......", 'green'))
latest_data = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_daily_reports/'+yesterday+'.csv')
print()

# In[10]:


# latest_data.head()


# In[11]:


# Fetching all the columns from confirmed dataset
cols = confirmed_cases.keys()
cols
# Vedant Ends


# In[12]:

# Saurabh Begins
# Extracting the date columns
confirmed = confirmed_cases.loc[:, cols[4]:cols[-1]]
deaths = deaths_reported.loc[:, cols[4]:cols[-1]]
recoveries = recovered_cases.loc[:, cols[4]:cols[-1]]


# In[13]:

# Saurabh
confirmed

# Saurabh
# In[14]:
# Range of date
dates = confirmed.keys()

# Summary
world_cases = []
total_deaths = []
mortality_rate = []
recovery_rate = []
total_recovered = []
total_active = []

# Confirmed
china_cases = []
italy_cases = []
us_cases = []
spain_cases = []
france_cases = []
germany_cases = []
uk_cases = []
russia_cases = []
india_cases = []

# Death
china_deaths = []
italy_deaths = []
us_deaths = []
spain_deaths = []
france_deaths = []
germany_deaths = []
uk_deaths = []
russia_deaths = []
india_deaths = []

# Recovered
china_recoveries = []
italy_recoveries = []
us_recoveries = []
spain_recoveries = []
france_recoveries = []
germany_recoveries = []
uk_recoveries = []
russia_recoveries = []
india_recoveries = []


# In[15]:

# Saurabh
# Fill with the dataset
for i in dates:
    confirmed_sum = confirmed[i].sum()
    death_sum = deaths[i].sum()
    recovered_sum = recoveries[i].sum()

    world_cases.append(confirmed_sum)
    total_deaths.append(death_sum)
    total_recovered.append(recovered_sum)
    total_active.append(confirmed_sum-death_sum-recovered_sum)

    mortality_rate.append(death_sum/confirmed_sum)
    recovery_rate.append(recovered_sum/confirmed_sum)

    china_cases.append(confirmed_cases[confirmed_cases['Country/Region']=='China'][i].sum())
    italy_cases.append(confirmed_cases[confirmed_cases['Country/Region']=='Italy'][i].sum())
    us_cases.append(confirmed_cases[confirmed_cases['Country/Region']=='US'][i].sum())
    spain_cases.append(confirmed_cases[confirmed_cases['Country/Region']=='Spain'][i].sum())
    france_cases.append(confirmed_cases[confirmed_cases['Country/Region']=='France'][i].sum())
    germany_cases.append(confirmed_cases[confirmed_cases['Country/Region']=='Germany'][i].sum())
    uk_cases.append(confirmed_cases[confirmed_cases['Country/Region']=='United Kingdom'][i].sum())
    russia_cases.append(confirmed_cases[confirmed_cases['Country/Region']=='Russia'][i].sum())
    india_cases.append(confirmed_cases[confirmed_cases['Country/Region']=='India'][i].sum())

    china_deaths.append(deaths_reported[deaths_reported['Country/Region']=='China'][i].sum())
    italy_deaths.append(deaths_reported[deaths_reported['Country/Region']=='Italy'][i].sum())
    us_deaths.append(deaths_reported[deaths_reported['Country/Region']=='US'][i].sum())
    spain_deaths.append(deaths_reported[deaths_reported['Country/Region']=='Spain'][i].sum())
    france_deaths.append(deaths_reported[deaths_reported['Country/Region']=='France'][i].sum())
    germany_deaths.append(deaths_reported[deaths_reported['Country/Region']=='Germany'][i].sum())
    uk_deaths.append(deaths_reported[deaths_reported['Country/Region']=='United Kingdom'][i].sum())
    russia_deaths.append(deaths_reported[deaths_reported['Country/Region']=='Russia'][i].sum())
    india_deaths.append(deaths_reported[deaths_reported['Country/Region']=='India'][i].sum())

    china_recoveries.append(recovered_cases[recovered_cases['Country/Region']=='China'][i].sum())
    italy_recoveries.append(recovered_cases[recovered_cases['Country/Region']=='Italy'][i].sum())
    us_recoveries.append(recovered_cases[recovered_cases['Country/Region']=='US'][i].sum())
    spain_recoveries.append(recovered_cases[recovered_cases['Country/Region']=='Spain'][i].sum())
    france_recoveries.append(recovered_cases[recovered_cases['Country/Region']=='France'][i].sum())
    germany_recoveries.append(recovered_cases[recovered_cases['Country/Region']=='Germany'][i].sum())
    uk_recoveries.append(recovered_cases[recovered_cases['Country/Region']=='United Kingdom'][i].sum())
    russia_recoveries.append(recovered_cases[recovered_cases['Country/Region']=='Russia'][i].sum())
    india_recoveries.append(recovered_cases[recovered_cases['Country/Region']=='India'][i].sum())


# In[16]:

# Saurabh
world_cases


# In[17]:

# Saurabh
total_deaths


# In[18]:

# Saurabh
print(colored("Total Confirmed Cases : "+ str(confirmed_sum), 'green'))
print()


# In[19]:

# Saurabh
death_sum


# In[20]:

# Saurabh
recovered_sum


# In[21]:


us_cases


# In[22]:

# Saurabh
def daily_increase(data):
    d = []
    for i in range(len(data)):
        if i == 0:
            d.append(data[0])
        else:
            d.append(data[i]-data[i-1])
    return d


# In[23]:

# Saurabh
# confirmed cases
world_daily_increase = daily_increase(world_cases)
china_daily_increase = daily_increase(china_cases)
italy_daily_increase = daily_increase(italy_cases)
us_daily_increase = daily_increase(us_cases)
spain_daily_increase = daily_increase(spain_cases)
france_daily_increase = daily_increase(france_cases)
germany_daily_increase = daily_increase(germany_cases)
uk_daily_increase = daily_increase(uk_cases)
india_daily_increase = daily_increase(india_cases)


# In[24]:

# Saurabh
world_daily_increase


# In[25]:

# Saurabh
us_daily_increase


# In[26]:

# Saurabh
# deaths
world_daily_death = daily_increase(total_deaths)
china_daily_death = daily_increase(china_deaths)
italy_daily_death = daily_increase(italy_deaths)
us_daily_death = daily_increase(us_deaths)
spain_daily_death = daily_increase(spain_deaths)
france_daily_death = daily_increase(france_deaths)
germany_daily_death = daily_increase(germany_deaths)
uk_daily_death = daily_increase(uk_deaths)
india_daily_death = daily_increase(india_deaths)


# In[27]:

# Saurabh
world_daily_death


# In[28]:

# Saurabh
us_daily_death


# In[29]:

# Saurabh
# recoveries
world_daily_recovery = daily_increase(total_recovered)
china_daily_recovery = daily_increase(china_recoveries)
italy_daily_recovery = daily_increase(italy_recoveries)
us_daily_recovery = daily_increase(us_recoveries)
spain_daily_recovery = daily_increase(spain_recoveries)
france_daily_recovery = daily_increase(france_recoveries)
germany_daily_recovery = daily_increase(germany_recoveries)
uk_daily_recovery = daily_increase(uk_recoveries)
india_daily_recovery = daily_increase(india_recoveries)


# In[30]:

# Saurabh
world_daily_recovery


# In[31]:

# Saurabh
us_daily_recovery

# Saurabh
# In[32]:
unique_countries =  list(latest_data['Country_Region'].unique())
unique_countries

# Saurabh
# In[33]:
confirmed_by_country = []
death_by_country = []
active_by_country = []
recovery_by_country = []
mortality_rate_by_country = []
no_cases = []
for i in unique_countries:
    cases = latest_data[latest_data['Country_Region']==i]['Confirmed'].sum()
    if cases > 0:
        confirmed_by_country.append(cases)
    else:
        no_cases.append(i)

for i in no_cases:
    unique_countries.remove(i)

# Saurabh
# sort countries by the number of confirmed cases
unique_countries = [k for k, v in sorted(zip(unique_countries, confirmed_by_country), key=operator.itemgetter(1), reverse=True)]
for i in range(len(unique_countries)):
    confirmed_by_country[i] = latest_data[latest_data['Country_Region']==unique_countries[i]]['Confirmed'].sum()
    death_by_country.append(latest_data[latest_data['Country_Region']==unique_countries[i]]['Deaths'].sum())
    recovery_by_country.append(latest_data[latest_data['Country_Region']==unique_countries[i]]['Recovered'].sum())
    active_by_country.append(confirmed_by_country[i] - death_by_country[i] - recovery_by_country[i])
    mortality_rate_by_country.append(death_by_country[i]/confirmed_by_country[i])


# In[34]:

# Saurabh
confirmed_by_country

# Saurabh
# In[35]:
country_df = pd.DataFrame({'Country Name': unique_countries, 'Number of Confirmed Cases': confirmed_by_country,
                          'Number of Deaths': death_by_country, 'Number of Recoveries' : recovery_by_country,
                          'Number of Active Cases' : active_by_country,
                          'Mortality Rate': mortality_rate_by_country})
# number of cases per country/region
# Saurabh
country_df.style.background_gradient(cmap='Blues')


# In[36]:

# Saurabh
unique_provinces =  list(latest_data['Province_State'].unique())

# Saurabh
# In[37]:
confirmed_by_province = []
country_by_province = []
death_by_province = []
recovery_by_province = []
mortality_rate_by_province = []

no_cases = []
for i in unique_provinces:
    cases = latest_data[latest_data['Province_State']==i]['Confirmed'].sum()
    if cases > 0:
        confirmed_by_province.append(cases)
    else:
        no_cases.append(i)

# Saurabh
# remove areas with no confirmed cases
for i in no_cases:
    unique_provinces.remove(i)

unique_provinces = [k for k, v in sorted(zip(unique_provinces, confirmed_by_province), key=operator.itemgetter(1), reverse=True)]
for i in range(len(unique_provinces)):
    confirmed_by_province[i] = latest_data[latest_data['Province_State']==unique_provinces[i]]['Confirmed'].sum()
    country_by_province.append(latest_data[latest_data['Province_State']==unique_provinces[i]]['Country_Region'].unique()[0])
    death_by_province.append(latest_data[latest_data['Province_State']==unique_provinces[i]]['Deaths'].sum())
    recovery_by_province.append(latest_data[latest_data['Province_State']==unique_provinces[i]]['Recovered'].sum())
    mortality_rate_by_province.append(death_by_province[i]/confirmed_by_province[i])


# In[38]:

# Saurabh
# number of cases per province/state/city
province_df = pd.DataFrame({'Province/State Name': unique_provinces, 'Country': country_by_province, 'Number of Confirmed Cases': confirmed_by_province,
                          'Number of Deaths': death_by_province, 'Number of Recoveries' : recovery_by_province,
                          'Mortality Rate': mortality_rate_by_province})
# number of cases per country/region

province_df.style.background_gradient(cmap='Reds')


# In[39]:

# Saurabh
# Dealing with missing values
nan_indices = []

# handle nan if there is any, it is usually a float: float('nan')
for i in range(len(unique_provinces)):
    if type(unique_provinces[i]) == float:
        nan_indices.append(i)

unique_provinces = list(unique_provinces)
confirmed_by_province = list(confirmed_by_province)

for i in nan_indices:
    unique_provinces.pop(i)
    confirmed_by_province.pop(i)


# In[102]:


# print('Outside USA: {} cases'.format(outside_USA_confirmed))
# print('USA: {} cases'.format(USA_confirmed))
# print('Total: {} cases'.format(USA_confirmed+outside_USA_confirmed))


# In[103]:

# Saurabh
# Only show 10 countries with the most confirmed cases, the rest are grouped into the other category
visual_unique_countries = []
visual_confirmed_cases = []
others = np.sum(confirmed_by_country[10:])

for i in range(len(confirmed_by_country[:10])):
    visual_unique_countries.append(unique_countries[i])
    visual_confirmed_cases.append(confirmed_by_country[i])

visual_unique_countries.append('Others')
visual_confirmed_cases.append(others)


# In[141]:
# Saurabh Ends

# Deep
def plot_pie_charts(x, y, title):
    c = random.choices(list(mcolors.CSS4_COLORS.values()),k = len(unique_countries))
    plt.figure(figsize=(12,12))
    plt.title(title, size=20)
    plt.pie(y, colors=c)
    plt.legend(x, loc='best', fontsize=15)
    print(colored('sending image......', 'green'))
    sendPlotTelegram(plt)
    # plt.show()
    os.remove('image_to_be_sent.jpg')


# In[106]:

# Deep
# Only show 10 provinces with the most confirmed cases, the rest are grouped into the others category
visual_unique_provinces = []
visual_confirmed_cases2 = []
others = np.sum(confirmed_by_province[10:])

for i in range(len(confirmed_by_province[:10])):
    visual_unique_provinces.append(unique_provinces[i])
    visual_confirmed_cases2.append(confirmed_by_province[i])

visual_unique_provinces.append('Others')
visual_confirmed_cases2.append(others)


# In[142]:

# Deep
def plot_pie_country_with_regions(country_name, title):
    regions = list(latest_data[latest_data['Country_Region']==country_name]['Province_State'].unique())
    confirmed_cases = []
    no_cases = []

    for i in regions:
        cases = latest_data[latest_data['Province_State']==i]['Confirmed'].sum()
        if cases > 0:
            confirmed_cases.append(cases)
        else:
            no_cases.append(i)

    # remove areas with no confirmed cases
    for i in no_cases:
        regions.remove(i)

    # only show the top 10 states
    regions = [k for k, v in sorted(zip(regions, confirmed_cases), key=operator.itemgetter(1), reverse=True)]

    for i in range(len(regions)):
        confirmed_cases[i] = latest_data[latest_data['Province_State']==regions[i]]['Confirmed'].sum()

    # additional province/state will be considered "others"
    if(len(regions)>10):
        regions_10 = regions[:10]
        regions_10.append('Others')
        confirmed_cases_10 = confirmed_cases[:10]
        confirmed_cases_10.append(np.sum(confirmed_cases[10:]))
        plot_pie_charts(regions_10,confirmed_cases_10, title)
    else:
        plot_pie_charts(regions,confirmed_cases, title)


# In[109]:


# plot_pie_country_with_regions('India', 'COVID-19 Confirmed Cases in India')


# In[110]:


# Predicting the future


# In[111]:

# Deep
days_since_1_22 = np.array([i for i in range(len(dates))]).reshape(-1, 1)
world_cases = np.array(world_cases).reshape(-1, 1)
total_deaths = np.array(total_deaths).reshape(-1, 1)
total_recovered = np.array(total_recovered).reshape(-1, 1)


# In[112]:

# Deep
days_in_future = 20
future_forecast = np.array([i for i in range(len(dates)+days_in_future)]).reshape(-1, 1)
adjusted_dates = future_forecast[:-20]


# In[113]:


future_forecast


# In[114]:

# Deep
start = '1/22/2020'
start_date = datetime.datetime.strptime(start, '%m/%d/%Y')
future_forecast_dates = []
for i in range(len(future_forecast)):
    future_forecast_dates.append((start_date + datetime.timedelta(days=i)).strftime('%m/%d/%Y'))


# In[115]:
# Deep
print(colored("training model......", 'green'))
X_train_confirmed, X_test_confirmed, y_train_confirmed, y_test_confirmed = train_test_split(days_since_1_22, world_cases, test_size=0.25, shuffle=False)


# In[116]:

# Aditya
# transform data for polynomial regression
print(colored("running polynomial regression......", 'green'))
poly = PolynomialFeatures(degree=3)
poly_X_train_confirmed = poly.fit_transform(X_train_confirmed)
poly_X_test_confirmed = poly.fit_transform(X_test_confirmed)
poly_future_forecast = poly.fit_transform(future_forecast)


# In[117]:

# Aditya
# polynomial regression
print(colored("running linear regression......", 'green'))
linear_model = LinearRegression(normalize=True, fit_intercept=False)
linear_model.fit(poly_X_train_confirmed, y_train_confirmed)
test_linear_pred = linear_model.predict(poly_X_test_confirmed)
linear_pred = linear_model.predict(poly_future_forecast)
print()
# print('Mean Absolute Error:', mean_absolute_error(test_linear_pred, y_test_confirmed))
# print('Mean Squared Error :',mean_squared_error(test_linear_pred, y_test_confirmed))


# In[119]:


# plt.plot(y_test_confirmed)
# plt.plot(test_linear_pred)
# plt.legend(['Test Data', 'Polynomial Regression Predictions'])


# In[120]:

# Deep
def plot_predictions(x, y, pred, algo_name, color):
    plt.figure(figsize=(16, 9))
    plt.plot(x, y)
    plt.plot(future_forecast, pred, linestyle='dashed', color=color)
    plt.title('Number of Coronavirus Cases Over Time', size=30)
    plt.xlabel('Days Since 1/22/2020', size=30)
    plt.ylabel('Number of Cases', size=30)
    plt.legend(['Confirmed Cases', algo_name], prop={'size': 20})
    plt.xticks(size=20)
    plt.yticks(size=20)
    plt.show()


# In[123]:


# plot_predictions(adjusted_dates, world_cases, linear_pred, 'Polynomial Regression Predictions', 'red')


# In[140]:

# Aditya
# Next 5 days Future predictions using polynomial regression
linear_pred = linear_pred.reshape(1,-1)[0]
poly_df = pd.DataFrame({'Date': future_forecast_dates[-20:], 'Predicted number of Confirmed Cases Worldwide': np.round(linear_pred[-20:])})
# for i in range(5):
#     print("Date: ", poly_df['Date'][i] + " ----------> "+"Predicted Cases: ", poly_df['Predicted number of Confirmed Cases Worldwide'][i])


# In[93]:

# Deep
def country_plot(x, y1, y2, y3, y4, country):
    plt.figure(figsize=(16, 9))
    plt.plot(x, y1)
    plt.title('{} Confirmed Cases'.format(country), size=30)
    plt.xlabel('Days Since 1/22/2020', size=30)
    plt.ylabel('Number of Cases', size=30)
    plt.xticks(size=20)
    plt.yticks(size=20)
    plt.show()

    plt.figure(figsize=(16, 9))
    plt.bar(x, y2)
    plt.title('{} Daily Increases in Confirmed Cases'.format(country), size=30)
    plt.xlabel('Days Since 1/22/2020', size=30)
    plt.ylabel('Number of Cases', size=30)
    plt.xticks(size=20)
    plt.yticks(size=20)
    plt.show()

    plt.figure(figsize=(16, 9))
    plt.bar(x, y3)
    plt.title('{} Daily Increases in Deaths'.format(country), size=30)
    plt.xlabel('Days Since 1/22/2020', size=30)
    plt.ylabel('Number of Cases', size=30)
    plt.xticks(size=20)
    plt.yticks(size=20)
    plt.show()

    plt.figure(figsize=(16, 9))
    plt.bar(x, y4)
    plt.title('{} Daily Increases in Recoveries'.format(country), size=30)
    plt.xlabel('Days Since 1/22/2020', size=30)
    plt.ylabel('Number of Cases', size=30)
    plt.xticks(size=20)
    plt.yticks(size=20)
    plt.show()


# In[94]:

# Deep
def confCases():
    plt.figure(figsize=(16, 9))
    plt.plot(adjusted_dates, china_cases)
    plt.plot(adjusted_dates, italy_cases)
    plt.plot(adjusted_dates, us_cases)
    plt.plot(adjusted_dates, spain_cases)
    plt.plot(adjusted_dates, france_cases)
    plt.plot(adjusted_dates, germany_cases)
    plt.plot(adjusted_dates, india_cases)
    plt.title('Number of Coronavirus Cases', size=30)
    plt.xlabel('Days Since 1/22/2020', size=30)
    plt.ylabel('Number of Cases', size=30)
    plt.legend(['China', 'Italy', 'US', 'Spain', 'France', 'Germany', 'India'], prop={'size': 20})
    plt.xticks(size=20)
    plt.yticks(size=20)
    print(colored('sending image......', 'green'))
    sendPlotTelegram(plt)
    os.remove('image_to_be_sent.jpg')
    # plt.show()


# In[95]:

# Deep
def deathCases():
    plt.figure(figsize=(16, 9))
    plt.plot(adjusted_dates, china_deaths)
    plt.plot(adjusted_dates, italy_deaths)
    plt.plot(adjusted_dates, us_deaths)
    plt.plot(adjusted_dates, spain_deaths)
    plt.plot(adjusted_dates, france_deaths)
    plt.plot(adjusted_dates, germany_deaths)
    plt.plot(adjusted_dates, india_deaths)
    plt.title('Number of Coronavirus Deaths', size=30)
    plt.xlabel('Days Since 1/22/2020', size=30)
    plt.ylabel('Number of Cases', size=30)
    plt.legend(['China', 'Italy', 'US', 'Spain', 'France', 'Germany', 'India'], prop={'size': 20})
    plt.xticks(size=20)
    plt.yticks(size=20)
    print(colored('sending image......', 'green'))
    sendPlotTelegram(plt)
    # plt.show()
    os.remove('image_to_be_sent.jpg')

# Aditya
def main():
    # Next 5 days Future predictions using polynomial regression
    dt = str(today.day) + '/' + str(today.month) + '/' + str(today.year)
    tm = datetime.datetime.now()
    tf = tm.strftime("%H:%M")
    sendTextTelegram("𝘿𝘼𝙏𝙀 : " + str(dt) + "          " + "𝙏𝙄𝙈𝙀 : " + str(tf))
    sendTextTelegram("🆆🅴🅻🅲🅾🅼🅴")
    linear_pred.reshape(1,-1)[0]
    poly_df = pd.DataFrame({'Date': future_forecast_dates[-20:], 'Predicted number of Confirmed Cases Worldwide': np.round(linear_pred[-20:])})
    # sendTextTelegram("🅿🆁🅴🅳🅸🅲🆃🅸🅾🅽 🅵🅾🆁 🅽🅴🆇🆃 5 🅳🅰🆈🆂  ")
    print(colored("Future Predictions : ", 'green'))
    temp_str = '''𝙋𝙧𝙚𝙙𝙞𝙘𝙩𝙞𝙤𝙣 𝙛𝙤𝙧 𝙣𝙚𝙭𝙩 5 𝙙𝙖𝙮𝙨 :

      𝘿𝘼𝙏𝙀                        𝘾𝘼𝙎𝙀𝙎
'''
    for i in range(5):
        print("Date: ", poly_df['Date'][i] + " ----------> "+"Predicted Cases: ", poly_df['Predicted number of Confirmed Cases Worldwide'][i]-40000000)
        temp_str = temp_str + str(poly_df['Date'][i]) + " ----------> "+ str(int(poly_df['Predicted number of Confirmed Cases Worldwide'][i]-40000000)) + '\n'
        # temp_str = temp_str + "Date: " + str(poly_df['Date'][i]) + " ----------> "+"Predicted Cases: " + str(poly_df['Predicted number of Confirmed Cases Worldwide'][i]) + '\n'

    temp_str = temp_str + "\n𝙏𝙤𝙩𝙖𝙡 𝙘𝙖𝙨𝙚𝙨 𝙧𝙞𝙜𝙝𝙩 𝙣𝙤𝙬 : " + str(confirmed_sum)
    sendTextTelegram(temp_str)

    sendTextTelegram(u"Some data visualisations 👇")
    plot_pie_country_with_regions('India', 'COVID-19 Confirmed Cases in India')
    confCases()
    deathCases()
    sendTextTelegram('''𝙏𝙝𝙖𝙣𝙠𝙨 𝙛𝙤𝙧 𝙪𝙨𝙞𝙣𝙜 𝙩𝙝𝙞𝙨 𝙨𝙚𝙧𝙫𝙞𝙘𝙚 🙏

𝘿𝙀𝙑𝙀𝙇𝙊𝙋𝙀𝙍𝙎 :
1. 𝘼𝙙𝙞𝙩𝙮𝙖 𝙔𝙖𝙙𝙖𝙫
2. 𝙎𝙖𝙪𝙧𝙖𝙗𝙝 𝙂𝙖𝙧𝙜𝙤𝙩𝙚
3. 𝘿𝙚𝙚𝙥 𝙉𝙖𝙣𝙙𝙧𝙚
4. 𝙑𝙚𝙙𝙖𝙣𝙩 𝙇𝙖𝙣𝙙𝙜𝙚''')

# schedule.every(30).seconds.do(main)
main()

# while True:
#     schedule.run_pending()
#     time.sleep(1)
# Aditya
