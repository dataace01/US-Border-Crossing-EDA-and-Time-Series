# Capstone-Project-US-Crossing-Border

Finding Data, Cleaning, EDA, NLP, Visualization, Time Series,Prediction model.

Capstone-Project-US-Crossing-Border-
Overview:

The Border Crossing data file contains whole data from January 1996 to December 2021 of the total incoming crossing counts into the US. It contains 7 columns specifying the port and its unique code, the border, the mode of vehicle used, number of people crossing the border into the US, the date and time of crossing, the state in which they entered. This dataset contains all the incidents of crossing the border into the US as provided by the Bureau of Transportation Statistics, Govt. of the US https://resources.data.gov/resources/dcat-us/ .

Content: 

In the project are used the following lybraries & packages:-

numpy
pandas 
matplotlib 
sklearn 
seaborn 
datetime
wordcloud

Problem Statement:

I am interested to use this dataset to learn about the incoming value into the US. It can also prove to useful to predict the daily, weekly, monthly or annual traffic that’s going to accumulate on the borders so that the authorities can be aware of the number beforehand.

This is an interesting dataset, great for exercising pandas skills and analysis. I will be trying some Machine Learning , but it is not so many options to do modeling with categorical data.  

Definitions:

The Border Crossing Data contains the following 7 columns:-

- Port Name:Name of the unique port where the border was crossed.
- State:- State where the border was crossed.
- Port Code:- Unique port code. Border:
- Border(US-Canada or US-Mexico)
- Date:- Date of crossing the border.
- Measure:- Mode of transportation and/or pedestrians. 
- Value:- Count of people/passengers crossed the US border.

The unique number of each column: 
- There are 121 port names. 
- There are 120 port codes. 
- State 15 
- Border 2 
- Measure 12 
- Value 10482135222 
- Month 12 
- Years 25

Measure by: 
- Busses
- Bus Passengers 
- Pedestrians 
- Trucks 
- Truck Containers Loaded
- Truck Containers Bobtail 
- Personal Vehicles 
- Personal Vehicle Passengers 
- Trains 
- Train Passengers 
- Rail Containers Loaded 
- Rail Containers Empty

Port Names :
['Alcan', 'Alexandria Bay', 'Ambrose', 'Andrade', 'Antler', 'Baudette', 'Beecher Falls', 'Blaine', 'Boundary', 'Bridgewater', 'Brownsville', 'Buffalo Niagara Falls', 'Calais', 'Calexico', 'Calexico East', 'Carbury', 'Champlain Rouses Point', 'Columbus', 'Cross Border Xpress', 'Dalton Cache', 'Danville', 'Del Bonita', 'Del Rio', 'Derby Line', 'Detroit', 'Douglas', 'Dunseith', 'Eagle Pass', 'Eastport', 'El Paso', 'Ferry', 'Fort Fairfield', 'Fort Kent', 'Fortuna', 'Frontier', 'Grand Portage', 'Hannah', 'Hansboro', 'Hidalgo', 'Highgate Springs', 'Houlton', 'International Falls', 'Jackman', 'Kenneth G Ward', 'Lancaster', 'Laredo', 'Laurier', 'Lukeville', 'Madawaska', 'Maida', 'Massena', 'Metaline Falls', 'Morgan', 'Naco', 'Neche', 'Nighthawk', 'Nogales', 'Noonan', 'Northgate', 'Norton', 'Ogdensburg', 'Opheim', 'Oroville', 'Otay Mesa', 'Pembina', 'Piegan', 'Pinecreek', 'Point Roberts', 'Portal', 'Port Angeles', 'Porthill', 'Port Huron', 'Presidio', 'Progreso', 'Raymond', 'Richford', 'Rio Grande City', 'Roma', 'Roosville', 'Roseau', 'San Luis', 'Santa Teresa', 'San Ysidro', 'Sarles', 'Sasabe', 'Sault Sainte Marie', 'Scobey', 'Sherwood', 'Skagway', 'St John', 'Sumas', 'Sweetgrass', 'Tecate', 'Tornillo', 'Trout River', 'Turner', 'Van Buren', 'Vanceboro', 'Walhalla', 'Warroad', 'Westhope', 'Whitlash', 'Wildhorse', 'Willow Creek', 'Ysleta', 'Algonac', 'Highgate Springs-Alburg', 'Houston', 'Kenneth G Ward Poe', 'Cape Vincent', 'Boquillas', 'Everett', 'Limestone', 'Anacortes', 'Friday Harbor', 'Ketchikan', 'Toledo', 'Portland', 'Whitetail', 'Bar Harbor', 'Noyes']

Border: 
- US-Border
- CA-Border

Border States:

AK- Arkansas 
ND- North Dakota 
ME- Maine 
CA- California 
WA- Washington 
MT- Montana 
NY- New York 
OH- Ohio 
ID- Idaho 
NM- New Mexico 
MN- Minessota 
VT- Vermont 
MI- Michigan 
AZ- Arizona 
TX- Texas

It's important to aknowledge that this dataset doesn't count the number of unique vehicles, passengers or pedestrians but rather the number of crossings. For example, the same truck can go back and forth across the border at least twice a day and data for each time will be collected. Also this data doesn't include the name, nationality,race,ethnicity,religion,disability, intersex, medical, or psychological conditionssex,sexuality,origin,anchestry,gender,age and/or status of the passengers or pedestrians or their reason for crossing the USA border.

KEY TAKEAWAYS:

.A time series is a data set that tracks a sample over time. In particular, a time series allows one to see what factors influence certain variables from period to period. .Time series analysis can be useful to see how a given asset, security or economic variable changes over time. .Forecasting methods: using time series are used in both fundamental and technical analysis. Although cross-sectional data is seen as the opposite of time series, the two are often used together in practice. 

.An ARIMA model is a class of statistical models for analyzing and forecasting time series data. ... ARIMA is an acronym that stands for AutoRegressive Integrated Moving Average. It is a generalization of the simpler AutoRegressive Moving Average and adds the notion of integration. . A seasonal autoregressive integrated moving average (SARIMA) model is one step different from an ARIMA model based on the concept of seasonal trends. In many time series data, frequent seasonal effects come into play. Take for example the average temperature measured in a location with four seasons.

.NLP based on Machine Learning can be used to establish communication channels between humans and machines. Using Word Cloud is a data visualization technique used for representing text data in which the size of each word indicates its frequency or importance. Significant textual data points can be highlighted using a word cloud. El Paso is most crossed border by people(Value), East Port is the most frequently crossed Port.

Summary:

On analyzing the Border parameter, about 76.1% of the inbound crossing in the dataset consisted of US-Mexico border meaning people from Mexico have tended to come in more frequently into the US as compared to the people from Canada beginning January,1996 to December,2021 , State Texas is having the maximum share throughout the Country.
About 5,794,493,889 people have crossed the US-Mexico and US-Canada border by Personal Vehicles as Passengers which shows people avoid public transport.
The Border Crossing is a critical situation to deal as the plot suggests the growth varying between 3 billion to 5.4 billion in range between 1996 to 2021. There is some improvement in 2019 with the data falling just short of 1 billion. The sudden fall in 2020 is solely owing to the COVID-19 outbreak. In a prediction of 2022-2023 there is an increase of the inbound number of people from both Mexico and Canada, it is definitely delivering a good impact to the economy and growing population in general after a very big distraction from the Covid-19 shutdown. I am sure that there are several techniques that can be also used to make even deeper analysis , explore States and Measures more into detailed observation using Time Series. It would be interesting also to add a few more datas straight related to Border Crossing , for example Fuel, Weather, Covid-19 number of cases,etc. All this would give a greater picture of the impact on all fields of immigration strategy. I'm still a beginner and want to improve myself in every way I can. In this Project I have used a good portion of EDA , some NLP,Forecasting With SARIMAX ,ARIMA, and looking forward to improving using all these techniques and skills.

