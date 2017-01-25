import csv
from teams import *
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn import metrics, linear_model, ensemble, tree
import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")


def remove_outliers(df, field, outlier_ids):
	"""

	:param df: dataframe of tickets sold
	:param field: which field to look for the outliers
	:param outlier_ids: The outlier id in that field
	:return: Same dataframe with outliers removed
	"""
	for outlier in outlier_ids:
		df = df.loc[df[field] != outlier, ]

	return df


def standardize_teams(df):
	old2new = {"Florida Marlins": "Miami Marlins",
			   "Anaheim Angels": "Los Angeles Angels",
			   "Montreal Expos" : "Washington Nationals"}

	for old, new in old2new.iteritems():
		df = df.replace(to_replace=old, value=new)

	return df


def promotion_cat(prom):
	"""

	:param prom: Full name of promotion
	:return: promotion category as specified below
	"""

	# Promotions which to categorize. Dictionary of 'search term':'Category' pairs.
	promotions = {}
	promotions['fire'] = 'Fireworks'
	promotions['bobbl'] = 'Bobble'
	promotions['jers'] = 'Jersey'
	promotions['opening'] = 'OpeningDay'
	promotions['exhib'] = 'Exhibition'


	if isinstance(prom, float):
		return None
	else:
		prom = prom.lower()

	for search_term, category in promotions.items():
		if search_term in prom:
			return category

	# Categorize every other promotion as Other
	return "Other"


def group_cat(group):
	"""

	:param group: Full name of Group Night
	:return: promotion category as specified below
	"""

	# Promotions which to categorize. Dictionary of 'search term':'Category' pairs.
	groups = {}
	groups['little'] = 'LittleLeagueDay'
	groups['scout'] = 'Scout'


	if isinstance(group, float):
		return None
	else:
		group = group.lower()

	for search_term, category in groups.items():
		if search_term in group:
			return category

	# Categorize every other Group Night as Other
	return None


def day_or_night(date):
	# Day games is anything before 3PM, night games is after 3PM
	game_cat = "Day" if date.hour < 15 else "Night"

	return game_cat


def add_holiday(df):
	"""

	:param df: Dataframe with 'Date' as a column
	:return: same dataframe with an added column named 'Holiday' which is a boolean,
	 		 true if the date in 'Date' column is a holiday
	"""
	cal = calendar()
	holidays = cal.holidays(start=df['Date'].min(), end=df['Date'].max())

	df['Holiday'] = df['Date'].dt.date.astype('datetime64').isin(holidays)


def add_month(date):
	"""
	Add the month to be used as a categorical variable.
	Note for simplicity: March = April, and October = September

	:param date: Month as number (January = 1)
	:return: string month
	"""
	m = date.month

	if m == 3:
		return "April"
	elif m == 10:
		return "September"
	else:
		return date.strftime('%B')


def avg_road_attendance(teams, season, team, years):
	"""
	Ex.
	input: (teams, 2014, "New York Mets", 4)
	output: Mets average road attendance from 2010 to 2013

	:param teams: Teams object with road_att attribute
	:param season: years before which season
	:param team: MLB team
	:param years: # of seasons to average
	:return: Average road attendance over that time.
	"""

	total = 0.0
	for yr in range(1, years + 1):
		total += teams.get_road_att(season - yr, team)

	avg = total / years

	return avg


def add_columns(df, teams):
	"""

	:param df: Games DataFrame
	:return: Games DataFrame with added columns
	"""

	# Date manipulation:
	# 1. Convert string to datetime objects
	# 2. Add boolean column for holiday
	# 3. Add Month as string
	try:
		df['Date'] = pd.to_datetime(df['event_time'], format='%m/%d/%y %I:%M %p')
	except (ValueError):
		df['Date'] = pd.to_datetime(df['event_time'], format='%m/%d/%y %H:%M')
	add_holiday(df)
	df['Month'] = df['Date'].apply(add_month)

	# Day and Time:
	# GameTime is day or night
	# DayTime is day_time combined (e.g. Monday_Night)
	df['GameTime'] = df['Date'].apply(day_or_night)
	df['DayTime'] = df.apply(lambda x: ''.join([x['DayOfWeek'], '_', x['GameTime']]), axis=1)

	# Promotion Category as defined in 'promotion_cat' function
	df['PromotionCat'] = df['Promotion'].apply(promotion_cat)

	df['Group'] = df['GroupNight'].apply(group_cat)

	# AL West or not
	df['ALWest'] = df.apply(lambda x: teams.is_al_west(x['Season'], x['mlb_opponent']), axis=1)

	# Opponent and A's last season wins
	df['LastSeasonOppWins'] = df.apply(lambda x: teams.get_wins(x['Season'] - 1, x['mlb_opponent']), axis=1)
	df['LastSeasonHomeWins'] = df.apply(lambda x: teams.get_wins(x['Season'] - 1, 'OAK'), axis=1)

	# Opponents average road attendance, last season
	df['AvgRoadAttendance_1'] = df.apply(lambda x: avg_road_attendance(teams, x['Season'], x['mlb_opponent'], 1), axis=1)
	# df['AvgRoadAttendance_5'] = df.apply(lambda x: avg_road_attendance(teams, x['Season'], x['mlb_opponent'], 5), axis=1)


	return df


def make_train_features(df, categorical_features, numerical_features):
	"""

	:param df: pandas DataFrame
	:return: feature array X, target array Y, list of headers
	"""

	# Categorical features to turn into dummy variables
	cat_features = df[categorical_features]

	# Pandas function to one-hot encode each category above.
	x = pd.get_dummies(cat_features)

	dummy_columns = x.columns

	# Add numerical categories
	for feat in numerical_features:
		x[feat] = df[feat]

	# Separate headers and numerical data
	headers = list(x.columns.values)
	x = np.asarray(x, dtype=np.float)

	# Numerical target (tickets sold)
	y = np.asarray(df["tickets_sold"])

	return x, y, dummy_columns, headers


def make_test_features(df, dummy_columns, categorical_features, numerical_features):
	"""

	:param df: pandas DataFrame
	:return: feature array X, target array Y, list of headers
	"""

	# Categorical features to turn into dummy variables
	cat_features = df[categorical_features]

	# Pandas function to one-hot encode each category above.
	x = pd.get_dummies(cat_features)
	x = x.reindex(columns=dummy_columns, fill_value=0)

	# Add numerical categories
	for feat in numerical_features:
		x[feat] = df[feat]

	# Separate headers and numerical data
	headers = list(x.columns.values)
	x = np.asarray(x, dtype=np.float)

	# Numerical target (tickets sold)
	y = None

	return x, y, headers


def cv_scores(model, train_x, train_y, plot=False):
	"""

	:param model: An algorithm such as LinearRegression
	:param train_x: training data
	:param train_y: training target
	:return: Root Mean Squared Error (RMSE)
	"""
	scores = cross_val_score(model, train_x, train_y, cv=10, scoring='neg_mean_squared_error')
	mse = -np.mean(scores)
	root_mse = np.sqrt(mse)

	if plot:
		predicted = cross_val_predict(model, train_x, train_y, cv=10)

		fig, ax = plt.subplots()
		ax.scatter(train_y, predicted)
		ax.plot([train_y.min(), train_y.max()], [train_y.min(), train_y.max()], 'k--', lw=4)
		ax.set_xlabel('Measured')
		ax.set_ylabel('Predicted')
		plt.show()


	return root_mse


def save_pred(filename, algo, train_x, train_y, test_x):
	algo.fit(train_x, train_y)
	pred_y = algo.predict(test_x)

	test_df['Prediction'] = pred_y

	test_df.to_csv(filename)


if __name__ == '__main__':
	# Training df: tickets sold 2009-2016
	train_df = pd.read_csv('data/schedule_history.csv')

	# Remove outliers for rainouts
	outliers = [202806, 202825]
	train_df = remove_outliers(train_df, 'EVENT_ID', outliers)

	# Remove outliers for exhibitions
	exhibition_ids = [1479, 3485, 2990, 20100001, 464, 20090001]
	train_df = remove_outliers(train_df, 'EVENT_ID', exhibition_ids)

	train_df = standardize_teams(train_df)

	# Test df: 2017 season
	test_df = pd.read_csv('data/schedule_2017.csv')

	# Teams object to be used in feature engineering
	teams = Teams()

	# Add columns and make features
	train_df = add_columns(train_df, teams)
	test_df = add_columns(test_df, teams)


	# train_df.to_csv('data/orig_train.csv', index=False)
	# test_df.to_csv('data/orig_test.csv', index=False)


	categorical = ['DayTime',
				   'Month',
				   'mlb_opponent',
				   'PromotionCat',
				   'Group']

	numerical = ['Holiday',
				 'LastSeasonOppWins',
				 'LastSeasonHomeWins',
				 'AvgRoadAttendance_1']

	train_x, train_y, dummy_columns, train_headers = make_train_features(train_df, categorical, numerical)
	test_x, test_y, test_headers = make_test_features(test_df, dummy_columns, categorical, numerical)

	# Model Evaluation
	algo = linear_model.LinearRegression()

	train_df = pd.DataFrame(data=train_x, columns=train_headers)
	train_df['tickets_sold'] = train_y

	algo.fit(train_x, train_y)
	preds = algo.predict(test_x)
	test_df['Prediction'] = preds

	test_df.to_csv('data/schedule_2017_with_pred.csv', index=False)

	# print cv_scores(algo, train_x, train_y, plot=False)

	save_pred('data/2017_pred.csv', algo, train_x, train_y, test_x)


	# Some Trees
	# rows = []
	#
	# for trees in range(25, 501, 25):
	# 	for d in range(1, 101):
	#
	# 		algo = ensemble.RandomForestRegressor(n_estimators=trees, max_depth=d)
	#
	# 		rmse = cv_scores(algo, train_x, train_y, plot=False)
	# 		r2 = np.mean(cross_val_score(algo, train_x, train_y, cv=10, scoring='r2'))
	#
	# 		rows.append(('RandomForest', trees, d, r2, rmse))
	#
	# 		print "Trees: %d, Max Depth: %d, RMSE: %d, R-Squared: %0.4f" % (trees, d, rmse, r2)
	#
	# with open('model_performance.csv', 'wb') as f:
	# 	csv_out = csv.writer(f)
	# 	csv_out.writerow(['Algorithm', 'Trees', 'Max_Depth', 'R2', 'RMSE'])
	# 	for row in rows:
	# 		csv_out.writerow(row)

	# Save model coefficients
	# with open('Model_Coefficients.csv', 'wb'):
	# 	coefs = algo.coef_
	#
	# 	for i in range(len(coefs)):
	# 		print "%s: %0.2f" % (train_headers[i], coefs[i])

	# fig, ax = plt.subplots()
	# ax.scatter(test_y, pred_y)
	# ax.plot([test_y.min(), test_y.max()], [test_y.min(), test_y.max()], 'k--', lw=4)
	# ax.set_xlabel('Measured')
	# ax.set_ylabel('Predicted')
	# plt.show()

	# print "Root Mean Squared Error: %d" % rmse
	# print "R-Squared: %0.4f" % r2

	# train_idx = games['Season'] != 2016
	# test_idx = games['Season'] == 2016
	#
	# train_x, train_y = x[np.where(train_idx, True, False),], y[np.where(train_idx, True, False)]
	# test_x, test_y = x[np.where(test_idx, True, False),], y[np.where(test_idx, True, False)]

	# print algo.coef_


	# dotfile = StringIO()
	# export_graphviz(classifier, out_file=dotfile,
	# 				feature_names=["Temperature", "Humidity", "Light", "CO2", "HumidityRatio"])
	# graph = pydotplus.graph_from_dot_data(dotfile.getvalue())
	# graph.write_pdf("classifier_tree.pdf")


"""
Original Headers of the dataframe for reference:

['Season', 'EVENT_ID', 'event_time', 'Event_Label', 'DayOfWeek', 'mlb_opponent', 'Promotion', 'Premium', 'PartialPlan', 'tickets_sold']
"""
