import pandas
import numpy

# print(pandas.read_csv(filepath_or_buffer="transactions.csv", sep=','))
tr_mcc_codes = pandas.read_csv(filepath_or_buffer="tr_mcc_codes.csv", sep=';')
# print(tr_mcc_codes)
tr_types = pandas.read_csv(filepath_or_buffer="tr_types.csv", sep=';')
# print(tr_types)
transactions = pandas.read_csv(filepath_or_buffer="transactions.csv", sep=',', nrows=1000000)
# print(transactions)
gender_train = pandas.read_csv(filepath_or_buffer="gender_train.csv", sep=',')
# print(gender_train)

transactions_tr_type = transactions['tr_type'].sample(n=1000)
transactions_description = pandas.merge(tr_types, transactions_tr_type, on='tr_type')
print("==========1===========")
print(transactions_description['tr_description'].str.contains(pat='POS|ATM').mean())

print("==========2===========")
transactions_tr_type_unique = transactions.groupby(['tr_type']).size().nlargest(10)

print(transactions_tr_type_unique)

print("==========3==========")
max_amount_client = transactions.where(transactions.amount > 0).groupby(['customer_id']).sum().nlargest(1, 'amount')
min_amount_client = transactions.where(transactions.amount < 0).groupby(['customer_id']).sum().nsmallest(1, 'amount')
max_min_diff = max_amount_client['amount'].values[0] - abs(min_amount_client['amount'].values[0])

print(max_amount_client)
print(min_amount_client)
print("========dif==========")
print(max_min_diff)

print("===========4==========")
average_task2 = transactions_tr_type_unique.mean()
median_task2 = transactions_tr_type_unique.median(axis=0)
average_task3_max = max_amount_client['tr_type'].mean()
average_task3_min = min_amount_client['tr_type'].mean()
median_task3_max = max_amount_client['tr_type'].median()
median_task3_min = min_amount_client['tr_type'].median()

print(average_task2)
print(median_task2)
print(average_task3_max)
print(average_task3_min)
print(median_task3_max)
print(median_task3_min)

print("============5===========")
transactions = pandas.merge(transactions, gender_train, how='left')
transactions = pandas.merge(transactions, tr_mcc_codes, how='inner')
transactions = pandas.merge(transactions, tr_types, how='inner')

average_spending_women = transactions['amount'].where((transactions.amount < 0) & (transactions.gender == 1)).mean()
average_spending_men = transactions['amount'].where((transactions.amount < 0) & (transactions.gender == 0)).mean()
average_spending_diff = abs(average_spending_women) - abs(average_spending_men)

average_income_women = transactions['amount'].where((transactions.amount > 0) & (transactions.gender == 1)).mean()
average_income_men = transactions['amount'].where((transactions.amount > 0) & (transactions.gender == 0)).mean()
average_income_diff = abs(average_income_women-average_income_men)

print(average_spending_women)
print(average_spending_men)
print(average_spending_diff)
print(average_income_women)
print(average_income_men)
print(average_income_diff)

print("===========6============")
max_income_tr_type_women = transactions.where((transactions.amount > 0) & (transactions.gender == 1)).groupby(['tr_type'])['amount'].max()
max_income_tr_type_men = transactions.where((transactions.amount > 0) & (transactions.gender == 0)).groupby(['tr_type'])['amount'].max()

anti_top_10_women = max_income_tr_type_women.nsmallest(10)
anti_top_10_men = max_income_tr_type_men.nsmallest(10)

common_anti_top_10 = pandas.merge(anti_top_10_men, anti_top_10_women, on='tr_type')

print(max_income_tr_type_women)
print("=====================")
print(max_income_tr_type_men)
print("=======================")
print(anti_top_10_women)
print("=======================")
print(anti_top_10_men)
print("=======================")
print(common_anti_top_10)

print("===========7===========")
spending_sum_tr_type_women = transactions.where((transactions.amount < 0) & (transactions.gender == 1)).groupby(['tr_type'])['amount'].sum()
spending_sum_tr_type_men = transactions.where((transactions.amount < 0) & (transactions.gender == 0)).groupby(['tr_type'])['amount'].sum()

spending_sum_diff = abs(abs(spending_sum_tr_type_women) - abs(spending_sum_tr_type_men)).nlargest(5)
print("====women====")
print(spending_sum_tr_type_women)
print("====men=====")
print(spending_sum_tr_type_men)
print("====diff=====")
print(spending_sum_diff)
print("===========8===========")

tr_hour = transactions['tr_datetime'].str[-8:-6]
night_income = transactions[pandas.to_numeric(tr_hour) <= 5].where(transactions.amount > 0)

print(night_income.groupby(['gender'])['amount'].sum())






