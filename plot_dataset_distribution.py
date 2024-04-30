import pandas as pd
import matplotlib.pyplot as plt
import json

#read the dataset configuration
dataset_name ='2024-02-22_15-58-58'
dataset_path = f'datasets_k_fold/{dataset_name}'
dataset_config_path = f'{dataset_path}/dataset_config.json'
fold_number = 1
train_data_path = f'{dataset_path}/fold_{fold_number}/train.csv'
validation_data_path = f'{dataset_path}/fold_{fold_number}/validation.csv'
test_data_path = f'{dataset_path}/All_test.csv'
with open(dataset_config_path, 'r') as f:
    config = json.load(f)
#get the label names
label_names = config['label_names']
df_train = pd.read_csv(train_data_path)
df_validation = pd.read_csv(validation_data_path)
df_test = pd.read_csv(test_data_path)

#sum the labels
train_label_sum = df_train[label_names].sum()
validation_label_sum = df_validation[label_names].sum()
test_label_sum = df_test[label_names].sum()

#plot the label distribution
fig, ax = plt.subplots()
train_label_sum.plot(kind='bar', ax=ax, color='b', position=1, width=0.25)
validation_label_sum.plot(kind='bar', ax=ax, color='r', position=2, width=0.25)
test_label_sum.plot(kind='bar', ax=ax, color='g', position=0, width=0.25)
ax.set_ylabel('Count')
ax.set_title('Label distribution')
ax.legend(['Train', 'Validation', 'Test'])
plt.show()

print(test_label_sum)