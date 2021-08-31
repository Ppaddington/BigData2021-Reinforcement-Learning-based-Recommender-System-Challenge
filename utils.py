import io
from sklearn.preprocessing import StandardScaler
import numpy as np

def load_dataset(path):
    i = 0
    user_id, user_click_history, user_protrait, exposed_items, labels, time = [], [], [], [], [], []
    with io.open(path,'r') as file:
        for line in file:
            if i > 0:
                user_id_1, user_click_history_1, user_protrait_1, exposed_items_1, labels_1, time_1 = line.split(' ')
                user_id.append(user_id_1)
                user_click_history.append(user_click_history_1)
                user_protrait.append(user_protrait_1)
                exposed_items.append(exposed_items_1)
                labels.append(labels_1)
                time.append(time_1)
            i = i + 1
    return user_id, user_click_history, user_protrait, exposed_items, labels, time


def data_processing(user_click_history, user_protrait, exposed_items, labels, item_info_list):
    user_click_history_processed = []
    for item in user_click_history:
        user_click_history_row = []
        item_split_list = item.split(',')
        for item_2 in item_split_list:
            click_history = float(item_2.split(':')[0])
            user_click_history_row.append(click_history)

        if len(user_click_history_row) < 249:
            for i in range(249-len(user_click_history_row)):
                user_click_history_row.append(0.0)
        
        if len(user_click_history_row) > 249:
            print("len(user_click_history_row): ", len(user_click_history_row))
            user_click_history_row = user_click_history_row[:249]

        user_click_history_processed.append(user_click_history_row)

    user_click_history_avg_processed = []
    for item in user_click_history:
        user_click_history_row_avg = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        item_split_list = item.split(',')
        for item_2 in item_split_list:
            click_history_item_id = float(item_2.split(':')[0])
            
            if click_history_item_id == 0.0:
                continue

            item_info_dic = item_info_list[int(click_history_item_id)-1]
            item_info = item_info_dic[float(click_history_item_id)]
            user_click_history_row_avg = user_click_history_row_avg + np.array(item_info)
            
        user_click_history_row_avg = user_click_history_row_avg / len(item_split_list)
        user_click_history_row_avg = user_click_history_row_avg.tolist()

        user_click_history_avg_processed.append(user_click_history_row_avg)

    user_protrait_processed = []
    for item in user_protrait:
        user_protrait_row = []
        item_split_list = item.split(',')
        for item_2 in item_split_list:
            user_protrait_row.append(float(item_2))
        user_protrait_processed.append(user_protrait_row)

    exposed_items_id = []
    for item in exposed_items:
        exposed_items_id_row = []
        item_split_list = item.split(',')
        for item_id in item_split_list:
            exposed_items_id_row.append(float(item_id))
        exposed_items_id.append(exposed_items_id_row)

    exposed_items_processed = []
    for item in exposed_items:
        exposed_items_row = []
        item_split_list = item.split(',')
        for item_id in item_split_list:
            item_info_dic = item_info_list[int(item_id)-1]
            item_info = item_info_dic[float(item_id)]

            exposed_items_row.append(item_info)
        exposed_items_processed.append(exposed_items_row)
    
    labels_processed = []
    for item in labels:
        labels_row = []
        item_split_list = item.split(',')
        for item_2 in item_split_list:
            labels_row.append(float(item_2))
        labels_processed.append(labels_row)
        
    return user_click_history_processed, user_click_history_avg_processed, user_protrait_processed, exposed_items_processed, labels_processed, exposed_items_id


def load_item_info():
    price_max = 150.0
    price_min = 16621.0
    item_info_list = []
    item_id, item_vec, price, location = [], [], [], []
    i = 0
    with io.open('./data/item_info.csv','r') as file:
        for line in file:
            if i > 0:
                item_info = {}
                item_vec_row = []

                item_id_1, item_vec_1, price_1, location_1 = line.split(' ')
                item_id_1 = float(item_id_1)
                price_1 = float(price_1)
                price_1 = (price_1 - price_min) / (price_max - price_min)
                location_1 = float(location_1)

                item_vec_list = item_vec_1.split(',')
                for item_2 in item_vec_list:
                    item_vec_row.append(float(item_2))

                item_id.append(item_id_1)
                item_vec.append(item_vec_row)
                price.append(price_1)
                location.append(location_1)

                item = []
                for j in range(len(item_vec_row)):
                    item.append(item_vec_row[j])
                item.append(price_1)
                item.append(location_1)
                item_info[item_id_1] = item
                
                item_info_list.insert(int(item_id_1)-1, item_info)

            i = i + 1
    return item_info_list


def concat_feature_batch(user_click_history_processed_batch, user_click_history_avg_processed_batch, user_protrait_processed_batch, exposed_item_feature_processed_batch):
    feature_batch = []
    for i in range(len(user_click_history_processed_batch)):
        feature_row = user_click_history_processed_batch[i] + user_click_history_avg_processed_batch[i] + user_protrait_processed_batch[i] + exposed_item_feature_processed_batch[i]
        feature_batch.append(feature_row)
    return feature_batch


def get_trainset_data(train_set_path):
    item_info_list = load_item_info()

    user_id, user_click_history, user_protrait, exposed_items, labels, time = load_dataset(train_set_path)

    user_click_history_processed, user_click_history_avg_processed, user_protrait_processed, exposed_items_processed, labels_processed, exposed_items_id = data_processing(user_click_history, user_protrait, exposed_items, labels, item_info_list)

    scaler = StandardScaler()
    user_click_history_processed = scaler.fit_transform(user_click_history_processed).tolist()
    user_protrait_processed = scaler.fit_transform(user_protrait_processed).tolist()

    return user_click_history_processed, user_click_history_avg_processed, user_protrait_processed, exposed_items_processed, labels_processed, exposed_items_id

def load_track2_test_dataset(path):
    i = 0
    user_id, user_click_history, user_protrait = [], [], []
    with io.open(path,'r') as file:
        for line in file:
            if i > 0:
                user_id_1, user_click_history_1, user_protrait_1 = line.split(' ')
                user_id.append(user_id_1)
                user_click_history.append(user_click_history_1)
                user_protrait.append(user_protrait_1)
            i = i + 1
    return user_id, user_click_history, user_protrait

def data_track2_test_processing(user_click_history, user_protrait, item_info_list):
    user_click_history_processed = []
    for item in user_click_history:
        user_click_history_row = []
        item_split_list = item.split(',')
        for item_2 in item_split_list:
            click_history = float(item_2.split(':')[0])
            user_click_history_row.append(click_history)

        if len(user_click_history_row) < 249:
            for i in range(249-len(user_click_history_row)):
                user_click_history_row.append(0.0)
        
        if len(user_click_history_row) > 249:
            user_click_history_row = user_click_history_row[:249]

        user_click_history_processed.append(user_click_history_row)

    user_click_history_avg_processed = []
    for item in user_click_history:
        user_click_history_row_avg = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        item_split_list = item.split(',')
        for item_2 in item_split_list:
            click_history_item_id = float(item_2.split(':')[0])
            
            if click_history_item_id == 0.0:
                continue

            item_info_dic = item_info_list[int(click_history_item_id)-1]
            item_info = item_info_dic[float(click_history_item_id)]
            user_click_history_row_avg = user_click_history_row_avg + np.array(item_info)
            
        user_click_history_row_avg = user_click_history_row_avg / len(item_split_list)
        user_click_history_row_avg = user_click_history_row_avg.tolist()

        user_click_history_avg_processed.append(user_click_history_row_avg)

    user_protrait_processed = []
    for item in user_protrait:
        user_protrait_row = []
        item_split_list = item.split(',')
        for item_2 in item_split_list:
            user_protrait_row.append(float(item_2))
        user_protrait_processed.append(user_protrait_row)

    return user_click_history_processed, user_click_history_avg_processed, user_protrait_processed


def get_track2_test_data(test_set_path):
    item_info_list = load_item_info()

    user_id, user_click_history, user_protrait = load_track2_test_dataset(test_set_path)

    user_click_history_processed, user_click_history_avg_processed, user_protrait_processed = data_track2_test_processing(user_click_history, user_protrait, item_info_list)

    scaler = StandardScaler()
    user_click_history_processed = scaler.fit_transform(user_click_history_processed).tolist()
    user_protrait_processed = scaler.fit_transform(user_protrait_processed).tolist()

    return user_click_history_processed, user_click_history_avg_processed, user_protrait_processed, item_info_list


def get_action_info(action, item_info_list):
    item_info_dic = item_info_list[int(action)-1]
    item_info = item_info_dic[float(action)]
    return item_info

def write_csv(action_result_list):
    import pandas as pd
    import csv

    test2_set = pd.read_csv('/root/rl_recsys/data/track2_testset.csv',' ')
    item_id_list = test2_set['user_id'].tolist()
    res_list = []

    for row_list in action_result_list:
        row_list = list(map(str, row_list))
        row_str = ' '.join(row_list)
        res_list.append(row_str)

    path = "/root/rl_recsys/data/submission.csv"
    with open(path, 'w', newline='', encoding='utf8') as f:
        csv_write = csv.writer(f)
        id = item_id_list
        id = [ str(i) for i in id]

        pred = res_list

        head = ('id', 'category')
        csv_write.writerow(head)
        for pair in zip(id, pred):
            csv_write.writerow(pair)