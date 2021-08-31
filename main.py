import numpy as np
import torch
import discrete_BCQ
from utils import *


# Trains BCQ offline Batch
def train_BCQ_batch(train_set_path, state_dim, num_actions, parameters, device, epoch_num=100, batch_size=1024):
    print("epoch_num: ", epoch_num)
    print("batch_size: ", batch_size)

    # Initialize and load policy
    policy = discrete_BCQ.discrete_BCQ(
        num_actions,
        state_dim,
        device,
        0.3,
        parameters["discount"],
        parameters["optimizer"],
        parameters["optimizer_parameters"],
        parameters["polyak_target_update"],
        parameters["target_update_freq"],
        parameters["tau"],
        parameters["initial_eps"],
        parameters["end_eps"],
        parameters["eps_decay_period"],
        parameters["eval_eps"]
	)
 
    # Load training dataset
    user_click_history_processed, user_click_history_avg_processed, user_protrait_processed, exposed_items_processed, labels_processed, exposed_items_id = get_trainset_data(train_set_path)

    num_train_set = len(user_click_history_processed)
    
    print("Training  ...")
    batch_num = num_train_set // batch_size
    q_loss_list = []
    reward_list = []
    for epoch in range(epoch_num):
        idx = np.random.permutation(num_train_set)
        q_loss_total = 0.0
        reward_total = 0.0
        for i in range(batch_num):
            batch_idx = idx[i*batch_size:(i+1)*batch_size].tolist()
            user_click_history_processed_batch = []
            user_click_history_avg_processed_batch = []
            user_protrait_processed_batch = []
            exposed_item_batch = []
            labels_batch = []
            exposed_items_id_batch = []
            for i_idx in batch_idx:
                user_click_history_processed_batch.append(user_click_history_processed[i_idx])
                user_click_history_avg_processed_batch.append(user_click_history_avg_processed[i_idx])
                user_protrait_processed_batch.append(user_protrait_processed[i_idx])
                exposed_item_batch.append(exposed_items_processed[i_idx])
                labels_batch.append(labels_processed[i_idx])
                exposed_items_id_batch.append(exposed_items_id[i_idx])

            num_step = 9
            for step in range(num_step):
                if step == 0:
                    exposed_item_feature_last_batch = []
                    for bth in range(batch_size):
                        exposed_item_feature_last_batch.append([0, 0, 0, 0, 0, 0, 0])
                else:
                    exposed_item_feature_last_batch = []
                    for bth in range(batch_size):
                        exposed_item_feature_last_batch.append(exposed_item_batch[bth][step - 1])
                exposed_item_feature_cur_batch = []
                for bth in range(batch_size):
                    exposed_item_feature_cur_batch.append(exposed_item_batch[bth][step])

                # state : user_click_history + user_protrait + last product features
                state = concat_feature_batch(user_click_history_processed_batch, user_click_history_avg_processed_batch, user_protrait_processed_batch, exposed_item_feature_last_batch) #list 266

                # next_state : user_click_history + user_protrait + current product features
                next_state = concat_feature_batch(user_click_history_processed_batch, user_click_history_avg_processed_batch, user_protrait_processed_batch, exposed_item_feature_cur_batch)#list 266

                # action
                reward = []
                action = []
                for bth in range(batch_size):
                    if labels_batch[bth][step] == 1.0:
                        reward.append(labels_batch[bth][step] * exposed_item_feature_cur_batch[bth][-2])
                    else:
                        reward.append((-0.25) * exposed_item_feature_cur_batch[bth][-2])
                    action.append(exposed_items_id_batch[bth][step])

                done = batch_size*[1.0] if (step == num_step-1) else batch_size*[0.0]
                q_loss = policy.train_batch(state, action, next_state, reward, done)
                q_loss_total += q_loss.item()
                reward_total += sum(reward)
        
        q_loss_list.append(q_loss_total)
        reward_list.append(reward_total)

        # evaluations
        if epoch > 0 and epoch % 30 == 0:
            _ = eval_policy(policy)

    print("Predicting & write_csv  ...")
    action_result_list = eval_policy(policy)
    write_csv(action_result_list)


def eval_policy(policy):
    print("Evaluation Policy   ...")
    test_set_path = "./data/track2_testset.csv"

    test_user_click_history_processed, test_user_click_history_avg_processed, test_user_protrait_processed, item_info_list = get_track2_test_data(test_set_path)

    num_track2_test_set = len(test_user_click_history_processed)


    action_result_list = []
    for test_iters in range(num_track2_test_set):
        test_user_click_history_processed_row = test_user_click_history_processed[test_iters]
        test_user_click_history_avg_processed_row = test_user_click_history_avg_processed[test_iters]
        test_user_protrait_processed_row = test_user_protrait_processed[test_iters]

        num_row = 9
        action_list = []
        reward_row = 0.0
        
        for i in range(num_row):
            if i == 0:
                action_item_feature = [0, 0, 0, 0, 0, 0, 0]
            else:
                action_item_feature = get_action_info(action_list[i-1], item_info_list)
            
            state = test_user_click_history_processed_row + test_user_click_history_avg_processed_row + test_user_protrait_processed_row + action_item_feature

            action = 1 + policy.select_action(state, i, action_list)
            
            action_list.append(action)

            reward = action_item_feature[-2]
            reward_row += reward
        
        action_result_list.append(action_list)

    return action_result_list
    

if __name__ == "__main__":
    parameters = {
		# Exploration
		"start_timesteps": 1e3,
		"initial_eps": 0.1,
		"end_eps": 0.1,
		"eps_decay_period": 1,
		# Evaluation
		"eval_freq": 1000,
		"eval_eps": 0,
		# Learning
		"discount": 0.99,
		"batch_size": 64,
		"optimizer": "Adam",
		"optimizer_parameters": {
			"lr": 3e-4
		},
		"train_freq": 1,
		"polyak_target_update": True,
		"target_update_freq": 1,
		"tau": 0.005
	}

    torch.manual_seed(2021)
    np.random.seed(2021)
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

    train_set_path = './data/trainset.csv'
    state_dim = 273  
    num_actions = 381

    train_BCQ_batch(train_set_path, state_dim, num_actions, parameters, device, 100, 1024)
