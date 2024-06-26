import numpy as np
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from DatasetLoad import DatasetLoad
from DatasetLoad import AddGaussianNoise
from torch import optim
import random
import copy
import time
from sys import getsizeof
# https://cryptobook.nakov.com/digital-signatures/rsa-sign-verify-examples
from Crypto.PublicKey import RSA
from hashlib import sha256
from Models import Mnist_2NN, Mnist_CNN
from Blockchain import Blockchain
import os

class Device:
    def __init__(self, idx, assigned_train_ds, assigned_test_dl, local_batch_size, learning_rate, loss_func, opti, network_stability, net, dev, miner_acception_wait_time, worker_acception_wait_time, miner_accepted_transactions_size_limit, validate_threshold, pow_difficulty, even_link_speed_strength, base_data_transmission_speed, even_computation_power, is_malicious, noise_variance, check_signature, not_resync_chain, malicious_updates_discount, knock_out_rounds, lazy_worker_knock_out_rounds):
        self.idx = idx
        # deep learning variables
        self.train_ds = assigned_train_ds
        self.test_dl = assigned_test_dl
        self.local_batch_size = local_batch_size
        self.loss_func = loss_func
        self.network_stability = network_stability
        self.net = copy.deepcopy(net)
        if opti == "SGD":
            self.opti = optim.SGD(self.net.parameters(), lr=learning_rate)
        self.dev = dev
        # in real system, new data can come in, so train_dl should get reassigned before training when that happens
        self.train_dl = DataLoader(self.train_ds, batch_size=self.local_batch_size, shuffle=True)
        self.local_train_parameters = None
        self.initial_net_parameters = None
        self.global_parameters = None
        self.candidate_parameters = None

        # blockchain variables
        self.role = None
        self.pow_difficulty = pow_difficulty
        if even_link_speed_strength:
            self.link_speed = base_data_transmission_speed
        else:
            self.link_speed = random.random() * base_data_transmission_speed
        self.devices_dict = None
        self.aio = False #all in one
        ''' simulating hardware equipment strength, such as good processors and RAM capacity. Following recorded times will be shrunk by this value of times
        # for workers, its update time
        # for miners, its PoW time
        # for validators, its validation time
        # might be able to simulate molopoly on computation power when there are block size limit, as faster devices' transactions will be accepted and verified first
        '''
        if even_computation_power:
            self.computation_power = 1
        else:
            self.computation_power = random.randint(0, 4)
        self.peer_list = set()
        # used in cross_verification and in the PoS
        self.online = True
        self.rewards = 0
        self.blockchain = Blockchain()
        # init key pair
        self.modulus = None
        self.private_key = None
        self.public_key = None
        self.generate_rsa_key()
        # black_list stores device index rather than the object
        self.black_list = set()
        self.knock_out_rounds = knock_out_rounds
        self.lazy_worker_knock_out_rounds = lazy_worker_knock_out_rounds
        self.worker_accuracy_accross_records = {}
        self.has_added_block = False
        self.the_added_block = None
        self.is_malicious = is_malicious
        self.noise_variance = noise_variance
        self.check_signature = check_signature
        self.not_resync_chain = not_resync_chain
        self.malicious_updates_discount = malicious_updates_discount
        # used to identify slow or lazy workers
        self.active_worker_record_by_round = {}
        self.untrustworthy_workers_record_by_comm_round = {}
        self.untrustworthy_validators_record_by_comm_round = {}
        # for picking PoS legitimate blockd;bs
        # self.stake_tracker = {} # used some tricks in main.py for ease of programming
        # used to determine the slowest device round end time to compare PoW with PoS round end time. If simulate under computation_power = 0, this may end up equaling infinity
        self.round_end_time = 0
        ''' For workers '''
        self.local_updates_dict = {}
        self.local_updates_rewards_per_transaction = 0
        self.received_block_from_miner = None
        self.accuracy_this_round = float('-inf')
        self.local_accuracy = 0
        self.worker_associated_miner_set = set()
        self.local_update_time = None
        self.local_total_epoch = 0
        self.worker_acception_wait_time = worker_acception_wait_time
        self.unordered_arrival_time_accepted_miner_candidate = {}
        self.final_candidate_queue_to_validate = {}
        self.worker_accepted_broadcasted_miner_candidate = None or []
        self.post_validation_candidate_queue = None or []
        self.unordered_downloaded_block_processing_queue = {}
        self.block_download_time = None

        ''' For validators '''   
        self.accuracies_this_round = {}  
        self.validate_threshold = validate_threshold  
        ''' For miners '''
        self.validation_rewards_this_round = 0
        self.miner_local_accuracy = None
        self.post_validation_transactions_queue = None or []
        self.miner_associated_worker_set = set()
        self.miner_accepted_broadcasted_worker_transactions = None or []
        self.final_transactions_queue_to_validate = {}
        self.unordered_arrival_time_accepted_worker_transactions = {}
        self.aggregate_time = None
        self.aggregate_rewards = 0
        self.aggregate_local_updates_info = []
        self.candidate_model_dict = {}
        self.final_candidate_transactions_queue_to_mine = []
        self.accepted_miner_broadcasted_worker_validated_candidate_transactions = None or []
        self.unordered_arrival_time_accepted_worker_validated_candidate_transactions = {}
        self.miner_acception_wait_time = miner_acception_wait_time
        self.miner_accepted_transactions_size_limit = miner_accepted_transactions_size_limit
        self.mined_rewards = 0
        self.unconfirmed_candidate_block = None
        self.unconfirmed_candidate_block_arrival_time = None
        #self.unordered_propagated_block_processing_queue = {} # pure simulation queue and does not exist in real distributed system
        self.mined_block = None

        # dict cannot be added to set()
        self.unconfirmmed_transactions = None or []
        self.broadcasted_transactions = None or []
        self.received_propagated_block = None
        self.received_propagated_validator_block = None
        self.block_generation_time_point = None

        ''' For malicious node '''
        self.variance_of_noises = None or []

    def worker_reset_vars_for_new_round(self):
        self.local_update_time = None
        self.local_updates_dict.clear()
        self.local_updates_rewards_per_transaction = 0
        self.local_total_epoch = 0
        self.local_accuracy = 0
        self.variance_of_noises.clear()
        self.worker_associated_miner_set.clear()  
        self.unordered_arrival_time_accepted_miner_candidate.clear()
        self.final_candidate_queue_to_validate.clear()
        self.worker_accepted_broadcasted_miner_candidate.clear()
        self.post_validation_candidate_queue.clear()
        self.unordered_downloaded_block_processing_queue.clear()
        self.block_download_time = None

        self.received_block_from_miner = None
        self.accuracy_this_round = float('-inf')
        self.has_added_block = False
        self.the_added_block = None   
        self.round_end_time = 0

 

    def miner_reset_vars_for_new_round(self):
        self.miner_associated_worker_set.clear()
        self.unconfirmmed_transactions.clear()
        self.broadcasted_transactions.clear()
        self.unordered_arrival_time_accepted_worker_transactions.clear()
        self.post_validation_transactions_queue.clear()
        self.miner_accepted_broadcasted_worker_transactions.clear()
        self.candidate_model_dict.clear()
        self.aggregate_local_updates_info.clear()
        self.aggregate_rewards = 0
        self.aggregate_time = None
        self.unordered_arrival_time_accepted_worker_validated_candidate_transactions.clear()
        self.final_candidate_transactions_queue_to_mine.clear()
        self.accepted_miner_broadcasted_worker_validated_candidate_transactions.clear()
        #self.mined_rewards = 0
        self.unconfirmed_candidate_block = None
        self.unconfirmed_candidate_block_arrival_time = None
        #self.unordered_propagated_block_processing_queue.clear()
        self.mined_block = None

        self.received_propagated_block = None
        self.received_propagated_validator_block = None
        self.has_added_block = False
        self.the_added_block = None
        # self.unordered_arrival_time_accepted_validator_transactions.clear()
        # self.miner_accepted_broadcasted_validator_transactions.clear()
        self.block_generation_time_point = None
		#self.block_to_add = None
        self.round_end_time = 0


    '''Step 0: workers assign associated miners'''
    def associate_with_miner(self):
        miners_in_peer_list = set()
        for peer in self.peer_list:
            if peer.return_role() == "miner":
                if not peer.return_idx() in self.black_list:
                    miners_in_peer_list.add(peer)
        if not miners_in_peer_list:
            return False
        random.seed(7)
        associate_miner_number = random.randint(1, len(miners_in_peer_list)) #TODO:优化连接的miner数量以最小化通信开销
        self.worker_associated_miner_set = random.sample(miners_in_peer_list, associate_miner_number)
        print(f"{self.role} {self.idx} associated with {len(self.worker_associated_miner_set)} miner(s): {[miner.return_idx() for miner in self.worker_associated_miner_set]}")
        return self.worker_associated_miner_set
    
    def add_device_to_association(self, to_add_device):
        if not to_add_device.return_idx() in self.black_list:
            vars(self)[f'{self.role}_associated_{to_add_device.return_role()}_set'].add(to_add_device)
        else:
            print(f"WARNING: {to_add_device.return_idx()} in {self.role} {self.idx}'s black list. Not added by the {self.role}.")

    def add_worker_to_association(self, worker_device):
        if not worker_device.return_idx() in self.black_list:
            self.associated_worker_set.add(worker_device)
        else:
            print(f"WARNING: {worker_device.return_idx()} in validator {self.idx}'s black list. Not added by the validator.")

    '''Step 1 - workers do local updates'''
    #TODO change to computation power
    def worker_local_update(self, rewards, log_files_folder_path_comm_round, comm_round, local_epochs=1):
        print(f"Worker {self.idx} is doing local_update with computation power {self.computation_power} and link speed {round(self.link_speed,3)} bytes/s")
        self.net.load_state_dict(self.global_parameters, strict=True) #Load the global parameters (model weights) into the worker's neural network
        self.local_update_time = time.time()
        # local worker update by specified epochs
        # logging maliciousness
        is_malicious_node = "M" if self.return_is_malicious() else "B" #Malicious node with 'M' or Benign node with 'B'
        self.local_updates_rewards_per_transaction = 0
        for epoch in range(local_epochs):
            for data, label in self.train_dl:
                data, label = data.to(self.dev), label.to(self.dev)
                preds = self.net(data)
                loss = self.loss_func(preds, label)
                loss.backward()
                self.opti.step()
                self.opti.zero_grad()
                self.local_updates_rewards_per_transaction += rewards * (label.shape[0]) #Accumulates rewards based on the number of processed samples.

            # record accuracies to find good -vh #Record accuracies during local updates in a file for analysis.
            with open(f"{log_files_folder_path_comm_round}/worker_{self.idx}_{is_malicious_node}_local_updating_accuracies_comm_{comm_round}.txt", "a") as file:
                file.write(f"{self.return_idx()} epoch_{epoch+1} {self.return_role()} {is_malicious_node}: {self.validate_model_weights(self.net.state_dict())}\n")
            self.local_total_epoch += 1

        # local update done
        try:
            self.local_update_time = (time.time() - self.local_update_time)/self.computation_power
        except:
            self.local_update_time = float('inf')

        if self.is_malicious:
            self.net.apply(self.malicious_worker_add_noise_to_weights)
            print(f"malicious worker {self.idx} has added noise to its local updated weights before transmitting")
            with open(f"{log_files_folder_path_comm_round}/comm_{comm_round}_variance_of_noises.txt", "a") as file:
                file.write(f"{self.return_idx()} {self.return_role()} {is_malicious_node} noise variances: {self.variance_of_noises}\n")

        # record accuracies to find good -vh
        with open(f"{log_files_folder_path_comm_round}/worker_final_local_accuracies_comm_{comm_round}.txt", "a") as file:
            file.write(f"{self.return_idx()} {self.return_role()} {is_malicious_node}: {self.validate_model_weights(self.net.state_dict())}\n")
        print(f"Done {local_epochs} epoch(s) and total {self.local_total_epoch} epochs")
        self.local_train_parameters = self.net.state_dict()
        self.local_accuracy = self.validate_model_weights(self.net.state_dict())

    
    def return_local_updates_and_signature(self, comm_round):
        # local_total_accumulated_epochs_this_round also stands for the lastest_epoch_seq for this transaction(local params are calculated after this amount of local epochs in this round)
        # last_local_iteration(s)_spent_time may be recorded to determine calculating time? But what if nodes do not wish to disclose its computation power
        self.local_updates_dict = {'worker_device_idx': self.idx, 'in_round_number': comm_round, "local_updates_params": copy.deepcopy(self.local_train_parameters), "local_updates_rewards": self.local_updates_rewards_per_transaction, "local_iteration(s)_spent_time": self.local_update_time, "local_total_accumulated_epochs_this_round": self.local_total_epoch, "worker_rsa_pub_key": self.return_rsa_pub_key()}
        self.local_updates_dict["worker_signature"] = self.sign_msg(sorted(self.local_updates_dict.items()))

    def validate_model_weights(self, weights_to_eval=None):
        with torch.no_grad():
            if weights_to_eval:
                self.net.load_state_dict(weights_to_eval, strict=True)
            else:
                self.net.load_state_dict(self.global_parameters, strict=True)
            sum_accu = 0
            num = 0
            for data, label in self.test_dl:
                data, label = data.to(self.dev), label.to(self.dev)
                preds = self.net(data)
                # print(f"preds shape:", preds.shape)
                # print(f"preds type:", preds.dtype)
                preds = torch.argmax(preds, dim=1).long()
                # print(f"preds shape after:", preds.shape)
                # print(f"preds type after:", preds.dtype)
                # print(f"label shape:", label.shape)
                # print(f"label type:", label.dtype)
                sum_accu += (preds == label).float().mean()
                num += 1            
            return sum_accu / num
        
    def malicious_worker_add_noise_to_weights(self, m):
        with torch.no_grad():
            if hasattr(m, 'weight'): #checks if the module m has a weight attribute
                noise = self.noise_variance * torch.randn(m.weight.size())
                variance_of_noise = torch.var(noise)
                m.weight.add_(noise.to(self.dev))
                self.variance_of_noises.append(float(variance_of_noise))
    
    ''' Step 2: miners accept local updates and broadcast to other miners in their respective peer lists.'''
    def return_associated_workers(self):
        return vars(self)[f'{self.role}_associated_worker_set']
 
    def set_unordered_arrival_time_accepted_worker_transactions(self, unordered_transaction_arrival_queue):
        self.unordered_arrival_time_accepted_worker_transactions = unordered_transaction_arrival_queue
    
    def set_transaction_for_final_validating_queue(self, final_transactions_arrival_queue):
        self.final_transactions_queue_to_validate = final_transactions_arrival_queue

    def miner_broadcast_worker_transactions(self):
        for peer in self.peer_list:
            if peer.is_online():
                if peer.return_role() == "miner":
                    if not peer.return_idx() in self.black_list:
                        print(f"miner {self.idx} is broadcasting received miner transactions to miner {peer.return_idx()}.")
                        final_broadcasting_unordered_arrival_time_accepted_worker_transactions_for_dest_miner = copy.copy(self.unordered_arrival_time_accepted_worker_transactions)
                        # if offline, it's like the broadcasted transaction was not received, so skip a transaction
                        for arrival_time, tx in self.unordered_arrival_time_accepted_worker_transactions.items():
                            if not (self.online_switcher() and peer.online_switcher()):
                                del final_broadcasting_unordered_arrival_time_accepted_worker_transactions_for_dest_miner[arrival_time]
                        # in the real distributed system, it should be broadcasting transaction one by one. Here we send the all received transactions(while online) and later calculate the order for the individual broadcasting transaction's arrival time mixed with the transactions itself received
                        peer.accept_miner_broadcasted_worker_transactions(self, final_broadcasting_unordered_arrival_time_accepted_worker_transactions_for_dest_miner)
                        print(f"miner {self.idx} has broadcasted {len(final_broadcasting_unordered_arrival_time_accepted_worker_transactions_for_dest_miner)} worker transactions to miner {peer.return_idx()}.")
                    else:
                        print(f"Destination miner {peer.return_idx()} is in this miner {self.idx}'s black_list. broadcasting skipped for this dest miner.")

    def accept_miner_broadcasted_worker_transactions(self, source_miner, unordered_transaction_arrival_queue_from_source_miner):
        unordered_transaction_arrival_queue_from_source_miner_to_added = {}
        if not source_miner.return_idx() in self.black_list:
            for arrival_time, tx in unordered_transaction_arrival_queue_from_source_miner.items():
                if not self.check_if_has_same_transaction(tx, self.unordered_arrival_time_accepted_worker_transactions, self.miner_accepted_broadcasted_worker_transactions):
                    unordered_transaction_arrival_queue_from_source_miner_to_added[arrival_time] = tx
            if len(unordered_transaction_arrival_queue_from_source_miner_to_added) > 0:
                self.miner_accepted_broadcasted_worker_transactions.append({'source_miner_link_speed': source_miner.return_link_speed(),'broadcasted_transactions': copy.deepcopy(unordered_transaction_arrival_queue_from_source_miner_to_added)})
                print(f"miner {self.idx} has accepted worker transactions from miner {source_miner.return_idx()}")
            else:
                print(f"miner {self.idx} already has the same worker transactions from miner {source_miner.return_idx()}")
        else:
            print(f"Source miner {source_miner.return_idx()} is in miner {self.idx}'s black list. Broadcasted transactions not accepted.")

    def check_if_has_same_transaction(self, transaction_received_from_miner, transactions_received_from_associated_workers, accepted_broadcasted_worker_transactions):
        if_in_transactions_received_from_associated_workers = False
        if_in_accepted_broadcasted_worker_transactions = False
        for   _, transaction in transactions_received_from_associated_workers.items():
            if transaction['worker_device_idx'] == transaction_received_from_miner['worker_device_idx']:
                if_in_transactions_received_from_associated_workers = True
                break
        for accepted_transactions_from_other_worker in accepted_broadcasted_worker_transactions:
            for _, tx in accepted_transactions_from_other_worker['broadcasted_transactions'].items():
                if tx['worker_device_idx'] == transaction_received_from_miner['worker_device_idx']:
                    if_in_accepted_broadcasted_worker_transactions = True
                    break
        return if_in_transactions_received_from_associated_workers or if_in_accepted_broadcasted_worker_transactions
    
    ''' Step 2.5 - with the broadcasted workers transactions, miners decide the final transaction arrival order \n'''
    def return_accepted_broadcasted_worker_transactions(self):
        return self.miner_accepted_broadcasted_worker_transactions
 
    def return_unordered_arrival_time_accepted_worker_transactions(self):
        return self.unordered_arrival_time_accepted_worker_transactions   

    ''' Step 3: miners do self and cross-validation(validate local updates from workers) by the order of transaction arrival time'''
    def return_final_transactions_validating_queue(self):
        return self.final_transactions_queue_to_validate

    def miner_update_model_by_one_epoch_and_validate_local_accuracy(self, opti):
        # return time spent
        print(f"miner {self.idx} is performing one epoch of local update and validation")
        if self.computation_power == 0:
            print(f"miner {self.idx} has computation power 0 and will not be able to complete this validation")
            return float('inf')
        else:
            updated_net = copy.deepcopy(self.net)
            currently_used_lr = 0.01 #learning rate
            for param_group in self.opti.param_groups:
                currently_used_lr = param_group['lr']
            # by default use SGD. Did not implement others
            if opti == 'SGD':
                validation_opti = optim.SGD(updated_net.parameters(), lr=currently_used_lr)
            local_validation_time = time.time()
            for data, label in self.train_dl:
                data, label = data.to(self.dev), label.to(self.dev)
                preds = updated_net(data)
                loss = self.loss_func(preds, label)
                loss.backward()
                validation_opti.step()
                validation_opti.zero_grad()
            # validate by local test set
            with torch.no_grad():
                sum_accu = 0
                num = 0
                for data, label in self.test_dl:
                    data, label = data.to(self.dev), label.to(self.dev)
                    preds = updated_net(data)
                    preds = torch.argmax(preds, dim=1).long()
                    sum_accu += (preds == label).float().mean()
                    num += 1
            self.miner_local_accuracy = sum_accu / num
            print(f"miner {self.idx} locally updated model has accuracy {self.miner_local_accuracy} on its local test set")
            return (time.time() - local_validation_time)/self.computation_power

    def validate_worker_transaction(self, transaction_to_validate, rewards, log_files_folder_path, comm_round, malicious_miner_on):
        log_files_folder_path_comm_round = f"{log_files_folder_path}/comm_{comm_round}"
        if self.computation_power == 0:
            print(f"miner {self.idx} has computation power 0 and will not be able to validate this transaction in time")
            return False, False
        else:
            worker_transaction_device_idx = transaction_to_validate['worker_device_idx']
            if worker_transaction_device_idx in self.black_list:
                print(f"{worker_transaction_device_idx} is in miner's blacklist. Trasaction won't get validated.")
                return False, False
            validation_time = time.time()
            if self.check_signature:
                transaction_before_signed = copy.deepcopy(transaction_to_validate)
                del transaction_before_signed["worker_signature"]
                modulus = transaction_to_validate['worker_rsa_pub_key']["modulus"]
                pub_key = transaction_to_validate['worker_rsa_pub_key']["pub_key"]
                signature = transaction_to_validate["worker_signature"]
                # begin validation
                # 1 - verify signature
                hash = int.from_bytes(sha256(str(sorted(transaction_before_signed.items())).encode('utf-8')).digest(), byteorder='big')
                hashFromSignature = pow(signature, pub_key, modulus)
                if hash == hashFromSignature:
                    print(f"Signature of transaction from worker {worker_transaction_device_idx} is verified by miner {self.idx}!")
                    transaction_to_validate['worker_signature_valid'] = True
                else:
                    print(f"Signature invalid. Transaction from worker {worker_transaction_device_idx} does NOT pass verification.")
                    # will also add sig not verified transaction due to the miner's verification effort and its rewards needs to be recorded in the block
                    transaction_to_validate['worker_signature_valid'] = False
            else:
                print(f"Signature of transaction from worker {worker_transaction_device_idx} is verified by miner {self.idx}!")
                transaction_to_validate['worker_signature_valid'] = True

            # 2 - validate worker's local_updates_params if worker's signature is valid
            if transaction_to_validate['worker_signature_valid']:
                # accuracy validated by worker's update
                accuracy_of_worker_update_using_own_data = self.validate_model_weights(transaction_to_validate["local_updates_params"]) #after "self.validate_model_weights()", self.net() already copy the model_weight to validate
                # if worker's accuracy larger, or lower but the difference falls within the validate threshold value, meaning worker's updated model favors miner's dataset, 
                #so their updates are in the same direction - True, otherwise False. We do not consider the accuracy gap so far, meaning if worker's update is way too good, it is still fine
                print(f'miner updated model accuracy - {self.miner_local_accuracy}') #miner's local model accuracy
                print(f"After applying worker's update, model accuracy becomes - {accuracy_of_worker_update_using_own_data}") #worker's model accuracy on miner test data
                # record their accuracies and difference for choosing a good miner threshold
                is_malicious_miner = "M" if self.is_malicious else "B"
                with open(f"{log_files_folder_path_comm_round}/miner_{self.idx}_{is_malicious_miner}_validation_records_comm_{comm_round}.txt", "a") as file:
                    is_malicious_node = "M" if self.devices_dict[worker_transaction_device_idx].return_is_malicious() else "B"
                    file.write(f"{accuracy_of_worker_update_using_own_data - self.miner_local_accuracy}: miner {self.return_idx()} {is_malicious_miner} in round {comm_round} evluating worker {worker_transaction_device_idx}, diff = v_acc:{self.miner_local_accuracy} - w_acc:{accuracy_of_worker_update_using_own_data} {worker_transaction_device_idx}_maliciousness: {is_malicious_node}\n")
                if accuracy_of_worker_update_using_own_data - self.miner_local_accuracy < self.validate_threshold * -1: #woker's model accuracy is lower than miner's 
                    transaction_to_validate['update_direction'] = False
                    print(f"NOTE: worker {worker_transaction_device_idx}'s updates is deemed as suspiciously malicious by miner {self.idx}") #woker 疑似恶意
                    # is it right?
                    if not self.devices_dict[worker_transaction_device_idx].return_is_malicious():
                        print(f"Warning - {worker_transaction_device_idx} is benign and this validation is wrong.")
                        # for experiments
                        with open(f"{log_files_folder_path}/false_negative_good_nodes_inside_victims.txt", 'a') as file:
                            file.write(f"{self.miner_local_accuracy - accuracy_of_worker_update_using_own_data} = current_miner_accuracy {self.miner_local_accuracy} - accuracy_by_worker_update_using_own_data {accuracy_of_worker_update_using_own_data} , by miner {self.idx} on worker {worker_transaction_device_idx} in round {comm_round}\n")
                    else:
                        with open(f"{log_files_folder_path}/true_negative_malicious_nodes_inside_caught.txt", 'a') as file:
                            file.write(f"{self.miner_local_accuracy - accuracy_of_worker_update_using_own_data} = current_miner_accuracy {self.miner_local_accuracy} - accuracy_by_worker_update_using_own_data {accuracy_of_worker_update_using_own_data} , by miner {self.idx} on worker {worker_transaction_device_idx} in round {comm_round}\n")
                else:
                    transaction_to_validate['update_direction'] = True
                    print(f"worker {worker_transaction_device_idx}'s' updates is deemed as GOOD by miner {self.idx}")
                    # is it right?
                    if self.devices_dict[worker_transaction_device_idx].return_is_malicious():
                        print(f"Warning - {worker_transaction_device_idx} is malicious and this validation is wrong.")
                        # for experiments
                        with open(f"{log_files_folder_path}/false_positive_malious_nodes_inside_slipped.txt", 'a') as file:
                            file.write(f"{self.miner_local_accuracy - accuracy_of_worker_update_using_own_data} = current_miner_accuracy {self.miner_local_accuracy} - accuracy_by_worker_update_using_own_data {accuracy_of_worker_update_using_own_data} , by miner {self.idx} on worker {worker_transaction_device_idx} in round {comm_round}\n")
                    else:
                        with open(f"{log_files_folder_path}/true_positive_good_nodes_inside_correct.txt", 'a') as file:
                            file.write(f"{self.miner_local_accuracy - accuracy_of_worker_update_using_own_data} = current_miner_accuracy {self.miner_local_accuracy} - accuracy_by_worker_update_using_own_data {accuracy_of_worker_update_using_own_data} , by miner {self.idx} on worker {worker_transaction_device_idx} in round {comm_round}\n")
                if self.is_malicious and malicious_miner_on:
                    old_voting = transaction_to_validate['update_direction']
                    transaction_to_validate['update_direction'] = not transaction_to_validate['update_direction']
                    with open(f"{log_files_folder_path_comm_round}/malicious_miner_log.txt", 'a') as file:
                        file.write(f"malicious miner {self.idx} has flipped the voting of worker {worker_transaction_device_idx} from {old_voting} to {transaction_to_validate['update_direction']} in round {comm_round}\n")
                transaction_to_validate['validation_rewards'] = rewards
            else:
                transaction_to_validate['update_direction'] = 'N/A'
                transaction_to_validate['validation_rewards'] = 0
            transaction_to_validate['validation_done_by'] = self.idx
            validation_time = (time.time() - validation_time)/self.computation_power
            transaction_to_validate['validation_time'] = validation_time
            transaction_to_validate['miner_rsa_pub_key'] = self.return_rsa_pub_key()
            # assume signing done in negligible time
            transaction_to_validate["miner_signature"] = self.sign_msg(sorted(transaction_to_validate.items()))
            return validation_time, transaction_to_validate
    
    def add_post_validation_transaction_to_queue(self, transaction_to_add):
        self.post_validation_transactions_queue.append(transaction_to_add)   

    ''' Step 4: miners aggregate their candidate models using the validated local updates from workers.'''    
    def return_post_validation_transactions_queue(self):
        return self.post_validation_transactions_queue
    
    def return_local_params_used_by_miner(self,post_validation_transactions_by_miner):
        local_params_used_by_miner = []
        for (_, _, post_validation_transaction) in post_validation_transactions_by_miner:
            if post_validation_transaction['update_direction']:
                local_params_used_by_miner.append((post_validation_transaction['worker_device_idx'], post_validation_transaction["local_updates_params"]))
        return local_params_used_by_miner
    
    def set_local_updates_used_info_by_miner(self,post_validation_transactions_by_miner):
        for (_, _, post_validation_transaction) in post_validation_transactions_by_miner:
            if post_validation_transaction['update_direction']:
                self.aggregate_local_updates_info.append({'worker_device_idx':post_validation_transaction['worker_device_idx'], "local_updates_rewards": post_validation_transaction["local_updates_rewards"], "validation_rewards": post_validation_transaction["validation_rewards"],"validation_done_by":post_validation_transaction["validation_done_by"]})
    
    #TODO different aggregate methods and malicious miners
    def aggregate_candidate_model(self, local_update_params_potentially_to_be_used, rewards, log_files_folder_path_comm_round, comm_round):
        print(f"Miner {self.idx} is aggregating candidate model with computation power {self.computation_power} and link speed {round(self.link_speed,3)} bytes/s")
        # filter local_params
        local_params_by_benign_workers = []
        self.aggregate_time = time.time()
        for (worker_device_idx, local_params) in local_update_params_potentially_to_be_used:
            if not worker_device_idx in self.black_list:
                local_params_by_benign_workers.append(local_params)
            else:
                print(f"global update skipped for a worker {worker_device_idx} in {self.idx}'s black list")
        if local_params_by_benign_workers:
            # avg the gradients
            sum_parameters = None
            for local_updates_params in local_params_by_benign_workers:
                if sum_parameters is None:
                    sum_parameters = copy.deepcopy(local_updates_params)
                else:
                    for var in sum_parameters:
                        sum_parameters[var] += local_updates_params[var]
            # number of finally filtered workers' updates
            num_participants = len(local_params_by_benign_workers)
            for var in self.candidate_parameters:
                self.candidate_parameters[var] = (sum_parameters[var] / num_participants)
            print(f"A candidate model is produced by {self.idx} using {num_participants} workers' local updates.")
            self.aggregate_rewards += rewards * num_participants #setting rewards 
            with open(f"{log_files_folder_path_comm_round}/miner_{self.idx}_candidate_model_accuracies_comm_{comm_round}.txt", "a") as file:
                file.write(f"{self.return_idx()} round_{comm_round} {self.return_role()}: {self.validate_model_weights(self.net.state_dict())}\n") #self.validate_model_weights(self.net.state_dict())是否可用？
            try:
                self.aggregate_time = (time.time() - self.aggregate_time)/self.computation_power
            except:
                self.aggregate_time = float('inf')


            if self.is_malicious:
                self.net.apply(self.malicious_miner_add_noise_to_weights)
                print(f"malicious miner {self.idx} has added noise to its candidate global model weights before transmitting")
                with open(f"{log_files_folder_path_comm_round}/comm_{comm_round}_variance_of_noises.txt", "a") as file:
                    file.write(f"{self.return_idx()} {self.return_role()}  noise variances: {self.variance_of_noises}\n")

        else:
            print(f"There are no available local params for {self.idx} to get candidate model in this comm round.")
    
    def malicious_miner_add_noise_to_weights(self, m):
        with torch.no_grad():
            if hasattr(m, 'weight'): #checks if the module m has a weight attribute
                noise = self.noise_variance * torch.randn(m.weight.size())
                variance_of_noise = torch.var(noise)
                m.weight.add_(noise.to(self.dev))
                self.variance_of_noises.append(float(variance_of_noise))

    def return_candidate_model_and_signature(self, comm_round):
        self.candidate_model_dict = {'miner_idx': self.idx, 'in_round_number': comm_round, "candidate_model_params": copy.deepcopy(self.candidate_parameters), "aggregate_rewards": self.aggregate_rewards, "aggregate_spent_time": self.aggregate_time, "aggregate_local_updates_info": self.aggregate_local_updates_info, "miner_rsa_pub_key": self.return_rsa_pub_key()}
        self.candidate_model_dict["miner_signature"] = self.sign_msg(sorted(self.candidate_model_dict.items()))

    ''' Step 5 - workers accept candidate models and broadcast to other workers in their respective peer lists .'''   
    def return_associated_miners(self):
        return vars(self)[f'{self.role}_associated_miner_set']
    
    def return_worker_acception_wait_time(self):
        return self.worker_acception_wait_time
    
    def set_unordered_arrival_time_accepted_miner_candidate(self, unordered_candidate_arrival_queue):
        self.unordered_arrival_time_accepted_miner_candidate = unordered_candidate_arrival_queue     #dict
    
    def set_candidate_for_final_validating_queue(self, final_candidate_arrival_queue):
        self.final_candidate_queue_to_validate = final_candidate_arrival_queue #dict

    def worker_broadcast_miner_candidate(self):
        for peer in self.peer_list:
            if peer.is_online():
                if peer.return_role() == "worker":
                    if not peer.return_idx() in self.black_list:
                        print(f"worker {self.idx} is broadcasting received miner candidate to worker {peer.return_idx()}.")
                        final_broadcasting_unordered_arrival_time_accepted_miner_candidate_for_dest_worker = copy.copy(self.unordered_arrival_time_accepted_miner_candidate)
                        # if offline, it's like the broadcasted transaction was not received, so skip a transaction
                        for arrival_time, tx in self.unordered_arrival_time_accepted_miner_candidate.items():
                            if not (self.online_switcher() and peer.online_switcher()):
                                del final_broadcasting_unordered_arrival_time_accepted_miner_candidate_for_dest_worker[arrival_time]
                        # in the real distributed system, it should be broadcasting transaction one by one. Here we send the all received candidate(while online) and later calculate the order for the individual broadcasting transaction's arrival time mixed with the candidate itself received
                        peer.accept_worker_broadcasted_miner_candidate(self, final_broadcasting_unordered_arrival_time_accepted_miner_candidate_for_dest_worker)
                        print(f"worker {self.idx} has broadcasted {len(final_broadcasting_unordered_arrival_time_accepted_miner_candidate_for_dest_worker)} miner candidate to worker {peer.return_idx()}.")
                    else:
                        print(f"Destination worker {peer.return_idx()} is in this worker {self.idx}'s black_list. broadcasting skipped for this dest worker.")

    def accept_worker_broadcasted_miner_candidate(self, source_worker, unordered_transaction_arrival_queue_from_source_worker):
        if not source_worker.return_idx() in self.black_list:
            unordered_transaction_arrival_queue_from_source_worker_to_added = {}
            for arrival_time, tx in unordered_transaction_arrival_queue_from_source_worker.items():
                if not self.check_if_has_same_candidate_transaction(tx, self.unordered_arrival_time_accepted_miner_candidate, self.worker_accepted_broadcasted_miner_candidate):
                    unordered_transaction_arrival_queue_from_source_worker_to_added[arrival_time] = tx
            self.worker_accepted_broadcasted_miner_candidate.append({'source_worker_link_speed': source_worker.return_link_speed(),'broadcasted_candidate': copy.deepcopy(unordered_transaction_arrival_queue_from_source_worker_to_added)})
            print(f"worker {self.idx} has accepted miner candidate from worker {source_worker.return_idx()}")
        else:
            print(f"Source worker {source_worker.return_idx()} is in worker {self.idx}'s black list. Broadcasted candidate not accepted.")
    
    def check_if_has_same_candidate_transaction(self, transaction_received_from_worker, transactions_received_from_associated_miners, worker_accepted_broadcasted_miner_candidate):
        if_in_transactions_received_from_associated_miners = False
        if_in_worker_accepted_broadcasted_miner_candidate = False
        for _, transaction in transactions_received_from_associated_miners.items():
            if transaction['miner_idx'] == transaction_received_from_worker['miner_idx']:
                if_in_transactions_received_from_associated_miners = True                
                break
        for txs in worker_accepted_broadcasted_miner_candidate:
            for _, tx in txs['broadcasted_candidate'].items():
                if tx['miner_idx'] == transaction_received_from_worker['miner_idx']:
                    if_in_worker_accepted_broadcasted_miner_candidate = True
                    break
        return if_in_transactions_received_from_associated_miners or if_in_worker_accepted_broadcasted_miner_candidate
    
    ''' Step 5.5: with the broadcasted miners candidate models, workers decide the final arrival order'''
    def return_accepted_broadcasted_miner_candidate(self):
        return self.worker_accepted_broadcasted_miner_candidate #list of dicts

    def return_unordered_arrival_time_accepted_miner_candidate(self):
        return  self.unordered_arrival_time_accepted_miner_candidate
    
    ''' Step 6: workers verify miners' signature and miners' candidate models by the order of transaction arrival time.'''
    
    def return_final_candidate_validating_queue(self):
        return self.final_candidate_queue_to_validate
       
    def validate_miner_candidate(self, candidate_to_validate, rewards, log_files_folder_path, comm_round, validate_threshold, malicious_worker_on):
        log_files_folder_path_comm_round = f"{log_files_folder_path}/comm_{comm_round}"
        if self.computation_power == 0:
            print(f"worker {self.idx} has computation power 0 and will not be able to validate this candidate in time")
            return False, False
        else:
            miner_candidate_device_idx = candidate_to_validate["miner_idx"]
            if miner_candidate_device_idx in self.black_list:
                print(f"{miner_candidate_device_idx} is in worker's blacklist. Candidate won't get validated.")
                return False, False
            validation_time = time.time()
            if self.check_signature:
                candidate_before_signed = copy.deepcopy(candidate_to_validate)
                del candidate_to_validate["miner_signature"]
                modulus = candidate_to_validate['miner_rsa_pub_key']["modulus"]
                pub_key = candidate_to_validate['miner_rsa_pub_key']["pub_key"]
                signature = candidate_to_validate["miner_signature"]
                # begin validation
                # 1 - verify signature
                hash = int.from_bytes(sha256(str(sorted(candidate_before_signed.items())).encode('utf-8')).digest(), byteorder='big')
                hashFromSignature = pow(signature, pub_key, modulus)
                if hash == hashFromSignature:
                    print(f"Signature of candidate from miner {miner_candidate_device_idx} is verified by worker {self.idx}!")
                    candidate_to_validate['miner_signature_valid'] = True
                else:
                    print(f"Signature invalid. candidate from miner {miner_candidate_device_idx} does NOT pass verification.")
                    # will also add sig not verified candidate due to the worker's verification effort and its rewards needs to be recorded in the block
                    candidate_to_validate['miner_signature_valid'] = False
            else:
                print(f"Signature of candidate from miner {miner_candidate_device_idx} is verified by worker {self.idx}!")
                candidate_to_validate['miner_signature_valid'] = True
            # 2 - validate miner's candidate model if miner's signature is valid
            if candidate_to_validate['miner_signature_valid']:
                accuracy_by_miner_candidate_using_worker_data = self.validate_model_weights(candidate_to_validate['candidate_model_params'])
                print(f"After applying miner's candidate model, model accuracy becomes - {accuracy_by_miner_candidate_using_worker_data}")
                candidate_to_validate["candidate_validation_accuracy"] = accuracy_by_miner_candidate_using_worker_data
                # record their accuracies and difference for choosing a good validation threshold
                is_malicious_worker = "M" if self.is_malicious else "B"
                with open(f"{log_files_folder_path_comm_round}/worker_{self.idx}_{is_malicious_worker}_validation_records_comm_{comm_round}.txt", "a") as file:
                    is_malicious_node = "M" if self.devices_dict[miner_candidate_device_idx].return_is_malicious() else "B"
                    file.write(f"{accuracy_by_miner_candidate_using_worker_data}: worker {self.return_idx()} {is_malicious_worker} in round {comm_round} evluating miner {miner_candidate_device_idx},  {miner_candidate_device_idx}_maliciousness: {is_malicious_node}\n")
                if accuracy_by_miner_candidate_using_worker_data - self.local_accuracy < validate_threshold * -1:
                    candidate_to_validate['candidate_direction'] = False
                    print(f"NOTE: miner {miner_candidate_device_idx}'s candidate model is deemed as suspiciously malicious by worker {self.idx}")
                    # is it right?
                    if not self.devices_dict[miner_candidate_device_idx].return_is_malicious():
                        print(f"Warning - {miner_candidate_device_idx} is benign and this validation is wrong.")
                        # for experiments
                        with open(f"{log_files_folder_path}/false_negative_good_nodes_inside_victims.txt", 'a') as file:
                            file.write(f"miner {miner_candidate_device_idx}'s candidate model accuracy is {accuracy_by_miner_candidate_using_worker_data}, by worker {self.idx} in round {comm_round}\n")
                    else:
                        with open(f"{log_files_folder_path}/true_negative_malicious_nodes_inside_caught.txt", 'a') as file:
                            file.write(f"miner {miner_candidate_device_idx}'s candidate model accuracy is {accuracy_by_miner_candidate_using_worker_data}, by worker {self.idx} in round {comm_round}\n")
                else:
                    candidate_to_validate['candidate_direction'] = True
                    print(f"miner {miner_candidate_device_idx}'s candidate model is deemed as GOOD by worker {self.idx}")
                    # is it right?
                    if self.devices_dict[miner_candidate_device_idx].return_is_malicious():
                        print(f"Warning - {miner_candidate_device_idx} is malicious and this validation is wrong.")
                        # for experiments
                        with open(f"{log_files_folder_path}/false_positive_malious_nodes_inside_slipped.txt", 'a') as file:
                            file.write(f"miner {miner_candidate_device_idx}'s candidate model accuracy is {accuracy_by_miner_candidate_using_worker_data}, by worker {self.idx} in round {comm_round}\n")
                    else:
                        with open(f"{log_files_folder_path}/true_positive_good_nodes_inside_correct.txt", 'a') as file:
                            file.write(f"miner {miner_candidate_device_idx}'s candidate model accuracy is {accuracy_by_miner_candidate_using_worker_data}, by worker {self.idx} in round {comm_round}\n")
                if self.is_malicious and malicious_worker_on:
                    old_voting = candidate_to_validate['candidate_direction']
                    candidate_to_validate['candidate_direction'] = not candidate_to_validate['candidate_direction']
                    with open(f"{log_files_folder_path_comm_round}/malicious_validator_log.txt", 'a') as file:
                        file.write(f"malicious worker {self.idx} has flipped the voting of miner {miner_candidate_device_idx} from {old_voting} to {candidate_to_validate['candidate_direction']} in round {comm_round}\n")
                candidate_to_validate['validation_rewards'] = rewards
            else:
                candidate_to_validate['candidate_direction'] = 'N/A'
                candidate_to_validate['validation_rewards'] = 0
                candidate_to_validate["candidate_validation_accuracy"] = 'N/A'
            candidate_to_validate['validation_done_by'] = self.idx
            validation_time = (time.time() - validation_time)/self.computation_power
            candidate_to_validate['validation_time'] = validation_time
            candidate_to_validate['worker_rsa_pub_key'] = self.return_rsa_pub_key()
            # assume signing done in negligible time
            candidate_to_validate["worker_signature"] = self.sign_msg(sorted(candidate_to_validate.items()))
            return validation_time, candidate_to_validate
        
    def add_post_validation_candidate_to_queue(self, candidate_to_add):
        self.post_validation_candidate_queue.append(candidate_to_add)       

    ''' Step 7 - workers send post validation candidate transactions to associated miner and miner broadcasts these to other miners in their respecitve peer lists.\n'''
    def return_post_validation_candidate_queue(self):
        return self.post_validation_candidate_queue

    def set_unordered_arrival_time_accepted_worker_validated_candidate_transactions(self, unordered_candidate_transaction_arrival_queue):
        self.unordered_arrival_time_accepted_worker_validated_candidate_transactions = unordered_candidate_transaction_arrival_queue
    
    def set_candidate_transactions_for_final_mining_queue(self, final_transactions_arrival_queue):
        self.final_candidate_transactions_queue_to_mine = final_transactions_arrival_queue
        
    def miner_broadcast_worker_validated_candidate_transactions(self):
        for peer in self.peer_list:
            if peer.is_online():
                if peer.return_role() == "miner":
                    if not peer.return_idx() in self.black_list:
                        print(f"miner {self.idx} is broadcasting received miner transactions to miner {peer.return_idx()}.")
                        final_broadcasting_unordered_arrival_time_accepted_worker_validated_candidate_transactions_for_dest_miner = copy.copy(self.unordered_arrival_time_accepted_worker_validated_candidate_transactions)
                        # if offline, it's like the broadcasted transaction was not received, so skip a transaction
                        for arrival_time, tx in self.unordered_arrival_time_accepted_worker_validated_candidate_transactions.items():
                            if not (self.online_switcher() and peer.online_switcher()):
                                del final_broadcasting_unordered_arrival_time_accepted_worker_validated_candidate_transactions_for_dest_miner[arrival_time]
                        # in the real distributed system, it should be broadcasting transaction one by one. Here we send the all received transactions(while online) and later calculate the order for the individual broadcasting transaction's arrival time mixed with the transactions itself received
                        peer.accept_miner_broadcasted_worker_validated_candidate_transactions(self, final_broadcasting_unordered_arrival_time_accepted_worker_validated_candidate_transactions_for_dest_miner)
                        print(f"miner {self.idx} has broadcasted {len(final_broadcasting_unordered_arrival_time_accepted_worker_validated_candidate_transactions_for_dest_miner)} worker transactions to miner {peer.return_idx()}.")
                    else:
                        print(f"Destination miner {peer.return_idx()} is in this miner {self.idx}'s black_list. broadcasting skipped for this dest miner.")

    def accept_miner_broadcasted_worker_validated_candidate_transactions(self, source_worker, unordered_candidate_transaction_arrival_queue_from_source_miner):
        if not source_worker.return_idx() in self.black_list:
            self.accepted_miner_broadcasted_worker_validated_candidate_transactions.append({'source_miner_link_speed': source_worker.return_link_speed(),'broadcasted_validated_candidate_transactions': copy.deepcopy(unordered_candidate_transaction_arrival_queue_from_source_miner)})
            print(f"worker {self.idx} has accepted miner candidate transactions from worker {source_worker.return_idx()}")
        else:
            print(f"Source worker {source_worker.return_idx()} is in worker {self.idx}'s black list. Broadcasted transactions not accepted.")

    ''' Step 7.5: with the broadcasted validated candidate transactions, miners decide the final candidate transaction arrival order. '''
    def return_accepted_miner_broadcasted_worker_validated_candidate_transactions(self):
        return self.accepted_miner_broadcasted_worker_validated_candidate_transactions
        
    def return_unordered_arrival_time_accepted_worker_validated_candidate_transactions(self):
        return self.unordered_arrival_time_accepted_worker_validated_candidate_transactions

    ''' Step 8: miners do self and cross-verification (verify signature) by the order of transaction arrival time 
    and record the transactions in the candidate block according to the limit size. 
    Also mine and propagate the block.'''    

    def return_final_candidate_transactions_mining_queue(self):
        return self.final_candidate_transactions_queue_to_mine

    def return_miner_acception_wait_time(self):
        return self.miner_acception_wait_time

    def return_miner_accepted_transactions_size_limit(self):
        return self.miner_accepted_transactions_size_limit

    def verify_worker_transaction(self, transaction_to_verify):
        if self.computation_power == 0:
            print(f"miner {self.idx} has computation power 0 and will not be able to verify this transaction in time")
            return False, None
        else:
            transaction_worker_idx = transaction_to_verify['validation_done_by']
            if transaction_worker_idx in self.black_list:
                print(f"{transaction_worker_idx} is in miner's blacklist. Trasaction won't get verified.")
                return False, None
            verification_time = time.time()
            if self.check_signature:
                transaction_before_signed = copy.deepcopy(transaction_to_verify)
                del transaction_before_signed["worker_signature"]
                modulus = transaction_to_verify['worker_rsa_pub_key']["modulus"]
                pub_key = transaction_to_verify['worker_rsa_pub_key']["pub_key"]
                signature = transaction_to_verify["worker_signature"]
                # begin verification
                hash = int.from_bytes(sha256(str(sorted(transaction_before_signed.items())).encode('utf-8')).digest(), byteorder='big')
                hashFromSignature = pow(signature, pub_key, modulus)
                if hash == hashFromSignature:
                    print(f"Signature of transaction from worker {transaction_worker_idx} is verified by {self.role} {self.idx}!")
                    verification_time = (time.time() - verification_time)/self.computation_power
                    return verification_time, True
                else:
                    print(f"Signature invalid. Transaction from worker {transaction_worker_idx} is NOT verified.")
                    return (time.time() - verification_time)/self.computation_power, False
            else:
                print(f"Signature of transaction from worker {transaction_worker_idx} is verified by {self.role} {self.idx}!")
                verification_time = (time.time() - verification_time)/self.computation_power
                return verification_time, True

    def sign_candidate_transaction(self, candidate_transaction):
        signing_time = time.time()
        candidate_transaction['miner_rsa_pub_key'] = self.return_rsa_pub_key()
        if 'miner_signature' in candidate_transaction.keys():
            del candidate_transaction['miner_signature']
        candidate_transaction["miner_signature"] = self.sign_msg(sorted(candidate_transaction.items()))
        signing_time = (time.time() - signing_time)/self.computation_power
        return signing_time
    
    def set_block_generation_time_point(self, block_generation_time_point):
        self.block_generation_time_point = block_generation_time_point

    # def mine_block(self, candidate_block, rewards):
    #     candidate_block.set_mined_by(self.idx)
    #     mined_block, leader_idx, max_candidate_model_accuracy = self.proof_of_endorsement(candidate_block)
    #     # pow_mined_block.set_mined_by(self.idx)
    #     mined_block.set_mining_rewards(rewards)
    #     mined_block.set_leader_id(self)
    #     mined_block.set_max_candidate_model_accuracy(max_candidate_model_accuracy)
    #     return mined_block

    #TODO find the leader among miners
        #TODO find the leader among miners
    # def proof_of_endorsement(self, candidate_block):
    #     candidate_block.set_mined_by(self.idx)
    #     transactions_in_candidate_block = candidate_block.return_transactions() 
    #     valid_sig_transacitons_in_candidate_block = transactions_in_candidate_block["valid_worker_sig_transacitons"] #list, 元素是字典变量post_validation_candidate
    #     for transaciton_in_candidate_block in valid_sig_transacitons_in_candidate_block: 
    #         supported_infos = transaciton_in_candidate_block['supported_workers']
    #         average_accuracy_of_this_candidate_model = sum(info['candidate_model_accuracy'] for info in supported_infos) / len(supported_infos)
    #         transaciton_in_candidate_block['average_accuracy_of_this_candidate_model'] = average_accuracy_of_this_candidate_model
    #     sorted_valid_sig_transacitons_in_candidate_block = sorted(valid_sig_transacitons_in_candidate_block, key=lambda x: x['average_accuracy_of_this_candidate_model'], reverse=True)
    #     transactions_in_candidate_block["valid_worker_sig_transacitons"] = sorted_valid_sig_transacitons_in_candidate_block
    #     leader_idx = sorted_valid_sig_transacitons_in_candidate_block[0]['miner_idx']
    #     max_candidate_model_accuracy = sorted_valid_sig_transacitons_in_candidate_block[0]['average_accuracy_of_this_candidate_model']
    #     current_hash = candidate_block.compute_hash()
    #     candidate_block.set_hash(current_hash)
    #     return candidate_block, leader_idx, max_candidate_model_accuracy
         
    def find_leader_and_max_accuracy_among_valid_candidate_transacitons(self, valid_sig_transacitons):
        for transaciton in valid_sig_transacitons: 
            supported_infos = transaciton['supported_workers']
            if len(supported_infos) != 0:
                average_accuracy_of_this_candidate_model = sum(info['candidate_model_accuracy'] for info in supported_infos) / len(supported_infos)
                transaciton['average_accuracy_of_this_candidate_model'] = average_accuracy_of_this_candidate_model
            else:
                transaciton['average_accuracy_of_this_candidate_model'] = 0
        sorted_valid_sig_transacitons = sorted(valid_sig_transacitons, key=lambda x: x['average_accuracy_of_this_candidate_model'], reverse=True)
        leader_idx = sorted_valid_sig_transacitons[0]['miner_idx']
        max_candidate_model_accuracy = sorted_valid_sig_transacitons[0]['average_accuracy_of_this_candidate_model']

        return leader_idx, max_candidate_model_accuracy, sorted_valid_sig_transacitons       

    def sign_block(self, block_to_sign):
        block_to_sign.set_signature(self.sign_msg(block_to_sign.__dict__))

    def set_mined_rewards(self, mined_rewards):
        self.mined_rewards = mined_rewards

    def set_mined_block(self, mined_block):
        self.mined_block = mined_block

    def propagated_the_block(self, propagating_time_point, block_to_propagate):
        for peer in self.peer_list:
            if peer.is_online():
                if peer.return_role() == "miner":
                    if not peer.return_idx() in self.black_list:
                        print(f"{self.role} {self.idx} is propagating its mined block to {peer.return_role()} {peer.return_idx()}.")
                        if peer.online_switcher():
                            peer.accept_the_propagated_block(self, propagating_time_point, block_to_propagate)
                    else:
                        print(f"Destination miner {peer.return_idx()} is in {self.role} {self.idx}'s black_list. Propagating skipped for this dest miner.")
   
    def accept_the_propagated_block(self, source_miner, source_miner_propagating_time_point, propagated_block):
        if not source_miner.return_idx() in self.black_list:
            source_miner_link_speed = source_miner.return_link_speed()
            this_miner_link_speed = self.link_speed
            lower_link_speed = this_miner_link_speed if this_miner_link_speed < source_miner_link_speed else source_miner_link_speed
            transmission_delay = getsizeof(str(propagated_block.__dict__))/lower_link_speed
            self.unconfirmed_candidate_block = propagated_block
            self.unconfirmed_candidate_block_arrival_time = source_miner_propagating_time_point + transmission_delay
            #self.unordered_propagated_block_processing_queue[source_miner_propagating_time_point + transmission_delay] = propagated_block
            print(f"{self.role} {self.idx} has accepted a propagated block from miner {source_miner.return_idx()}")
        else:
            print(f"Source miner {source_miner.return_role()} {source_miner.return_idx()} is in {self.role} {self.idx}'s black list. Propagated block not accepted.")
    

    ''' Step 9: miners decide if adding a propagated block or its own mined block as the legitimate block, and request its associated devices to download this block'''
    
    def return_unconfirmed_candidate_block(self):
        return self.unconfirmed_candidate_block
    
    def return_unconfirmed_candidate_block_arrival_time(self):
        return self.unconfirmed_candidate_block_arrival_time
        
    # def return_unordered_propagated_block_processing_queue(self):
    #     return self.unordered_propagated_block_processing_queue

    def return_mined_block(self):
        return self.mined_block
        
    def verify_block(self, block_to_verify, sending_miner):
        if not self.online_switcher():
            print(f"{self.idx} goes offline when verifying a block")
            return False, False
        verification_time = time.time()
        if sending_miner in self.black_list:
            print(f"The miner propagating/sending this block {sending_miner} is in {self.idx}'s black list. Block will not be verified.")
            return False, False
        if sending_miner in self.black_list:
            print(f"The miner {sending_miner} mined this block is in {self.idx}'s black list. Block will not be verified.")
            return False, False
        # check if the proof is valid(verify _block_hash).
        if not self.check_hash(block_to_verify):
            print(f"Hash of the block from miner {self.idx} is not verified.")
            return False, False
        # check if miner's signature is valid
        if self.check_signature:
            signature_dict = block_to_verify.return_miner_rsa_pub_key()
            modulus = signature_dict["modulus"]
            pub_key = signature_dict["pub_key"]
            signature = block_to_verify.return_signature()
            # verify signature
            block_to_verify_before_sign = copy.deepcopy(block_to_verify)
            block_to_verify_before_sign.remove_signature_for_verification()
            hash = int.from_bytes(sha256(str(block_to_verify_before_sign.__dict__).encode('utf-8')).digest(), byteorder='big')
            hashFromSignature = pow(signature, pub_key, modulus)
            if hash != hashFromSignature:
                print(f"Signature of the block sent by miner {sending_miner} mined by miner {mined_by} is not verified by {self.role} {self.idx}.")
                return False, False
            # check previous hash based on own chain
            last_block = self.return_blockchain_object().return_last_block()
            if last_block is not None:
                # check if the previous_hash referred in the block and the hash of latest block in the chain match.
                last_block_hash = last_block.compute_hash(hash_entire_block=True)
                if block_to_verify.return_previous_block_hash() != last_block_hash:
                    print(f"Block sent by miner {sending_miner} mined by miner {mined_by} has the previous hash recorded as {block_to_verify.return_previous_block_hash()}, but the last block's hash in chain is {last_block_hash}. This is possibly due to a forking event from last round. Block not verified and won't be added. Device needs to resync chain next round.")
                    return False, False
        # check if mined by leader
        recorded_transactions = block_to_verify.return_transactions()
        leader_idx = recorded_transactions['valid_worker_sig_transacitons'][0]['miner_idx']
        if sending_miner != leader_idx:
            print(f"The block sent by miner {sending_miner} is not mined by the leader {leader_idx}. Block not verified and won't be added. Device needs to resync chain next round.")
            return False, False
        # All verifications done.
        print(f"Block accepted from miner {sending_miner} has been verified by {self.idx}!")
        verification_time = (time.time() - verification_time)/self.computation_power
        return block_to_verify, verification_time

    def check_hash(self, block_to_check):
        # remove its block hash(compute_hash() by default) to verify hash as block hash was set after pow
        hash = block_to_check.return_hash()
        # print("hash", hash)
        # print("compute_hash", block_to_check.compute_hash())
        #It checks if the PoW starts with a string of zeros whose length is equal to the PoW difficulty specified in the class (self.pow_difficulty).
        #It also verifies that the PoW matches the hash of the block (block_to_check.compute_hash()).
        return hash == block_to_check.compute_hash() 

    def add_block(self, block_to_add):
        self.return_blockchain_object().append_block(block_to_add)
        print(f"d_{self.idx.split('_')[-1]} - {self.role[0]} has appened a block to its chain. Chain length now - {self.return_blockchain_object().return_chain_length()}")
        # TODO delete has_added_block
        # self.has_added_block = True
        self.the_added_block = block_to_add
        return True      

    def set_block_download_time(self, block_download_time):
        self.block_download_time = block_download_time
    
    def return_block_download_time(self):
        return self.block_download_time

    # def request_to_download(self, block_to_download, requesting_time_point):
    #     print(f"miner {self.idx} is requesting its associated devices to download the block it just added to its chain")
    #     devices_in_association = self.miner_associated_worker_set
    #     for device in devices_in_association:
    #         # theoratically, one device is associated to a specific miner, so we don't have a miner_block_arrival_queue here
    #         if self.online_switcher() and device.online_switcher():
    #             miner_link_speed = self.return_link_speed()
    #             device_link_speed = device.return_link_speed()
    #             lower_link_speed = device_link_speed if device_link_speed < miner_link_speed else miner_link_speed
    #             transmission_delay = getsizeof(str(block_to_download.__dict__))/lower_link_speed
    #             verified_block, verification_time = device.verify_block(block_to_download, block_to_download.return_mined_by())
    #             if verified_block:
    #                 # forgot to check for maliciousness of the block miner
    #                 device.add_block(verified_block)
    #             device.add_to_round_end_time(requesting_time_point + transmission_delay + verification_time)
    #         else:
    #             print(f"Unfortunately, either miner {self.idx} or {device.return_idx()} goes offline while processing this request-to-download block.")
    def request_to_download(self, block_to_download, requesting_time_point):
        print(f"miner {self.idx} is requesting its associated devices to download the block it just added to its chain")
        devices_in_association = self.miner_associated_worker_set
        for device in devices_in_association:
            # theoratically, one device is associated to a specific miner, so we don't have a miner_block_arrival_queue here
            if self.online_switcher() and device.online_switcher():
                device.download_the_propagated_block(self, requesting_time_point, block_to_download)
                # miner_link_speed = self.return_link_speed()
                # device_link_speed = device.return_link_speed()
                # lower_link_speed = device_link_speed if device_link_speed < miner_link_speed else miner_link_speed
                # transmission_delay = getsizeof(str(block_to_download.__dict__))/lower_link_speed
                # verified_block, verification_time = device.verify_block(block_to_download, block_to_download.return_mined_by())
                # if verified_block:
                #     # forgot to check for maliciousness of the block miner
                #     device.add_block(verified_block)
                # device.add_to_round_end_time(requesting_time_point + transmission_delay + verification_time)
            else:
                print(f"Unfortunately, either miner {self.idx} or {device.return_idx()} goes offline while processing this request-to-download block.")

    def download_the_propagated_block(self, source_miner, source_miner_propagating_time_point, propagated_block):
        if not source_miner.return_idx() in self.black_list:
            source_miner_link_speed = source_miner.return_link_speed()
            device_link_speed = self.link_speed
            lower_link_speed = device_link_speed if device_link_speed < source_miner_link_speed else source_miner_link_speed
            transmission_delay = getsizeof(str(propagated_block.__dict__))/lower_link_speed
            self.unordered_downloaded_block_processing_queue[source_miner_propagating_time_point + transmission_delay] = propagated_block
            print(f"{self.role} {self.idx} has accepted a propagated block from miner {source_miner.return_idx()}")
        else:
            print(f"Source miner {source_miner.return_role()} {source_miner.return_idx()} is in {self.role} {self.idx}'s black list. Propagated block not accepted.")
    
    def return_unordered_downloaded_block_processing_queue(self):
        return self.unordered_downloaded_block_processing_queue

    ''' 
    Step 10 last step : process the added block 
    1.collect usable candidate models 
    2.malicious nodes identification
    3.get rewards
    This code block is skipped if no valid block was generated in this round
    '''

    # Main function
    # also accumulate rewards here
    def process_block(self, block_to_process, log_files_folder_path, conn, conn_cursor, when_resync=False):
        # get the global model from the block
        # get rewards
        # return process time
        # malicious nodes identification(TODO)
        processing_time = time.time()
        if not self.online_switcher():
            print(f"{self.role} {self.idx} goes offline when processing the added block. Model not updated and rewards information not upgraded. Outdated information may be obtained by this node if it never resyncs to a different chain.") # may need to set up a flag indicating if a block has been processed
        if block_to_process:
            mined_by = block_to_process.return_mined_by()
            if mined_by in self.black_list:
                # in this system black list is also consistent across devices as it is calculated based on the information on chain, but individual device can decide its own validation/verification mechanisms and has its own 
                print(f"The added block is mined by miner {block_to_process.return_mined_by()}, which is in this device's black list. Block will not be processed.")
            else:
                # process worker sig valid transactions
                # used to count positive and negative transactions worker by worker, select the transaction to do global update and identify potential malicious worker
                self_rewards_accumulator = 0
                #valid_transactions_records_by_worker = {}#####
                valid_worker_sig_transacitons_in_block = block_to_process.return_transactions()['valid_worker_sig_transacitons']
                comm_round = block_to_process.return_block_idx()
                #self.active_worker_record_by_round[comm_round] = set()#####
                finally_used_candidate_model_params = []
                for valid_worker_sig_miner_transaciton in valid_worker_sig_transacitons_in_block:
                    # verify miner's signature
                    if self.verify_miner_transaction_by_signature(valid_worker_sig_miner_transaciton, mined_by):###check def verify_miner_transaction_by_signature
                        miner_device_idx = valid_worker_sig_miner_transaciton['miner_idx'] # Extract miner Device Index
                        aggregate_local_updates_info = valid_worker_sig_miner_transaciton['aggregate_local_updates_info'] # [{'worker_device_idx':post_validation_transaction['worker_device_idx'], "local_updates_rewards": post_validation_transaction["local_updates_rewards"], "validation_rewards": post_validation_transaction["validation_rewards"],"validation_done_by":post_validation_transaction["validation_done_by”}]
                        supported_workers = valid_worker_sig_miner_transaciton['supported_workers'] #[{this_candidate_tx_info},{this_candidate_tx_info}]
                        opposed_workers = valid_worker_sig_miner_transaciton['opposed_workers']
                        
                        #get rewards
                        # give leader rewards
                        if self.idx == miner_device_idx:
                            self_rewards_accumulator += valid_worker_sig_miner_transaciton['aggregate_rewards']
                        # give worker validate rewards
                        for validate_worker_record in supported_workers + opposed_workers:
                            if self.idx == validate_worker_record['validation_done_by_worker']:
                                self_rewards_accumulator += validate_worker_record['validation_reward_for_worker']
                        # give worker update rewards and give miner validate rewards
                        for local_update_info in aggregate_local_updates_info:
                            if self.idx == local_update_info["worker_device_idx"]:
                                self_rewards_accumulator += local_update_info["local_updates_rewards"]
                            if self.idx == local_update_info["validation_done_by"]:
                                self_rewards_accumulator += local_update_info["validation_rewards"]
                        
                        # TODO: #get can be used candidate models to update personalized global model from the block
                        # if len(valid_worker_sig_transacitons_in_block) == 1:
                        #     self.global_parameters = copy.deepcopy(valid_worker_sig_miner_transaciton["candidate_model_params"])
                        # else:
                        #     finally_used_candidate_model_params.append(valid_worker_sig_miner_transaciton["candidate_model_params"])
                    else:
                        print(f"one validator transaction miner sig found invalid in this block. {self.idx} will drop this block and roll back rewards information")
                        return
                #TODO personalized update    
                # if len(valid_worker_sig_transacitons_in_block) > 1:
                #     print(f"more than one candidate models found in this block. {self.idx} will aggregate these candidate models as the final global model")
                #     if self.online_switcher():
                #         self.global_update(finally_used_candidate_model_params)
                #     else:
                #         print(f"Unfortunately, {self.role} {self.idx} goes offline when it's doing global_updates.")
                
                #get the final global model，i.e., the leader's candidate model
                self.global_parameters = copy.deepcopy(valid_worker_sig_transacitons_in_block[0]["candidate_model_params"])

                # miner gets mining rewards(TODO)
                # if self.idx == mined_by:
                #     self_rewards_accumulator += block_to_process.return_mining_rewards()
                # set received rewards this round based on info from this block
                self.receive_rewards(self_rewards_accumulator)
                print(f"{self.role} {self.idx} has received total {self_rewards_accumulator} rewards for this comm round.")

                # # collect usable worker updates and do global updates
                # finally_used_local_params = []
                # # record True Positive, False Positive, True Negative and False Negative for identified workers
                # TP, FP, TN, FN = 0, 0, 0, 0
                # for worker_device_idx, local_params_record in valid_transactions_records_by_worker.items():
                #     is_worker_malicious = self.devices_dict[worker_device_idx].return_is_malicious()
                #     if local_params_record['finally_used_params']:
                #         # identified as benigh worker
                #         finally_used_local_params.append((worker_device_idx, local_params_record['finally_used_params'])) # could be None
                #         if not is_worker_malicious:
                #             TP += 1
                #         else:
                #             FP += 1
                #     else:
                #         # identified as malicious worker
                #         if is_worker_malicious:
                #             TN += 1
                #         else:
                #             FN += 1
                # if self.online_switcher():
                #     self.global_update(finally_used_local_params)
                # else:
                #     print(f"Unfortunately, {self.role} {self.idx} goes offline when it's doing global_updates.")
        
        # malicious_worker_validation_log_path = f"{log_files_folder_path}/comm_{comm_round}/malicious_worker_validation_log.txt"
        # if not os.path.exists(malicious_worker_validation_log_path):
        #     with open(malicious_worker_validation_log_path, 'w') as file:
        #         accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN) else 0
        #         precision = TP / (TP + FP) if TP else 0
        #         recall = TP / (TP + FN) if TP else 0
        #         f1 = precision * recall / (precision + recall) if precision * recall else 0
        #         file.write(f"In comm_{comm_round} of validating workers, TP = {TP}, FP = {FP}, TN = {TN}, FN = {FN}. \
        #                 \nAccuracy = {accuracy}, Precision = {precision}, Recall = {recall}, F1 Score = {f1}")
                
        processing_time = (time.time() - processing_time)/self.computation_power
        return processing_time
    
    def verify_miner_transaction_by_signature(self, transaction_to_verify, miner_device_idx):
        if miner_device_idx in self.black_list:
            print(f"{miner_device_idx} is in miner's blacklist. Trasaction won't get verified.")
            return False
        if self.check_signature:
            transaction_before_signed = copy.deepcopy(transaction_to_verify)
            del transaction_before_signed["miner_signature"]
            modulus = transaction_to_verify['miner_rsa_pub_key']["modulus"]
            pub_key = transaction_to_verify['miner_rsa_pub_key']["pub_key"]
            signature = transaction_to_verify["miner_signature"]
            # verify
            hash = int.from_bytes(sha256(str(sorted(transaction_before_signed.items())).encode('utf-8')).digest(), byteorder='big')
            hashFromSignature = pow(signature, pub_key, modulus)
            if hash == hashFromSignature:
                print(f"A transaction recorded by miner {miner_device_idx} in the block is verified!")
                return True
            else:
                print(f"Signature invalid. Transaction recorded by {miner_device_idx} is NOT verified.")
                return False
        else:
            print(f"A transaction recorded by miner {miner_device_idx} in the block is verified!")
            return True
    
    # TODO 更改聚合
    def global_update(self, local_update_params_potentially_to_be_used):
        # filter local_params
        local_params_by_benign_workers = []
        for (worker_device_idx, local_params) in local_update_params_potentially_to_be_used:
            if not worker_device_idx in self.black_list:
                local_params_by_benign_workers.append(local_params)
            else:
                print(f"global update skipped for a worker {worker_device_idx} in {self.idx}'s black list")
        if local_params_by_benign_workers:
            # avg the gradients
            sum_parameters = None
            for local_updates_params in local_params_by_benign_workers:
                if sum_parameters is None:
                    sum_parameters = copy.deepcopy(local_updates_params)
                else:
                    for var in sum_parameters:
                        sum_parameters[var] += local_updates_params[var]
            # number of finally filtered workers' updates
            num_participants = len(local_params_by_benign_workers)
            for var in self.global_parameters: 
                self.global_parameters[var] = (sum_parameters[var] / num_participants)
            print(f"global updates done by {self.idx}")
        else:
            print(f"There are no available local params for {self.idx} to perform global updates in this comm round.")       

    
    def other_tasks_at_the_end_of_comm_round(self, this_comm_round, log_files_folder_path):
        self.kick_out_slow_or_lazy_workers(this_comm_round, log_files_folder_path)

    def kick_out_slow_or_lazy_workers(self, this_comm_round, log_files_folder_path):
        for device in self.peer_list:
            if device.return_role() == 'worker':
                if this_comm_round in self.active_worker_record_by_round.keys():
                    if not device.return_idx() in self.active_worker_record_by_round[this_comm_round]:
                        not_active_accumulator = 1
                        # check if not active for the past (lazy_worker_knock_out_rounds - 1) rounds
                        for comm_round_to_check in range(this_comm_round - self.lazy_worker_knock_out_rounds + 1, this_comm_round):
                            if comm_round_to_check in self.active_worker_record_by_round.keys():
                                if not device.return_idx() in self.active_worker_record_by_round[comm_round_to_check]:
                                    not_active_accumulator += 1
                        if not_active_accumulator == self.lazy_worker_knock_out_rounds:
                            # kick out
                            self.black_list.add(device.return_idx())
                            msg = f"worker {device.return_idx()} has been regarded as a lazy worker by {self.idx} in comm_round {this_comm_round}.\n"
                            with open(f"{log_files_folder_path}/kicked_lazy_workers.txt", 'a') as file:
                                file.write(msg)
                else:
                    # this may happen when a device is put into black list by every worker in a certain comm round
                    pass

    def add_to_round_end_time(self, time_to_add):
        self.round_end_time += time_to_add

    ''' Common Methods '''

    ''' setters '''

    def set_devices_dict_and_aio(self, devices_dict, aio):
        self.devices_dict = devices_dict
        self.aio = aio
    
    def generate_rsa_key(self):
        keyPair = RSA.generate(bits=1024)
        self.modulus = keyPair.n
        self.private_key = keyPair.d
        self.public_key = keyPair.e
    
    def init_global_parameters(self):
        self.initial_net_parameters = self.net.state_dict()
        self.global_parameters = self.net.state_dict()
    
    def init_candidate_parameters(self):
        self.initial_net_parameters = self.net.state_dict()
        self.candidate_parameters = self.net.state_dict()

    def assign_role(self):
        # equal probability
        role_choice = random.randint(0, 1)
        if role_choice == 0:
            self.role = "worker"
        else:
            self.role = "miner"

    # used for hard_assign
    def assign_miner_role(self):
        self.role = "miner"

    def assign_worker_role(self):
        self.role = "worker"

    ''' getters '''

    def return_idx(self):
        return self.idx
    
    def return_rsa_pub_key(self):
        return {"modulus": self.modulus, "pub_key": self.public_key}

    def return_peers(self):
        return self.peer_list

    def return_role(self):
        return self.role

    def is_online(self):
        return self.online

    def return_is_malicious(self):
        return self.is_malicious

    def return_black_list(self):
        return self.black_list

    def return_blockchain_object(self):
        return self.blockchain

    def return_stake(self):
        return self.rewards

    def return_computation_power(self):
        return self.computation_power

    def return_the_added_block(self):
        return self.the_added_block

    def return_round_end_time(self):
        return self.round_end_time
    
    def return_candidate_parameters(self):
        return self.candidate_parameters

    ''' functions '''
    
    def sign_msg(self, msg): #msg: The message to be signed.
        hash = int.from_bytes(sha256(str(msg).encode('utf-8')).digest(), byteorder='big') #The digest is converted to an integer (hash) using int.from_bytes, specifying the byte order as big endian.
        # pow() is python built-in modular exponentiation function
        signature = pow(hash, self.private_key, self.modulus)
        return signature

    def add_peers(self, new_peers):
        if isinstance(new_peers, Device): #The isinstance() function in Python is used to check whether an object belongs to a specified class or its subclasses.
            self.peer_list.add(new_peers)
        else:
            self.peer_list.update(new_peers)

    def remove_peers(self, peers_to_remove):
        if isinstance(peers_to_remove, Device):
            self.peer_list.discard(peers_to_remove)
        else:
            self.peer_list.difference_update(peers_to_remove)

    def online_switcher(self):
        old_status = self.online
        online_indicator = random.random()
        if online_indicator < self.network_stability:
            self.online = True
            # if back online, update peer and resync chain
            if old_status == False:
                print(f"{self.idx} goes back online.")
                # update peer list
                self.update_peer_list()
                # resync chain
                if self.pow_resync_chain():
                    self.update_model_after_chain_resync()
        else:
            self.online = False
            print(f"{self.idx} goes offline.")
        return self.online

    def update_peer_list(self):
        print(f"\n{self.idx} - {self.role} is updating peer list...")
        old_peer_list = copy.copy(self.peer_list)
        online_peers = set()
        for peer in self.peer_list:
            if peer.is_online():
                online_peers.add(peer)
        # for online peers, suck in their peer list
        for online_peer in online_peers:
            self.add_peers(online_peer.return_peers())
        # remove itself from the peer_list if there is
        self.remove_peers(self)
        # remove malicious peers
        removed_peers = []
        potential_malicious_peer_set = set()
        for peer in self.peer_list:
            if peer.return_idx() in self.black_list:
                potential_malicious_peer_set.add(peer)
        self.remove_peers(potential_malicious_peer_set)
        removed_peers.extend(potential_malicious_peer_set)
        # print updated peer result
        if old_peer_list == self.peer_list:
            print("Peer list NOT changed.")
        else:
            print("Peer list has been changed.")
            added_peers = self.peer_list.difference(old_peer_list)
            if potential_malicious_peer_set:
                print("These malicious peers are removed")
                for peer in removed_peers:
                    print(f"d_{peer.return_idx().split('_')[-1]} - {peer.return_role()[0]}", end=', ')
                print()
            if added_peers:
                print("These peers are added")
                for peer in added_peers:
                    print(f"d_{peer.return_idx().split('_')[-1]} - {peer.return_role()[0]}", end=', ')
                print()
            print("Final peer list:")
            for peer in self.peer_list:
                print(f"d_{peer.return_idx().split('_')[-1]} - {peer.return_role()[0]}", end=', ')
            print()
        # WILL ALWAYS RETURN TRUE AS OFFLINE PEERS WON'T BE REMOVED ANY MORE, UNLESS ALL PEERS ARE MALICIOUS...but then it should not register with any other peer. Original purpose - if peer_list ends up empty, randomly register with another device
        return False if not self.peer_list else True
  
    def check_chain_validity(self, chain_to_check):
        chain_len = chain_to_check.return_chain_length()
        if chain_len == 0 or chain_len == 1:
            pass
        else:
            chain_to_check = chain_to_check.return_chain_structure()
            for block in chain_to_check[1:]:
                if self.check_hash(block) and block.return_previous_block_hash() == chain_to_check[chain_to_check.index(block) - 1].compute_hash(hash_entire_block=True):
                    pass
                else:
                    return False
        return True

    def accumulate_chain_stake(self, chain_to_accumulate):
        accumulated_stake = 0
        chain_to_accumulate = chain_to_accumulate.return_chain_structure()
        for block in chain_to_accumulate:
            accumulated_stake += self.devices_dict[block.return_mined_by()].return_stake()
        return accumulated_stake

    def resync_chain(self, mining_consensus):
        if self.not_resync_chain:
            return # temporary workaround to save GPU memory
        if mining_consensus == 'PoW': #consensus
            self.pow_resync_chain()
        else:
            self.pos_resync_chain()

    def pos_resync_chain(self):
        print(f"{self.role} {self.idx} is looking for a chain with the highest accumulated miner's stake in the network...")
        highest_stake_chain = None
        updated_from_peer = None
        curr_chain_stake = self.accumulate_chain_stake(self.return_blockchain_object())
        for peer in self.peer_list:
            if peer.is_online():
                peer_chain = peer.return_blockchain_object()
                peer_chain_stake = self.accumulate_chain_stake(peer_chain)
                if peer_chain_stake > curr_chain_stake:
                    if self.check_chain_validity(peer_chain):
                        print(f"A chain from {peer.return_idx()} with total stake {peer_chain_stake} has been found (> currently compared chain stake {curr_chain_stake}) and verified.")
                        # Higher stake valid chain found!
                        curr_chain_stake = peer_chain_stake
                        highest_stake_chain = peer_chain
                        updated_from_peer = peer.return_idx()
                    else:
                        print(f"A chain from {peer.return_idx()} with higher stake has been found BUT NOT verified. Skipped this chain for syncing.")
        if highest_stake_chain:
            # compare chain difference
            highest_stake_chain_structure = highest_stake_chain.return_chain_structure()
            # need more efficient machenism which is to reverse updates by # of blocks
            self.return_blockchain_object().replace_chain(highest_stake_chain_structure)
            print(f"{self.idx} chain resynced from peer {updated_from_peer}.")
            #return block_iter
            return True 
        print("Chain not resynced.")
        return False

    def pow_resync_chain(self):
        print(f"{self.role} {self.idx} is looking for a longer chain in the network...")
        longest_chain = None
        updated_from_peer = None
        curr_chain_len = self.return_blockchain_object().return_chain_length()
        for peer in self.peer_list:
            if peer.is_online():
                peer_chain = peer.return_blockchain_object()
                if peer_chain.return_chain_length() > curr_chain_len:
                    if self.check_chain_validity(peer_chain):
                        print(f"A longer chain from {peer.return_idx()} with chain length {peer_chain.return_chain_length()} has been found (> currently compared chain length {curr_chain_len}) and verified.")
                        # Longer valid chain found!
                        curr_chain_len = peer_chain.return_chain_length()
                        longest_chain = peer_chain
                        updated_from_peer = peer.return_idx()
                    else:
                        print(f"A longer chain from {peer.return_idx()} has been found BUT NOT verified. Skipped this chain for syncing.")
        if longest_chain:
            # compare chain difference
            longest_chain_structure = longest_chain.return_chain_structure()
            # need more efficient machenism which is to reverse updates by # of blocks
            self.return_blockchain_object().replace_chain(longest_chain_structure)
            print(f"{self.idx} chain resynced from peer {updated_from_peer}.")
            #return block_iter
            return True 
        print("Chain not resynced.")
        return False
    
    def receive_rewards(self, rewards):
        self.rewards += rewards

    def update_model_after_chain_resync(self, log_files_folder_path, conn, conn_cursor):
        # reset global params to the initial weights of the net
        self.global_parameters = copy.deepcopy(self.initial_net_parameters)
        # in future version, develop efficient updating algorithm based on chain difference
        for block in self.return_blockchain_object().return_chain_structure():
            self.process_block(block, log_files_folder_path, conn, conn_cursor, when_resync=True)

    def return_pow_difficulty(self):
        return self.pow_difficulty

    def register_in_the_network(self, check_online=False):
        if self.aio:
            self.add_peers(set(self.devices_dict.values()))
        else:
            potential_registrars = set(self.devices_dict.values())
            # it cannot register with itself
            potential_registrars.discard(self)		
            # pick a registrar
            registrar = random.sample(potential_registrars, 1)[0]
            if check_online:
                if not registrar.is_online():
                    online_registrars = set()
                    for registrar in potential_registrars:
                        if registrar.is_online():
                            online_registrars.add(registrar)
                    if not online_registrars:
                        return False
                    registrar = random.sample(online_registrars, 1)[0]
            # registrant add registrar to its peer list
            self.add_peers(registrar)
            # this device sucks in registrar's peer list
            self.add_peers(registrar.return_peers())
            # registrar adds registrant(must in this order, or registrant will add itself from registrar's peer list)
            registrar.add_peers(self)
            return True
            
    ''' Worker '''	 
    # used to simulate time waste when worker goes offline during transmission to validator
    def waste_one_epoch_local_update_time(self, opti):
        if self.computation_power == 0:
            return float('inf'), None
        else:
            validation_net = copy.deepcopy(self.net)
            currently_used_lr = 0.01
            for param_group in self.opti.param_groups:
                currently_used_lr = param_group['lr']
            # by default use SGD. Did not implement others
            if opti == 'SGD':
                validation_opti = optim.SGD(validation_net.parameters(), lr=currently_used_lr)
            local_update_time = time.time()
            for data, label in self.train_dl:
                data, label = data.to(self.dev), label.to(self.dev)
                preds = validation_net(data)
                loss = self.loss_func(preds, label)
                loss.backward()
                validation_opti.step()
                validation_opti.zero_grad()
            return (time.time() - local_update_time)/self.computation_power, validation_net.state_dict()       
   
    def set_accuracy_this_round(self, accuracy):
        self.accuracy_this_round = accuracy

    def return_accuracy_this_round(self):
        return self.accuracy_this_round

    def return_link_speed(self):
        return self.link_speed

    def receive_block_from_miner(self, received_block, source_miner):
        if not (received_block.return_mined_by() in self.black_list or source_miner in self.black_list):
            self.received_block_from_miner = copy.deepcopy(received_block)
            print(f"{self.role} {self.idx} has received a new block from {source_miner} mined by {received_block.return_mined_by()}.")
        else:
            print(f"Either the block sending miner {source_miner} or the miner {received_block.return_mined_by()} mined this block is in worker {self.idx}'s black list. Block is not accepted.")
            
    def toss_received_block(self):
        self.received_block_from_miner = None

    def return_received_block_from_miner(self):
        return self.received_block_from_miner
       
    ''' miner '''
    def return_miners_eligible_to_continue(self):
        miners_set = set()
        for peer in self.peer_list:
            if peer.return_role() == 'miner':
                miners_set.add(peer)
        miners_set.add(self)
        return miners_set

    def return_accepted_broadcasted_transactions(self):
        return self.broadcasted_transactions

    def proof_of_work(self, candidate_block, starting_nonce=0):
        candidate_block.set_mined_by(self.idx)
        ''' Brute Force the nonce '''
        candidate_block.set_nonce(starting_nonce)
        current_hash = candidate_block.compute_hash()
        # candidate_block.set_pow_difficulty(self.pow_difficulty)
        while not current_hash.startswith('0' * self.pow_difficulty):
            candidate_block.nonce_increment()
            current_hash = candidate_block.compute_hash()
        # return the qualified hash as a PoW proof, to be verified by other devices before adding the block
        # also set its hash as well. block_hash is the same as pow proof
        candidate_block.set_pow_proof(current_hash)
        
        return candidate_block
    
    def return_block_generation_time_point(self):
        return self.block_generation_time_point

    def receive_propagated_block(self, received_propagated_block):
        if not received_propagated_block.return_mined_by() in self.black_list:
            self.received_propagated_block = copy.deepcopy(received_propagated_block)
            print(f"Miner {self.idx} has received a propagated block from {received_propagated_block.return_mined_by()}.")
        else:
            print(f"Propagated block miner {received_propagated_block.return_mined_by()} is in miner {self.idx}'s blacklist. Block not accepted.")

    def receive_propagated_validator_block(self, received_propagated_validator_block):
        if not received_propagated_validator_block.return_mined_by() in self.black_list:
            self.received_propagated_validator_block = copy.deepcopy(received_propagated_validator_block)
            print(f"Miner {self.idx} has received a propagated validator block from {received_propagated_validator_block.return_mined_by()}.")
        else:
            print(f"Propagated validator block miner {received_propagated_validator_block.return_mined_by()} is in miner {self.idx}'s blacklist. Block not accepted.")
    
    def return_propagated_block(self):
        return self.received_propagated_block

    def return_propagated_validator_block(self):
        return self.received_propagated_validator_block
        
    def toss_propagated_block(self):
        self.received_propagated_block = None
        
    def toss_propagated_validator_block(self):
        self.received_propagated_validator_block = None

    def return_online_workers(self):
        online_workers_in_peer_list = set()
        for peer in self.peer_list:
            if peer.is_online():
                if peer.return_role() == "worker":
                    online_workers_in_peer_list.add(peer)
        return online_workers_in_peer_list

    def return_validations_and_signature(self, comm_round):
        validation_transaction_dict = {'validator_device_idx': self.idx, 'round_number': comm_round, 'accuracies_this_round': copy.deepcopy(self.accuracies_this_round), 'validation_effort_rewards': self.validation_rewards_this_round, "rsa_pub_key": self.return_rsa_pub_key()}
        validation_transaction_dict["signature"] = self.sign_msg(sorted(validation_transaction_dict.items()))
        return validation_transaction_dict

    def add_unconfirmmed_transaction(self, unconfirmmed_transaction, souce_device_idx):
        if not souce_device_idx in self.black_list:
            self.unconfirmmed_transactions.append(copy.deepcopy(unconfirmmed_transaction))
            print(f"{souce_device_idx}'s transaction has been recorded by {self.role} {self.idx}")
        else:
            print(f"Source device {souce_device_idx} is in the black list of {self.role} {self.idx}. Transaction has not been recorded.")

    def return_unconfirmmed_transactions(self):
        return self.unconfirmmed_transactions

    def broadcast_transactions(self):
        for peer in self.peer_list:
            if peer.is_online():
                if peer.return_role() == self.role:
                    if not peer.return_idx() in self.black_list:
                        print(f"{self.role} {self.idx} is broadcasting transactions to {peer.return_role()} {peer.return_idx()}.")
                        peer.accept_broadcasted_transactions(self, self.unconfirmmed_transactions)
                    else:
                        print(f"Destination {peer.return_role()} {peer.return_idx()} is in {self.role} {self.idx}'s black_list. broadcasting skipped.")

    def accept_broadcasted_transactions(self, source_device, broadcasted_transactions):
        # discard malicious node
        if not source_device.return_idx() in self.black_list:
            self.broadcasted_transactions.append(copy.deepcopy(broadcasted_transactions))
            print(f"{self.role} {self.idx} has accepted transactions from {source_device.return_role()} {source_device.return_idx()}")
        else:
            print(f"Source {source_device.return_role()} {source_device.return_idx()} is in {self.role} {self.idx}'s black list. Transaction not accepted.")


class DevicesInNetwork(object):
    def __init__(self, data_set_name, is_iid, batch_size, learning_rate, loss_func, opti, num_devices, roles_requirement, network_stability, net, dev, knock_out_rounds, lazy_worker_knock_out_rounds, shard_test_data, miner_acception_wait_time, worker_acception_wait_time, miner_accepted_transactions_size_limit, validate_threshold, pow_difficulty, even_link_speed_strength, base_data_transmission_speed, even_computation_power, malicious_updates_discount, num_malicious, noise_variance, check_signature, not_resync_chain):
        self.data_set_name = data_set_name
        self.is_iid = is_iid
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.loss_func = loss_func
        self.opti = opti
        self.num_devices = num_devices
        self.roles_requirement = roles_requirement
        self.net = net
        self.dev = dev
        self.devices_set = {}
        self.knock_out_rounds = knock_out_rounds
        self.lazy_worker_knock_out_rounds = lazy_worker_knock_out_rounds
        # self.test_data_loader = None
        self.default_network_stability = network_stability
        self.shard_test_data = shard_test_data
        self.even_link_speed_strength = even_link_speed_strength
        self.base_data_transmission_speed = base_data_transmission_speed
        self.even_computation_power = even_computation_power
        self.num_malicious = num_malicious
        self.malicious_updates_discount = malicious_updates_discount
        self.noise_variance = noise_variance
        self.check_signature = check_signature
        self.not_resync_chain = not_resync_chain
        self.worker_acception_wait_time = worker_acception_wait_time
        # distribute dataset
        ''' validate '''
        self.validate_threshold = validate_threshold
        ''' miner '''
        self.miner_acception_wait_time = miner_acception_wait_time
        self.miner_accepted_transactions_size_limit = miner_accepted_transactions_size_limit
        self.pow_difficulty = pow_difficulty
        ''' shard '''
        self.data_set_balanced_allocation()

    # distribute the dataset evenly to the devices
    def data_set_balanced_allocation(self):
        # read dataset
        dataset = DatasetLoad(self.data_set_name, self.is_iid)
        
        # perpare training data
        train_data = dataset.train_data
        train_label = dataset.train_label

        # shard dataset and distribute among devices
        # shard train
        shard_size_train = dataset.train_data_size // self.num_devices // 2
        random.seed(7)
        shards_id_train = np.random.permutation(dataset.train_data_size // shard_size_train) #shuffles the indices of the shards randomly

        # perpare test data
        if not self.shard_test_data:
            test_data = torch.tensor(dataset.test_data)
            test_label = torch.tensor(dataset.test_label)
            if test_label.dim() > 1:
                test_label = torch.argmax(torch.tensor(dataset.test_label), dim=1)
            print(f"Test data shape: {dataset.test_data.shape}")
            print(f"Test labels shape: {dataset.test_label.shape}")
            test_data_loader = DataLoader(TensorDataset(test_data, test_label.long()), batch_size=100, shuffle=False)
        else:
            test_data = dataset.test_data
            test_label = dataset.test_label
            # shard test
            shard_size_test = dataset.test_data_size // self.num_devices // 2  
            random.seed(7)
            shards_id_test = np.random.permutation(dataset.test_data_size // shard_size_test)
        
        # malicious_nodes_set = []
        malicious_workers_set = []
        malicious_miners_set = []
        if self.num_malicious[0]>0:
            random.seed(7)
            malicious_workers_set = random.sample(range(self.roles_requirement[0]), self.num_malicious[0])
        if self.num_malicious[-1]>0:
            random.seed(7)
            malicious_miners_set = random.sample(range(self.roles_requirement[-1]), self.num_malicious[-1])

        for i in range(self.num_devices):
            is_malicious = False
            # make it more random by introducing two shards
            shards_id_train1 = shards_id_train[i * 2]
            shards_id_train2 = shards_id_train[i * 2 + 1]
            # distribute training data
            data_shards1 = train_data[shards_id_train1 * shard_size_train: shards_id_train1 * shard_size_train + shard_size_train]
            data_shards2 = train_data[shards_id_train2 * shard_size_train: shards_id_train2 * shard_size_train + shard_size_train]
            label_shards1 = train_label[shards_id_train1 * shard_size_train: shards_id_train1 * shard_size_train + shard_size_train]
            label_shards2 = train_label[shards_id_train2 * shard_size_train: shards_id_train2 * shard_size_train + shard_size_train]
            local_train_data, local_train_label = np.vstack((data_shards1, data_shards2)), np.concatenate((label_shards1, label_shards2))
            if test_label.dim() > 1:
                local_train_label = np.argmax(local_train_label, axis=1)
            print(f"local_train_data shape: {local_train_data.shape}")
            print(f"local_train_label shape: {local_train_label.shape}")
            # distribute test data
            if self.shard_test_data:
                shards_id_test1 = shards_id_test[i * 2]
                shards_id_test2 = shards_id_test[i * 2 + 1]
                data_shards1 = test_data[shards_id_test1 * shard_size_test: shards_id_test1 * shard_size_test + shard_size_test]
                data_shards2 = test_data[shards_id_test2 * shard_size_test: shards_id_test2 * shard_size_test + shard_size_test]
                label_shards1 = test_label[shards_id_test1 * shard_size_test: shards_id_test1 * shard_size_test + shard_size_test]
                label_shards2 = test_label[shards_id_test2 * shard_size_test: shards_id_test2 * shard_size_test + shard_size_test]
                local_test_data, local_test_label = np.vstack((data_shards1, data_shards2)), np.vstack((label_shards1, label_shards2))
                local_test_label = torch.argmax(torch.tensor(local_test_label), dim=1)
                print(f"Local Test data shape: {dataset.test_data.shape}")
                print(f"Local Test labels shape: {dataset.test_label.shape}")
                test_data_loader = DataLoader(TensorDataset(torch.tensor(local_test_data), torch.tensor(local_test_label, dtype=torch.int64)), batch_size=100, shuffle=False)
            # assign data to a device and put in the devices set
            if i in malicious_workers_set or i in malicious_miners_set:
                is_malicious = True
                # add Gussian Noise

            device_idx = f'device_{i+1}'
            a_device = Device(device_idx, TensorDataset(torch.tensor(local_train_data), torch.tensor(local_train_label, dtype=torch.int64)), test_data_loader, self.batch_size, self.learning_rate, self.loss_func, self.opti, self.default_network_stability, self.net, self.dev, self.miner_acception_wait_time, self.worker_acception_wait_time, self.miner_accepted_transactions_size_limit, self.validate_threshold, self.pow_difficulty, self.even_link_speed_strength, self.base_data_transmission_speed, self.even_computation_power, is_malicious, self.noise_variance, self.check_signature, self.not_resync_chain, self.malicious_updates_discount, self.knock_out_rounds, self.lazy_worker_knock_out_rounds)
            # device index starts from 1
            # # 迭代数据集
            # for data, label in a_device.train_dl:
            #     print("Label dtype before conversion:", label.dtype) #Label dtype before conversion: torch.float64
            #     print("Label shape:", label.shape)#Label shape: torch.Size([10, 10])
            #     label = label.long()
            #     print("Label dtype after conversion:", label.dtype) # Label dtype after conversion: torch.int64
            #     break  # 这里只迭代一次用于调试
            self.devices_set[device_idx] = a_device
            print(f"Sharding dataset to {device_idx} done.")
        print(f"Sharding dataset done!")