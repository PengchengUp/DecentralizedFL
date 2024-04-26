# DecentralizedFL by simulated blockchain
## Introduction

## Environments Setup

## Run Simulation
```python
$python main.py -nd 20 -max_ncomm 100 -ha 12,8 -aio 1 -pow 0 -ko 6 -nm 3 -vh 0.08 -cs 0 -B 10 -mn mnist_cnn -iid 0 -lr 0.01 -dtx 1 
```

（1）-nd: number of devices.

（2）-max_ncomm: maximum number of communication rounds.

（3）-ha: hard assign number of roles in the network, order by worker and miner. e.g. 12,8 assign 12 workers and 8 miners. \"*,*\" means completely random role-assigning in each communication round.

(4) -aio: aio means "all in one network", namely, every device in the emulation has every other device in its peer list. This is to simulate that VBFL runs on a permissioned blockchain. If using -aio 0, the emulation will let a device (registrant) randomly register with another device (register) and copy the register's peer list.

(5) -pow: pow means "proof of work", which is used to simulate the mining process. If using -pow 1, the emulation will use a proof of work algorithm to mine blocks. If using -pow 0, the emulation will use a simplified mining process that only requires a random delay between blocks.

(6) -ko: ko means "keep-online", which is used to simulate the network's behavior when a device is offline. If using -ko 1, the emulation will keep a device online by sending heartbeats to its peer devices. If using -ko 0, the emulation will simulate a device being offline by not sending heartbeats to its peer devices.

(7) -nm 3: exactly 3 devices will be malicious nodes.

(8) -vh 0.08: a threshold value of accuracy difference to determine malicious worker.

(9) -cs 0: as the emulation does not include mechanisms to disturb digital signature of the transactions, this argument turns off signature checking to speed up the execution.

(10) -B 10: batch size of the training data.

(11) -mn mnist_cnn: the name of the model to be trained

(12) -iid 0: shard the training data set in Non-IID way.

(13) -lr 0.01: learning rate of the training process.

(14) -dtx 1: currently transactions stored in the blocks are occupying GPU ram and have not figured out a way to move them to CPU ram or harddisk, so turn it on to save GPU ram in order for PoS to run 100+ rounds. NOT GOOD if there needs to perform chain resyncing