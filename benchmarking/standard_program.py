import sys
sys.path.append('/home/heberto/learning/attractor_sequences/')

from network import BCPNNFast, NetworkManager, Protocol
from analysis_functions import calculate_recall_success

def run_standard_program(hypercolumns, minicolumns, epochs):
    # Patterns parameters
    hypercolumns = hypercolumns
    minicolumns = minicolumns

    # Manager properties
    dt = 0.001
    values_to_save = ['o']

    # Protocol
    training_time = 0.1
    inter_sequence_interval = 1.0
    inter_pulse_interval = 0.0
    number_of_patterns = 10
    epochs = epochs

    # Build the network
    nn = BCPNNFast(hypercolumns, minicolumns)

    # Build the manager
    manager = NetworkManager(nn=nn, dt=dt, values_to_save=values_to_save)

    # Build the protocol for training
    protocol = Protocol()
    patterns_indexes = [i for i in range(number_of_patterns)]
    protocol.simple_protocol(patterns_indexes, training_time=training_time, inter_pulse_interval=inter_pulse_interval,
                             inter_sequence_interval=inter_sequence_interval, epochs=epochs)

    # Train
    manager.run_network_protocol(protocol=protocol, verbose=False, values_to_save_epoch=[])

    return manager


def training_program(T_recall, manager):
    print(T_recall)
    manager.run_network_recall(T_recall=T_recall, T_cue=0.100, I_cue=0, reset=True, empty_history=True)


def calculate_succes_program(T_recall, manager):
    patterns_indexes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    calculate_recall_success(manager=manager, T_recall=T_recall, I_cue=0, T_cue=0, n=1,
                             patterns_indexes=patterns_indexes)

if __name__== '__main__':
    hypercolumns = 4
    minicolumns = 10
    epochs = 1
    run_standard_program(hypercolumns, minicolumns, epochs)