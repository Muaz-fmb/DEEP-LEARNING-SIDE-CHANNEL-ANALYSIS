import os
import sys
import time
import h5py
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# The AES SBox that we will use to compute the rank
AES_Sbox = np.array([
        0x63, 0x7C, 0x77, 0x7B, 0xF2, 0x6B, 0x6F, 0xC5, 0x30, 0x01, 0x67, 0x2B, 0xFE, 0xD7, 0xAB, 0x76,
        0xCA, 0x82, 0xC9, 0x7D, 0xFA, 0x59, 0x47, 0xF0, 0xAD, 0xD4, 0xA2, 0xAF, 0x9C, 0xA4, 0x72, 0xC0,
        0xB7, 0xFD, 0x93, 0x26, 0x36, 0x3F, 0xF7, 0xCC, 0x34, 0xA5, 0xE5, 0xF1, 0x71, 0xD8, 0x31, 0x15,
        0x04, 0xC7, 0x23, 0xC3, 0x18, 0x96, 0x05, 0x9A, 0x07, 0x12, 0x80, 0xE2, 0xEB, 0x27, 0xB2, 0x75,
        0x09, 0x83, 0x2C, 0x1A, 0x1B, 0x6E, 0x5A, 0xA0, 0x52, 0x3B, 0xD6, 0xB3, 0x29, 0xE3, 0x2F, 0x84,
        0x53, 0xD1, 0x00, 0xED, 0x20, 0xFC, 0xB1, 0x5B, 0x6A, 0xCB, 0xBE, 0x39, 0x4A, 0x4C, 0x58, 0xCF,
        0xD0, 0xEF, 0xAA, 0xFB, 0x43, 0x4D, 0x33, 0x85, 0x45, 0xF9, 0x02, 0x7F, 0x50, 0x3C, 0x9F, 0xA8,
        0x51, 0xA3, 0x40, 0x8F, 0x92, 0x9D, 0x38, 0xF5, 0xBC, 0xB6, 0xDA, 0x21, 0x10, 0xFF, 0xF3, 0xD2,
        0xCD, 0x0C, 0x13, 0xEC, 0x5F, 0x97, 0x44, 0x17, 0xC4, 0xA7, 0x7E, 0x3D, 0x64, 0x5D, 0x19, 0x73,
        0x60, 0x81, 0x4F, 0xDC, 0x22, 0x2A, 0x90, 0x88, 0x46, 0xEE, 0xB8, 0x14, 0xDE, 0x5E, 0x0B, 0xDB,
        0xE0, 0x32, 0x3A, 0x0A, 0x49, 0x06, 0x24, 0x5C, 0xC2, 0xD3, 0xAC, 0x62, 0x91, 0x95, 0xE4, 0x79,
        0xE7, 0xC8, 0x37, 0x6D, 0x8D, 0xD5, 0x4E, 0xA9, 0x6C, 0x56, 0xF4, 0xEA, 0x65, 0x7A, 0xAE, 0x08,
        0xBA, 0x78, 0x25, 0x2E, 0x1C, 0xA6, 0xB4, 0xC6, 0xE8, 0xDD, 0x74, 0x1F, 0x4B, 0xBD, 0x8B, 0x8A,
        0x70, 0x3E, 0xB5, 0x66, 0x48, 0x03, 0xF6, 0x0E, 0x61, 0x35, 0x57, 0xB9, 0x86, 0xC1, 0x1D, 0x9E,
        0xE1, 0xF8, 0x98, 0x11, 0x69, 0xD9, 0x8E, 0x94, 0x9B, 0x1E, 0x87, 0xE9, 0xCE, 0x55, 0x28, 0xDF,
        0x8C, 0xA1, 0x89, 0x0D, 0xBF, 0xE6, 0x42, 0x68, 0x41, 0x99, 0x2D, 0x0F, 0xB0, 0x54, 0xBB, 0x16
        ])

def check_file_exists(file_path):
    file_path = os.path.normpath(file_path)
    if not os.path.exists(file_path):
        print(f"Error: provided file path '{file_path}' does not exist!")
        sys.exit(-1)

def load_sca_model(model_file):
    check_file_exists(model_file)
    try:
        model = load_model(model_file)
    except Exception as e:
        print(f"Error: can't load Keras model file '{model_file}'. Reason: {e}")
        sys.exit(-1)
    return model

def load_ascad(ascad_database_file):
    check_file_exists(ascad_database_file)
    try:
        in_file = h5py.File(ascad_database_file, "r")
    except Exception as e:
        print(f"Error: can't open HDF5 file '{ascad_database_file}' for reading. Reason: {e}")
        sys.exit(-1)
    
    X_attack = np.array(in_file['Attack_traces/traces'], dtype=np.int8)
    Metadata_attack = in_file['Attack_traces/metadata']
    return X_attack, Metadata_attack

def rank(predictions, metadata, real_key, min_trace_idx, max_trace_idx, last_key_bytes_proba, target_byte):
    if len(last_key_bytes_proba) == 0:
        key_bytes_proba = np.zeros(256)
    else:
        key_bytes_proba = last_key_bytes_proba

    for p in range(0, max_trace_idx - min_trace_idx):
        plaintext = metadata[min_trace_idx + p]['plaintext'][target_byte]
        
        for i in range(0, 256):
            proba = predictions[p][AES_Sbox[plaintext ^ i]]
            if proba != 0:
                key_bytes_proba[i] += np.log(proba)
            else:
                min_proba_predictions = predictions[p][np.array(predictions[p]) != 0]
                if len(min_proba_predictions) == 0:
                    print("Error: got a prediction with only zeroes!")
                    sys.exit(-1)
                min_proba = min(min_proba_predictions)
                key_bytes_proba[i] += np.log(min_proba**2)
                
    sorted_proba = np.array(list(map(lambda a : key_bytes_proba[a], key_bytes_proba.argsort()[::-1])))
    real_key_rank = np.where(sorted_proba == key_bytes_proba[real_key])[0][0]
    return real_key_rank, key_bytes_proba

def full_ranks(predictions, metadata, min_trace_idx, max_trace_idx, rank_step, target_byte):
    real_key = metadata[0]['key'][target_byte]
    
    index = np.arange(min_trace_idx + rank_step, max_trace_idx + rank_step, rank_step)
    f_ranks = np.zeros((len(index), 2), dtype=np.uint32)
    key_bytes_proba = []
    
    for t, i in zip(index, range(0, len(index))):
        real_key_rank, key_bytes_proba = rank(predictions[t-rank_step:t], metadata, real_key, t-rank_step, t, key_bytes_proba, target_byte)
        f_ranks[i] = [t - min_trace_idx, real_key_rank]
    return f_ranks

def main():
    # Byte indices to target (exclude 8, 13, 15)
    target_bytes = [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 14]
    num_traces = 1000
    
    # Dictionaries to hold configurations for each byte
    # NOTE FOR USER: PLEASE UPDATE these paths with the specific models and datasets you want to use!
    model_paths = {
        0: "ATMEGA_AES_v1/ATM_AES_v1_fixed_key/ASCAD_data/ASCAD_trained_models/cnn_best_byte0_win1_desync0_epochs2_batchsize50.h5",
        1: "ATMEGA_AES_v1/ATM_AES_v1_fixed_key/ASCAD_data/ASCAD_trained_models/cnn_best_byte1_win1_desync0_epochs3_batchsize200.h5",
        2: "ATMEGA_AES_v1/ATM_AES_v1_fixed_key/ASCAD_data/ASCAD_trained_models/cnn_best_byte2_win1_desync0_epochs75_batchsize50.h5",
        3: "ATMEGA_AES_v1/ATM_AES_v1_fixed_key/ASCAD_data/ASCAD_trained_models/cnn_best_byte3_win1_desync0_epochs75_batchsize50.h5",
        4: "ATMEGA_AES_v1/ATM_AES_v1_fixed_key/ASCAD_data/ASCAD_trained_models/cnn_best_byte4_win1_desync0_epochs75_batchsize50.h5",
        5: "ATMEGA_AES_v1/ATM_AES_v1_fixed_key/ASCAD_data/ASCAD_trained_models/cnn_best_byte5_win1_desync0_epochs50_batchsize50.h5",
        6: "ATMEGA_AES_v1/ATM_AES_v1_fixed_key/ASCAD_data/ASCAD_trained_models/cnn_best_byte6_win2_desync0_epochs75_batchsize50.h5",
        7: "ATMEGA_AES_v1/ATM_AES_v1_fixed_key/ASCAD_data/ASCAD_trained_models/cnn_best_byte7_win1_desync0_epochs75_batchsize50.h5",
        9: "ATMEGA_AES_v1/ATM_AES_v1_fixed_key/ASCAD_data/ASCAD_trained_models/cnn_best_byte9_win1_desync0_epochs100_batchsize50.h5",
        10: "ATMEGA_AES_v1/ATM_AES_v1_fixed_key/ASCAD_data/ASCAD_trained_models/cnn_best_byte10_win1_desync0_epochs75_batchsize50.h5",
        11: "ATMEGA_AES_v1/ATM_AES_v1_fixed_key/ASCAD_data/ASCAD_trained_models/cnn_best_byte11_win1_desync0_epochs75_batchsize50.h5",
        12: "ATMEGA_AES_v1/ATM_AES_v1_fixed_key/ASCAD_data/ASCAD_trained_models/cnn_best_byte12_win1_desync0_epochs75_batchsize50.h5",
        14: "ATMEGA_AES_v1/ATM_AES_v1_fixed_key/ASCAD_data/ASCAD_trained_models/cnn_best_byte14_win1_desync0_epochs75_batchsize50.h5"
    }
    
    dataset_paths = {
        0: "ATMEGA_AES_v1/ATM_AES_v1_fixed_key/ASCAD_data/ASCAD_databases_byte0_win1/ASCAD_byte0.h5",
        1: "ATMEGA_AES_v1/ATM_AES_v1_fixed_key/ASCAD_data/ASCAD_databases_byte1_win1/ASCAD_byte1.h5",
        2: "ATMEGA_AES_v1/ATM_AES_v1_fixed_key/ASCAD_data/ASCAD_databases_byte2_win1/ASCAD_byte2.h5",
        3: "ATMEGA_AES_v1/ATM_AES_v1_fixed_key/ASCAD_data/ASCAD_databases_byte3_win1/ASCAD_byte3.h5",
        4: "ATMEGA_AES_v1/ATM_AES_v1_fixed_key/ASCAD_data/ASCAD_databases_byte4_win1/ASCAD_byte4.h5",
        5: "ATMEGA_AES_v1/ATM_AES_v1_fixed_key/ASCAD_data/ASCAD_databases_byte5_win1/ASCAD_byte5.h5",
        6: "ATMEGA_AES_v1/ATM_AES_v1_fixed_key/ASCAD_data/ASCAD_databases_byte6_win2/ASCAD_byte6.h5",
        7: "ATMEGA_AES_v1/ATM_AES_v1_fixed_key/ASCAD_data/ASCAD_databases_byte7_win1/ASCAD_byte7.h5",
        9: "ATMEGA_AES_v1/ATM_AES_v1_fixed_key/ASCAD_data/ASCAD_databases_byte9_win1/ASCAD_byte9.h5",
        10: "ATMEGA_AES_v1/ATM_AES_v1_fixed_key/ASCAD_data/ASCAD_databases_byte10_win1/ASCAD_byte10.h5",
        11: "ATMEGA_AES_v1/ATM_AES_v1_fixed_key/ASCAD_data/ASCAD_databases_byte11_win1/ASCAD_byte11.h5",
        12: "ATMEGA_AES_v1/ATM_AES_v1_fixed_key/ASCAD_data/ASCAD_databases_byte12_win1/ASCAD_byte12.h5",
        14: "ATMEGA_AES_v1/ATM_AES_v1_fixed_key/ASCAD_data/ASCAD_databases_byte14_win1/ASCAD_byte14.h5"
    }
    
    print(f"Starting prediction process for {len(target_bytes)} key bytes: {target_bytes}")
    print(f"Using first {num_traces} traces.\n")
    
    overall_start_time = time.time()

    results = {}

    for b in target_bytes:
        print(f"=====================================")
        print(f"Processing Key Byte {b}")
        print(f"=====================================")
        
        m_path = model_paths[b]
        d_path = dataset_paths[b]
        
        # Check if user updated paths exist, otherwise we skip gracefully or warning
        if not os.path.exists(m_path):
            print(f"[WARNING] Model path {m_path} not found. Please specify the exact model path in the script. Skipping byte {b}.")
            results[b] = "Model File Missing"
            continue
            
        if not os.path.exists(d_path):
            print(f"[WARNING] Dataset path {d_path} not found. Skipping byte {b}.")
            results[b] = "Dataset File Missing"
            continue
        
        # 1. Load Traces
        print(f"Loading dataset from {d_path}...")
        X_attack, Metadata_attack = load_ascad(d_path)
        
        # 2. Load Model
        print(f"Loading Keras model from {m_path}...")
        model = load_sca_model(m_path)
        
        # Shape data appropriately
        input_layer_shape = model.input_shape
        if len(input_layer_shape) == 2:
            input_data = X_attack[:num_traces, :]
        elif len(input_layer_shape) == 3:
            input_data = X_attack[:num_traces, :]
            input_data = input_data.reshape((input_data.shape[0], input_data.shape[1], 1))
        
        # 3. Predict simultaneously for the batch of 1000 traces
        print(f"Predicting on {num_traces} traces...")
        predictions = model.predict(input_data)
        
        # 4. Compute rank traces requirement (step = 1 to find exact trace count)
        print("Computing ranks to find traces needed for rank 0...")
        ranks = full_ranks(predictions, Metadata_attack, 0, num_traces, 1, b)
        
        traces_needed = -1
        # Find the first trace count where rank reaches 0
        for i in range(ranks.shape[0]):
            r = ranks[i][1]
            if r == 0:
                traces_needed = ranks[i][0]
                break
                
        if traces_needed != -1:
            print(f"-> Key byte {b} found! Traces required: {traces_needed}")
            results[b] = f"{traces_needed} traces"
        else:
            final_rank = ranks[-1][1]
            print(f"-> Key byte {b} NOT found within {num_traces} traces. Final rank: {final_rank}")
            results[b] = f"Not found (> {num_traces} traces). Final rank: {final_rank}"
        
        print()

    print("\n=====================================")
    print("         SUMMARY OF RESULTS          ")
    print("=====================================")
    for b in target_bytes:
        res = results.get(b, "Skipped")
        print(f"Byte {b:>2}: {res}")
        
    overall_end_time = time.time()
    print(f"\nTotal Prediction and Evaluation Time: {overall_end_time - overall_start_time:.2f} seconds")

if __name__ == "__main__":
    main()
