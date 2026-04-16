# DEEP-LEARNING-SIDE-CHANNEL-ANALYSIS
An open-source, hybrid Deep Learning Side-Channel Analysis (DL-SCA) and multi-threaded brute-force framework for full AES-128 key recovery on the ASCAD dataset.

## Project Files Description

This repository contains several scripts for analyzing, generating, training, and performing attacks on the AES-128 implementation from the ASCAD dataset.

### Analysis and Pre-processing
- **[plot_snr.py](file:///Users/muazalbaghdadi/DEEP-LEARNING-SIDE-CHANNEL-ANALYSIS/plot_snr.py)**: Used to plot all SNR (Signal-to-Noise Ratio) traces on a single chart. Before running, specify the target key byte by setting the `target_byte` value in the `main` function.
- **[plot_snr_separet.py](file:///Users/muazalbaghdadi/DEEP-LEARNING-SIDE-CHANNEL-ANALYSIS/plot_snr_separet.py)**: Similar to `plot_snr.py`, but generates a separate plot for each SNR function. Specify the target byte by setting the `target_byte` parameter in the `main` function definition.
- **[ASCAD_find_window.py](file:///Users/muazalbaghdadi/DEEP-LEARNING-SIDE-CHANNEL-ANALYSIS/ASCAD_find_window.py)**: Based on the peak values identified from SNR analysis, this script highlights the necessary window around the peak for visual inspection.

### Dataset Generation
- **[ASCAD_generate.py](file:///Users/muazalbaghdadi/DEEP-LEARNING-SIDE-CHANNEL-ANALYSIS/ASCAD_generate.py)**: Originating from the official ASCAD repository, this script generates a dataset from raw traces dedicated to the 3rd byte, based on windows specified from SNR peaks.
- **[ASCAD_generate_byte.py](file:///Users/muazalbaghdadi/DEEP-LEARNING-SIDE-CHANNEL-ANALYSIS/ASCAD_generate_byte.py)**: Used to generate separate datasets for any targeted key byte.
    - Update the index in the `labelize()` function.
    - Update `target_points()` indices to specify the start and end points from the original traces.
    - Reads raw traces from: `ATMEGA_AES_v1/ATM_AES_v1_fixed_key/ASCAD_data/ASCAD_databases/ATMega8515_raw_traces.h5`.
    - Generates datasets under: `ATMEGA_AES_v1/ATM_AES_v1_fixed_key/ASCAD_data/ASCAD_databases/ASCAD_databases_byte%d/`.
    - Produces three versions: zero desynchronization, 50, and 100 desynchronization.

### Training and Evaluation
- **[ASCAD_train_models.py](file:///Users/muazalbaghdadi/DEEP-LEARNING-SIDE-CHANNEL-ANALYSIS/ASCAD_train_models.py)**: The original code for training models.
- **[ASCAD_train_models_byte.py](file:///Users/muazalbaghdadi/DEEP-LEARNING-SIDE-CHANNEL-ANALYSIS/ASCAD_train_models_byte.py)**: An updated version allowing the specification of `target_byte` before training. Use this to train models for different key bytes and tune hyperparameters.
- **[ASCAD_test_models.py](file:///Users/muazalbaghdadi/DEEP-LEARNING-SIDE-CHANNEL-ANALYSIS/ASCAD_test_models.py)**: The original testing script.
- **[ASCAD_test_models_byte.py](file:///Users/muazalbaghdadi/DEEP-LEARNING-SIDE-CHANNEL-ANALYSIS/ASCAD_test_models_byte.py)**: Updated version where you can specify `target_byte`, `model_file`, and `ascad_database` when calling the `main` function.
- **[ASCAD_test_models_byte_slicing_index.py](file:///Users/muazalbaghdadi/DEEP-LEARNING-SIDE-CHANNEL-ANALYSIS/ASCAD_test_models_byte_slicing_index.py)**: Performs validity tests, enabling testing on different trace sets to ensure model performance consistency.

### Key Recovery Attacks
- **[bruteforce_single_thread.py](file:///Users/muazalbaghdadi/DEEP-LEARNING-SIDE-CHANNEL-ANALYSIS/bruteforce_single_thread.py)**: Performs sequential brute force on the remaining 3 key bytes (indices 8, 13, and 15) using predicted values as a cryptographic anchor.
- **[bruteforce_multi_thread.py](file:///Users/muazalbaghdadi/DEEP-LEARNING-SIDE-CHANNEL-ANALYSIS/bruteforce_multi_thread.py)**: A faster version of the brute-force script utilizing multiple CPU cores.
- **[attack_full_key.py](file:///Users/muazalbaghdadi/DEEP-LEARNING-SIDE-CHANNEL-ANALYSIS/attack_full_key.py)**: Performs the full attack. It loads pre-trained models for high-confidence key bytes, predicts their values, and then uses those values as a cryptographic anchor for multi-threaded brute-forcing of the remaining bytes.

## Usage Workflow

1. **Load Dataset**: Ensure the dataset is correctly placed in the expected paths.
2. **Generate Datasets**: Use SNR analysis to identify windows and generate dedicated datasets using `ASCAD_generate_byte.py`.
3. **Train Models**: Train models for the targeted bytes as per settings reported in the thesis.
4. **Full Attack**: Run `attack_full_key.py` to perform the complete key recovery.
