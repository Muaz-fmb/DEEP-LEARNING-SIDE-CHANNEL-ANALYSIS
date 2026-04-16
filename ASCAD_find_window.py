print("Starting script...", flush=True)
import h5py
import numpy as np
import matplotlib.pyplot as plt
import sys

# AES SBox
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

# Hamming Weight table
HW = [bin(x).count("1") for x in range(256)]

def check_file_exists(file_path):
    import os
    if not os.path.exists(file_path):
        print("Error: file '%s' does not exist!" % file_path)
        sys.exit(-1)

def perform_cpa(traces_file, target_byte):
    print(f"Loading traces from: {traces_file}")
    check_file_exists(traces_file)
    
    in_file = h5py.File(traces_file, "r")
    
    # Load a subset of traces for analysis (e.g., 5000 is enough for unmasked/simple masked CPA)
    # The file has 60k traces.
    num_traces = 2000
    
    # Load traces
    # Note: Need to check the dataset name in the file. 
    # Based on other scripts it is 'traces'
    print(f"Loading first {num_traces} traces...", flush=True)
    traces = in_file['traces'][:num_traces]
    print("Traces loaded.", flush=True)
    
    # Load metadata
    metadata = in_file['metadata']
    print("Loading metadata...", flush=True)
    # Read the structured array first, then access fields
    # metadata is a dataset of compound type.
    meta_chunk = metadata[:num_traces]
    plaintexts = meta_chunk['plaintext']
    keys = meta_chunk['key']
    print("Metadata loaded.", flush=True)
    
    print(f"Traces shape: {traces.shape}")
    
    # Calculate Intermediate Value: SBox[ p[i] ^ k[i] ]
    # We target the specific byte index
    print(f"Calculating intermediate values for Byte {target_byte} (Index {target_byte})...")
    
    # Make sure we use the correct index
    p = plaintexts[:, target_byte]
    k = keys[:, target_byte]
    
    intermediate = AES_Sbox[p ^ k]
    
    # Calculate Hypothetical Power (Hamming Weight)
    # This assumes the device leaks the Hamming Weight of the SBox output
    # (or the transition from previous state, but HW is a good first guess for 8-bit)
    hypothetical_power = np.array([HW[val] for val in intermediate])
    
    print("Calculating correlation...")
    
    # Calculate Correlation: Pearson Correlation Coefficient
    # Correlate `hypothetical_power` column vector with `traces` matrix
    
    # Centering
    input_centered = traces - np.mean(traces, axis=0)
    target_centered = hypothetical_power - np.mean(hypothetical_power)
    
    # Denominator
    input_std = np.std(traces, axis=0)
    target_std = np.std(hypothetical_power)
    
    # Avoid division by zero
    input_std[input_std == 0] = 1
    
    # Covariance / (std_x * std_y)
    numerator = np.dot(target_centered, input_centered) / num_traces
    correlation = numerator / (input_std * target_std)
    
    # Find peak
    peak_idx = np.argmax(np.abs(correlation))
    peak_val = correlation[peak_idx]
    
    print(f"\n   Analysis Results for Byte {target_byte}    ")
    print(f"Max Absolute Correlation: {np.abs(peak_val):.4f}")
    print(f"Peak Location (Time Sample): {peak_idx}")
    
    # Define a window around the peak
    # The original script used a window of 700 points.
    # Let's suggest a window centered on the peak.
    window_start = max(0, peak_idx - 350)
    window_end = peak_idx + 350
    print(f"Suggested Window (700 points centered): [{window_start}, {window_end}]")
    
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(np.abs(correlation))
    plt.title(f"CPA Results - Target Byte {target_byte}")
    plt.xlabel("Time Sample")
    plt.ylabel("Absolute Correlation")
    plt.axvline(x=peak_idx, color='r', linestyle='--', label=f'Peak @ {peak_idx}')
    plt.legend()
    plt.savefig(f"cpa_byte_{target_byte}.png")
    print(f"Plot saved to cpa_byte_{target_byte}.png")
    
    in_file.close()

if __name__ == "__main__":
    # Path to raw traces
    traces_file = "ATMEGA_AES_v1/ATM_AES_v1_fixed_key/ASCAD_data/ASCAD_databases/ATMega8515_raw_traces.h5"
    
    # Target Byte 2 (Index 1)
    target_byte = 15
    
    perform_cpa(traces_file, target_byte)
