import h5py
import numpy as np
import matplotlib.pyplot as plt

# AES S-box
SBOX = np.array([
    0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,
    0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0,
    0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15,
    0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75,
    0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84,
    0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf,
    0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8,
    0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2,
    0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,
    0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb,
    0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79,
    0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08,
    0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a,
    0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e,
    0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,
    0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16
], dtype=np.uint8)

def calculate_snr(traces, labels):
    """Calculates SNR: Variance of the Means / Mean of the Variances."""
    num_samples = traces.shape[1]
    means = np.zeros((256, num_samples))
    variances = np.zeros((256, num_samples))
    for z in range(256):
        indices = np.where(labels == z)
        if len(indices[0]) > 0:
            subset = traces[indices]
            means[z] = np.mean(subset, axis=0)
            variances[z] = np.var(subset, axis=0)
    
    # Avoid division by zero warnings
    mean_of_variances = np.mean(variances, axis=0)
    # If mean_of_variances is 0, replace with 1 (or small epsilon) to avoid inf/nan, 
    # though with real traces noise is expected.
    # For now, let's just do standard division as in original script, maybe adding a small check if needed.
    return np.var(means, axis=0) / (mean_of_variances + 1e-20)

def main(filename, target_byte=9, num_traces=10000):
    print(f"Opening file: {filename}")
    try:
        with h5py.File(filename, 'r') as f:
            print(f"Loading {num_traces} traces...")
            traces = f['traces'][:num_traces].astype(np.float32)
            metadata = f['metadata'][:num_traces]
            p = metadata['plaintext'][:, target_byte]
            k = metadata['key'][:, target_byte]
            masks = metadata['masks']
            
            # rout is at the last index of the mask array (index 15)
            # rout = masks[:, 15]        # ***** uncomment this when caculating for bytes other than 0 and 1
            rout = masks[:, 15].astype(np.uint8)
            # Byte-specific mask (r_i). For target_byte=2 (3rd byte), r_3 is masks[:, 0]

            # r_i = masks[:, target_byte - 2] if target_byte >= 2 else np.zeros(num_traces)  # ***** uncomment this when caculating for bytes other than 0 and 1
            if target_byte >= 2:
                r_i = masks[:, target_byte - 2].astype(np.uint8)
            else:
                # Initialize as zeros with the correct integer type
                r_i = np.zeros(num_traces, dtype=np.uint8)

            
    except FileNotFoundError:
        print(f"Error: File not found at {filename}")
        return
    except Exception as e:
        print(f"Error loading file: {e}")
        return

    sbox_out = SBOX[p ^ k]

    # Defining the 5 targets for SNR computation
    targets = {
        # "snr1": (sbox_out, 'gray', "Sbox(Pi ^ Ki)"),
        "snr2": (sbox_out ^ rout, 'blue', "Sbox(Pi ^ Ki) ^ rout"),
        # "snr3": (rout, 'green', "rout"),
        # "snr4": (sbox_out ^ r_i, 'brown', "Sbox(Pi ^ Ki) ^ r_i"),
        # "snr5": (r_i, 'yellow', "r_i")
    }

    # Setup subplot figure: 5 rows, 1 column
    fig, axes = plt.subplots(len(targets), 1, figsize=(20, 4 * len(targets)), sharex=True)
    if len(targets) == 1:
        axes = [axes]
    
    print("\n   Peak SNR Locations    ")
    
    for i, (name, (label_data, color, label_text)) in enumerate(targets.items()):
        print(f"Computing {name}...")
        snr_vals = calculate_snr(traces, label_data)
        
        # Find peak
        peak_idx = np.argmax(snr_vals)
        peak_val = snr_vals[peak_idx]
        print(f"{name} ({label_text}) Peak: {peak_val:.4f} at index {peak_idx}")
        
        # Plot
        ax = axes[i]
        ax.plot(snr_vals, color=color, label=f"{name}: {label_text}", linewidth=1.0)
        
        if name == "snr2":
            highlight_start = max(0, peak_idx - 150)
            highlight_end = min(len(snr_vals), peak_idx + 150)
            ax.axvspan(highlight_start, highlight_end, color='red', alpha=0.3, label='Peak Region (±150)')
        
        # Mark peak
        ax.plot(peak_idx, peak_val, 'ro')
        ax.annotate(f"{peak_val:.2f} @ {peak_idx}", 
                    xy=(peak_idx, peak_val), 
                    xytext=(peak_idx, peak_val + (peak_val*0.1)),
                    arrowprops=dict(facecolor='black', shrink=0.05),
                    horizontalalignment='center')
        
        ax.set_ylabel("SNR Value")
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_title(f"{name}: {label_text}")

    axes[-1].set_xlabel("Time Samples")
    plt.suptitle(f"SNR2 Metric for Byte Index {target_byte}", fontsize=16)
    # plt.tight_layout(rect=[0, 0.03, 1, 0.98]) # Adjust for suptitle
    plt.tight_layout(rect=[0, 0.03, 1, 0.98], h_pad=3.0)
    
    output_filename = f"snr2_byte_{target_byte}_150.png"
    plt.savefig(output_filename)
    print(f"\nPlot saved to {output_filename}")

if __name__ == "__main__":
    main("ATMEGA_AES_v1/ATM_AES_v1_fixed_key/ASCAD_data/ASCAD_databases/ATMega8515_raw_traces.h5")
