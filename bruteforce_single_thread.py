import h5py
import time
import sys
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

def bruteforce_aes():
    db_path = "/home/user/ASCAD/ATMEGA_AES_v1/ATM_AES_v1_fixed_key/ASCAD_data/ASCAD_databases/ATMega8515_raw_traces.h5"
    print(f"Opening dataset: {db_path}")
    
    try:
        with h5py.File(db_path, "r") as f:
            metadata = f['metadata'][0]
            plaintext = bytes(metadata['plaintext'])
            expected_ciphertext = bytes(metadata['ciphertext'])
            original_key = bytes(metadata['key'])
    except Exception as e:
        print(f"Error opening dataset: {e}")
        return
        
    print(f"Plaintext:            {plaintext.hex()}")
    print(f"Expected Ciphertext:  {expected_ciphertext.hex()}")
    print(f"Original Key:         {original_key.hex()}")
    
    # Brute forcing bytes at index 8, 13, 15
    base_key = bytearray(original_key)
    
    real_k8 = original_key[8]
    real_k13 = original_key[13]
    real_k15 = original_key[15]
    print(f"Target bytes to find - 8: {hex(real_k8)}, 13: {hex(real_k13)}, 15: {hex(real_k15)}")
    
    # Set them to 0 for brute force baseline
    base_key[8] = 0
    base_key[13] = 0
    base_key[15] = 0
    
    print("\nStarting cryptographic brute force (AES encryption match)...")
    print("This will test up to 16,777,216 combinations. It may take a few minutes.")
    start_time = time.time()
    
    guessed_key = bytearray(base_key)
    mode = modes.ECB()
    
    r256 = range(256)
    
    total_combinations = 256 * 256 * 256
    combinations_tried = 0
    
    found = False
    
    for k8 in r256:
        guessed_key[8] = k8
        # Print progress report periodically
        sys.stdout.write(f"\rProgress: {combinations_tried / total_combinations * 100:.2f}% (Testing k8={hex(k8)})")
        sys.stdout.flush()
        
        for k13 in r256:
            guessed_key[13] = k13
            
            for k15 in r256:
                guessed_key[15] = k15
                
                # Check combination by doing an AES encryption:
                cipher = Cipher(algorithms.AES(bytes(guessed_key)), mode)
                encryptor = cipher.encryptor()
                encrypted = encryptor.update(plaintext) + encryptor.finalize()
                
                if encrypted == expected_ciphertext:
                    end_time = time.time()
                    print("\n\n[SUCCESS] Key found!")
                    print(f"Recovered Key: {guessed_key.hex()}")
                    print(f"Key byte 8 : {hex(k8)}")
                    print(f"Key byte 13: {hex(k13)}")
                    print(f"Key byte 15: {hex(k15)}")
                    print(f"Time taken : {end_time - start_time:.4f} seconds")
                    found = True
                    break
            
            combinations_tried += 256
            if found:
                break
        if found:
            break

    if not found:
        end_time = time.time()
        print("\n\n[FAILED] Key not found in brute force space.")
        print(f"Time taken: {end_time - start_time:.4f} seconds")

if __name__ == '__main__':
    bruteforce_aes()
