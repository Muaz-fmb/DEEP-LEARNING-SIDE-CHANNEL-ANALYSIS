import h5py
import time
import sys
import multiprocessing
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

def worker_task(args):
    """
    Worker function to check a specific value of k8.
    It loops through all combinations of k13 and k15.
    """
    k8, plaintext, expected_ciphertext, base_key = args
    guessed_key = bytearray(base_key)
    guessed_key[8] = k8
    mode = modes.ECB()
    
    r256 = range(256)
    
    for k13 in r256:
        guessed_key[13] = k13
        for k15 in r256:
            guessed_key[15] = k15
            
            # Setup AES block cipher for this combination
            cipher = Cipher(algorithms.AES(bytes(guessed_key)), mode)
            encryptor = cipher.encryptor()
            encrypted = encryptor.update(plaintext) + encryptor.finalize()
            
            if encrypted == expected_ciphertext:
                return (True, k8, k13, k15, guessed_key)
                
    return (False, k8, None, None, None)

def bruteforce_aes_parallel():
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
        sys.exit(1)
        
    print(f"Plaintext:            {plaintext.hex()}")
    print(f"Expected Ciphertext:  {expected_ciphertext.hex()}")
    print(f"Original Key:         {original_key.hex()}")
    
    base_key = bytearray(original_key)
    
    real_k8 = original_key[8]
    real_k13 = original_key[13]
    real_k15 = original_key[15]
    print(f"Target bytes to find - 8: {hex(real_k8)}, 13: {hex(real_k13)}, 15: {hex(real_k15)}")
    
    # Set them to 0 for brute force baseline
    base_key[8] = 0
    base_key[13] = 0
    base_key[15] = 0
    
    # Use multiprocessing since Python's Global Interpreter Lock (GIL) limits true multithreading for CPU-heavy tasks
    num_cores = multiprocessing.cpu_count()
    print(f"\nStarting parallel brute force using {num_cores} cores...")
    start_time = time.time()
    
    # We split the work into 256 tasks (one for each possible value of k8)
    tasks = [(k8, plaintext, expected_ciphertext, bytes(base_key)) for k8 in range(256)]
    
    found = False
    tasks_completed = 0
    
    with multiprocessing.Pool(processes=num_cores) as pool:
        # imap_unordered allows us to process results as soon as ANY worker finishes a task
        for result in pool.imap_unordered(worker_task, tasks):
            success, k8, k13, k15, guessed_key = result
            tasks_completed += 1
            
            # Print progress block completion
            sys.stdout.write(f"\rProgress: {tasks_completed}/256 k8-blocks checked ({(tasks_completed / 256) * 100:.2f}%)")
            sys.stdout.flush()
            
            if success:
                end_time = time.time()
                print("\n\n[SUCCESS] Key found!")
                print(f"Recovered Key: {guessed_key.hex()}")
                print(f"Key byte 8 : {hex(k8)}")
                print(f"Key byte 13: {hex(k13)}")
                print(f"Key byte 15: {hex(k15)}")
                print(f"Time taken : {end_time - start_time:.4f} seconds")
                found = True
                
                # Terminate remaining processes safely since we found the key
                pool.terminate()
                break

    if not found:
        end_time = time.time()
        print("\n\n[FAILED] Key not found in brute force space.")
        print(f"Time taken: {end_time - start_time:.4f} seconds")

if __name__ == '__main__':
    bruteforce_aes_parallel()
