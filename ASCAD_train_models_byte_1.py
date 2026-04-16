import os
import os.path
import sys
import h5py
import numpy as np
import time

import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Flatten, Dense, Input, Conv1D, MaxPooling1D, GlobalAveragePooling1D, GlobalMaxPooling1D, AveragePooling1D, BatchNormalization, Activation, Add, add
from tensorflow.keras import backend as K
from tensorflow.keras.applications.imagenet_utils import decode_predictions
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model

# The AES SBox that we will use to generate our labels
# Copied from ASCAD_generate.py
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

def labelize(plaintexts, keys, byte_index):
    return np.uint8(AES_Sbox[plaintexts[:, byte_index] ^ keys[:, byte_index]])

def check_file_exists(file_path):
    file_path = os.path.normpath(file_path)
    if os.path.exists(file_path) == False:
        print("Error: provided file path '%s' does not exist!" % file_path)
        sys.exit(-1)
    return

#### MLP Best model (6 layers of 200 units)
def mlp_best(node=200,layer_nb=6,input_dim=1400):
    model = Sequential()
    model.add(Dense(node, input_dim=input_dim, activation='relu'))
    for i in range(layer_nb-2):
        model.add(Dense(node, activation='relu'))
    model.add(Dense(256, activation='softmax'))
    optimizer = RMSprop(learning_rate=0.00001)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

### CNN Best model (update the input_dim based on the size of the generated window)
def cnn_best(classes=256,input_dim=500):
    # From VGG16 design
    input_shape = (input_dim,1)
    img_input = Input(shape=input_shape)
    # Block 1
    x = Conv1D(64, 11, activation='relu', padding='same', name='block1_conv1')(img_input)
    x = AveragePooling1D(2, strides=2, name='block1_pool')(x)
    # Block 2
    x = Conv1D(128, 11, activation='relu', padding='same', name='block2_conv1')(x)
    x = AveragePooling1D(2, strides=2, name='block2_pool')(x)
    # Block 3
    x = Conv1D(256, 11, activation='relu', padding='same', name='block3_conv1')(x)
    x = AveragePooling1D(2, strides=2, name='block3_pool')(x)
    # Block 4
    x = Conv1D(512, 11, activation='relu', padding='same', name='block4_conv1')(x)
    x = AveragePooling1D(2, strides=2, name='block4_pool')(x)
    # Block 5
    x = Conv1D(512, 11, activation='relu', padding='same', name='block5_conv1')(x)
    x = AveragePooling1D(2, strides=2, name='block5_pool')(x)
    # Classification block
    x = Flatten(name='flatten')(x)
    x = Dense(4096, activation='relu', name='fc1')(x)
    x = Dense(4096, activation='relu', name='fc2')(x)
    x = Dense(classes, activation='softmax', name='predictions')(x)

    inputs = img_input 
    # Create model.
    model = Model(inputs, x, name='cnn_best')
    optimizer = RMSprop(learning_rate=0.00001)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

### CNN Best model
def cnn_best2(classes=256,input_dim=1400):
    # From VGG16 design
    input_shape = (input_dim,1)
    img_input = Input(shape=input_shape)
    # Block 1
    x = Conv1D(64, 11, strides=2, activation='relu', padding='same', name='block1_conv1')(img_input)
    x = AveragePooling1D(2, strides=2, name='block1_pool')(x)
    # Block 2
    x = Conv1D(128, 11, activation='relu', padding='same', name='block2_conv1')(x)
    x = AveragePooling1D(2, strides=2, name='block2_pool')(x)
    # Block 3
    x = Conv1D(256, 11, activation='relu', padding='same', name='block3_conv1')(x)
    x = AveragePooling1D(2, strides=2, name='block3_pool')(x)
    # Block 4
    x = Conv1D(512, 11, activation='relu', padding='same', name='block4_conv1')(x)
    x = AveragePooling1D(2, strides=2, name='block4_pool')(x)
    # Block 5
    x = Conv1D(512, 11, activation='relu', padding='same', name='block5_conv1')(x)
    x = AveragePooling1D(2, strides=2, name='block5_pool')(x)
    # Classification block
    x = Flatten(name='flatten')(x)
    x = Dense(4096, activation='relu', name='fc1')(x)
    x = Dense(4096, activation='relu', name='fc2')(x)
    x = Dense(classes, activation='softmax', name='predictions')(x)

    inputs = img_input
    # Create model.
    model = Model(inputs, x, name='cnn_best2')
    optimizer = RMSprop(learning_rate=0.00001)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


### Resnet layer sub-function of ResNetSCA
def resnet_layer(inputs,
                 num_filters=16,
                 kernel_size=11,
                 strides=1,
                 activation='relu',
                 batch_normalization=True,
                 conv_first=True):
    conv = Conv1D(num_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  kernel_initializer='he_normal')

    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)
    return x

### Branch of ResNetSCA that predict the multiplicative mask alpha
def alpha_branch(x):
    x = Dense(1024, activation='relu', name='fc1_alpha')(x)
    x = BatchNormalization()(x)
    x = Dense(256, activation="softmax", name='alpha_output')(x)
    return x

### Branch of ResNetSCA that predict the additive mask beta
def beta_branch(x):
    x = Dense(1024, activation='relu', name='fc1_beta')(x)
    x = BatchNormalization()(x)
    x = Dense(256, activation="softmax", name='beta_output')(x)
    return x

### Branch of ResNetSCA that predict the masked sbox output
def sbox_branch(x,i):
    x = Dense(1024, activation='relu', name='fc1_sbox_'+str(i))(x)
    x = BatchNormalization()(x)
    x = Dense(256, activation="softmax", name='sbox_'+str(i)+'_output')(x)
    return x

### Branch of ResNetSCA that predict the pemutation indices
def permind_branch(x,i):
    x = Dense(1024, activation='relu', name='fc1_pemind_'+str(i))(x)
    x = BatchNormalization()(x)
    x = Dense(16, activation="softmax", name='permind_'+str(i)+'_output')(x)
    return x

### Generic function that produce the ResNetSCA architecture.
### If without_permind option is set to 1, the ResNetSCA model is built without permindices branch
def resnet_v1(input_shape, depth, num_classes=256, without_permind=0):
    if (depth - 1) % 18 != 0:
        raise ValueError('depth should be 18n+1 (eg 19, 37, 55 ...)')
    # Start model definition.
    num_filters = 16
    num_res_blocks = int((depth - 1) / 18)
    inputs = Input(shape=input_shape)
    x = resnet_layer(inputs=inputs)
    # Instantiate the stack of residual units
    for stack in range(9):
        for res_block in range(num_res_blocks):
            strides = 1
            if stack > 0 and res_block == 0:
                strides = 2
            y = resnet_layer(inputs=x,
                             num_filters=num_filters,
                             strides=strides)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters,
                             activation=None)
            if stack > 0 and res_block == 0:
                x = resnet_layer(inputs=x,
                                 num_filters=num_filters,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False)
            x = add([x, y])
            x = Activation('relu')(x)
        if (num_filters<256):
            num_filters *= 2
    x = AveragePooling1D(pool_size=4)(x)
    x = Flatten()(x)
    x_alpha = alpha_branch(x)
    x_beta = beta_branch(x)
    x_sbox_l = []
    x_permind_l = []
    for i in range(16):
        x_sbox_l.append(sbox_branch(x,i))
        x_permind_l.append(permind_branch(x,i))
    if without_permind!=1:
      model = Model(inputs, [x_alpha, x_beta] + x_sbox_l + x_permind_l, name='extract_resnet')
    else:
      model = Model(inputs, [x_alpha, x_beta] + x_sbox_l, name='extract_resnet_without_permind')
    optimizer = Adam()
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

### CNN multilabel test function. This model is only used for debugging.
def multi_test(input_dim=1400):
    input_shape = (input_dim,1)
    inputs = Input(shape=input_shape)
    # Block 1
    x = Conv1D(3, 11, strides=100, activation='relu', padding='same', name='block1_conv1')(inputs)
    x = Flatten()(x)
    x_alpha = alpha_branch(x)
    x_beta = beta_branch(x)
    x_sbox_l = []
    x_permind_l = []
    for i in range(16):
        x_sbox_l.append(sbox_branch(x,i))
        x_permind_l.append(permind_branch(x,i))
    model = Model(inputs, [x_alpha, x_beta] + x_sbox_l + x_permind_l, name='test_multi')
    optimizer = Adam()
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

def load_sca_model(model_file):
    check_file_exists(model_file)
    try:
            model = load_model(model_file)
    except:
        print("Error: can't load Keras model file '%s'" % model_file)
        sys.exit(-1)
    return model


#### ASCAD helper to load profiling and attack data (traces and labels)
# Loads the profiling and attack datasets from the ASCAD
# database
def load_ascad(ascad_database_file, load_metadata=False, target_byte=None):
    check_file_exists(ascad_database_file)
    # Open the ASCAD database HDF5 for reading
    try:
        in_file  = h5py.File(ascad_database_file, "r")
    except:
        print("Error: can't open HDF5 file '%s' for reading (it might be malformed) ..." % ascad_database_file)
        sys.exit(-1)
    # Load profiling traces
    X_profiling = np.array(in_file['Profiling_traces/traces'], dtype=np.int8)

    # Load labels
    if target_byte is not None:
        print("Computing labels for target byte %d from metadata..." % target_byte)
        
        # Load profiling metadata
        profiling_metadata = in_file['Profiling_traces/metadata']
        # Extract plaintext and key
        # Depending on h5py version and file structure, we might need to be careful. 
        # But assuming the structure seen in generate script:
        plaintexts_profiling = profiling_metadata['plaintext']
        keys_profiling = profiling_metadata['key']
        Y_profiling = labelize(plaintexts_profiling, keys_profiling, target_byte)

        # Load attacking traces
        X_attack = np.array(in_file['Attack_traces/traces'], dtype=np.int8)
        
        # Load attacking metadata
        attack_metadata = in_file['Attack_traces/metadata']
        plaintexts_attack = attack_metadata['plaintext']
        keys_attack = attack_metadata['key']
        Y_attack = labelize(plaintexts_attack, keys_attack, target_byte)
        
    else:
        # Load profiling labels from file (default byte 2)
        Y_profiling = np.array(in_file['Profiling_traces/labels'])
        # Load attacking traces
        X_attack = np.array(in_file['Attack_traces/traces'], dtype=np.int8)
        # Load attacking labels
        Y_attack = np.array(in_file['Attack_traces/labels'])

    if load_metadata == False:
        return (X_profiling, Y_profiling), (X_attack, Y_attack)
    else:
        return (X_profiling, Y_profiling), (X_attack, Y_attack), (in_file['Profiling_traces/metadata'], in_file['Attack_traces/metadata'])

def multilabel_to_categorical(Y):
    y = {}
    y['alpha_output'] = to_categorical(Y['alpha_mask'], num_classes=256)
    y['beta_output'] = to_categorical(Y['beta_mask'], num_classes=256)
    for i in range(16):
        y['sbox_'+str(i)+'_output'] = to_categorical(Y['sbox_masked'][:,i], num_classes=256)
    for i in range(16):
        y['permind_'+str(i)+'_output'] = to_categorical(Y['perm_index'][:,i], num_classes=16)
    return y

def multilabel_without_permind_to_categorical(Y):
    y = {}
    y['alpha_output'] = to_categorical(Y['alpha_mask'], num_classes=256)
    y['beta_output'] = to_categorical(Y['beta_mask'], num_classes=256)
    for i in range(16):
        y['sbox_'+str(i)+'_output'] = to_categorical(Y['sbox_masked_with_perm'][:,i], num_classes=256)
    return y

#### Training high level function
def train_model(X_profiling, Y_profiling, model, save_file_name, epochs=150, batch_size=100, multilabel=0, validation_split=0, early_stopping=0):
    check_file_exists(os.path.dirname(save_file_name))
    # Save model calllback
    save_model = ModelCheckpoint(save_file_name)
    callbacks=[save_model]
    # Early stopping callback
    if (early_stopping != 0):
        if validation_split == 0:
            validation_split=0.1
        callbacks.append(EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True))
    
    # Get the input layer shape
    input_layer_shape = model.input_shape
    # Sanity check
    if input_layer_shape[1] != len(X_profiling[0]):
        print("Error: model input shape %d instead of %d is not expected ..." % (input_layer_shape[1], len(X_profiling[0])))
        sys.exit(-1)
    # Adapt the data shape according our model input
    if len(input_layer_shape) == 2:
        # This is a MLP
        Reshaped_X_profiling = X_profiling
    elif len(input_layer_shape) == 3:
        # This is a CNN: expand the dimensions
        Reshaped_X_profiling = X_profiling.reshape((X_profiling.shape[0], X_profiling.shape[1], 1))
    else:
        print("Error: model input shape length %d is not expected ..." % len(input_layer_shape))
        sys.exit(-1)
    if (multilabel==1):
        y=multilabel_to_categorical(Y_profiling)
    elif (multilabel==2):
        y=multilabel_without_permind_to_categorical(Y_profiling)
    else:
        y=to_categorical(Y_profiling, num_classes=256)
    
    # --- TIMER START ---
    print("\nTraining started...")
    start_time = time.time()

    history = model.fit(x=Reshaped_X_profiling, y=y, batch_size=batch_size, verbose = 1, validation_split=validation_split, epochs=epochs, callbacks=callbacks)

    end_time = time.time()
    elapsed_time_minutes = (end_time - start_time) / 60
    print("-" * 30)
    print(f"Training completed in: {elapsed_time_minutes:.2f} minutes")
    print("-" * 30)

    return history

def read_parameters_from_file(param_filename):
    #read parameters for the train_model and load_ascad functions
    #TODO: sanity checks on parameters
    param_file = open(param_filename,"r")

    #TODO: replace eval() by ast.linear_eval()
    my_parameters= eval(param_file.read())

    ascad_database = my_parameters["ascad_database"]
    training_model = my_parameters["training_model"]
    network_type = my_parameters["network_type"]
    epochs = my_parameters["epochs"]
    batch_size = my_parameters["batch_size"]
    train_len = 0
    if ("train_len" in my_parameters):
        train_len = my_parameters["train_len"]
    validation_split = 0
    if ("validation_split" in my_parameters):
        validation_split = my_parameters["validation_split"]
    multilabel = 0
    if ("multilabel" in my_parameters):
        multilabel = my_parameters["multilabel"]
    early_stopping = 0
    if ("early_stopping" in my_parameters):
        early_stopping = my_parameters["early_stopping"]
    return ascad_database, training_model, network_type, epochs, batch_size, train_len, validation_split, multilabel, early_stopping


if __name__ == "__main__":
    
    # Target byte: 1 (The second byte)
    target_byte = 9

    if len(sys.argv)!=2:
        #default parameters values
        # Updated to use the newly generated dataset for Byte 1
        ascad_database = "ATMEGA_AES_v1/ATM_AES_v1_fixed_key/ASCAD_data/ASCAD_databases_byte%d_win1_4/ASCAD_byte%d.h5" % (target_byte, target_byte)
        
        #MLP training
        # network_type = "mlp"
        # training_model = "ATMEGA_AES_v1/ATM_AES_v1_fixed_key/ASCAD_data/ASCAD_trained_models/my_mlp_best_desync0_epochs75_batchsize200.h5"

        #CNN training
        network_type = "cnn"
        # Updated training model name to reflect byte 1 target
        training_model = "ATMEGA_AES_v1/ATM_AES_v1_fixed_key/ASCAD_data/ASCAD_trained_models/cnn_best_byte%d_win1_4_desync0_epochs100_batchsize50.h5" % target_byte

        #CNN training
        #network_type = "cnn2"
        #training_model = "ATMEGA_AES_v1/ATM_AES_v1_fixed_key/ASCAD_data/ASCAD_trained_models/my_cnn_best_desync0_epochs200_batchsize50.h5"
        validation_split = 0
        multilabel = 0
        train_len = 0
        # epochs = 75
        epochs = 100
        # batch_size = 200
        batch_size = 50
        bugfix = 0
        early_stopping = 0
    else:
        #get parameters from user input
        ascad_database, training_model, network_type, epochs, batch_size, train_len, validation_split, multilabel, early_stopping = read_parameters_from_file(sys.argv[1])
        # Note: If parameters are read from file, target_byte is not currently supported in the param file format
        # but we can enforce it here if we want, or rely on defaults. 
        # For now, let's just print a warning if we are in this mode.
        print("Note: Running with user provided parameters. Target byte override to %d active." % target_byte)

    #load traces
    # Passing target_byte=None to use the pre-calculated labels from the new dataset file
    (X_profiling, Y_profiling), (X_attack, Y_attack) = load_ascad(ascad_database, target_byte=None)

    #get network type
    if(network_type=="mlp"):
        best_model = mlp_best(input_dim=len(X_profiling[0]))
    elif(network_type=="cnn"):
        best_model = cnn_best(input_dim=len(X_profiling[0]))
    elif(network_type=="cnn2"):
        best_model = cnn_best2(input_dim=len(X_profiling[0]))
    elif(network_type=="multi_test"):
        best_model = multi_test(input_dim=len(X_profiling[0]))
    elif(network_type=="multi_resnet"):
        best_model = resnet_v1((15000,1), 19)
    elif(network_type=="multi_resnet_without_permind"):
        best_model = resnet_v1((15000,1), 19, without_permind=1)
    else: #display an error and abort
        print("Error: no topology found for network '%s' ..." % network_type)
        sys.exit(-1);
    #  print best_model.summary()

    ### training
    print("DEBUG: Saving training model to: %s" % training_model)
    if (train_len == 0):
        train_model(X_profiling, Y_profiling, best_model, training_model, epochs, batch_size, multilabel, validation_split, early_stopping)
    else:
        train_model(X_profiling[:train_len], Y_profiling[:train_len], best_model, training_model, epochs, batch_size, multilabel, validation_split, early_stopping)
