''' A CNN to classify 6 fret-string positions
    at the frame level during guitar performance
'''

from __future__ import print_function
import tensorflow.keras as keras
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Reshape, Activation
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv1D, Lambda
from tensorflow.keras import backend as K
from DataGenerator import DataGenerator
import pandas as pd
import numpy as np
import datetime
from Metrics import *
import tensorflow as tf
from sklearn.model_selection import KFold

# Configure GPU for TensorFlow
def configure_gpu():
    """Configure TensorFlow to use GPU efficiently."""
    try:
        # Get list of GPU devices
        gpus = tf.config.experimental.list_physical_devices('GPU')
        
        if gpus:
            gpu = gpus[0]  # Use the first GPU
            print(f"Found GPU: {gpu}")
            
            # Configure memory growth to avoid allocating all memory at once
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
                print("Memory growth enabled")
            except RuntimeError as e:
                print(f"Memory growth can't be enabled: {e}")
            
            # Set visible devices to just this GPU
            try:
                tf.config.experimental.set_visible_devices(gpu, 'GPU')
                print(f"Using GPU: {gpu}")
            except RuntimeError as e:
                print(f"Failed to set visible devices: {e}")
                
            # Log GPU info
            gpu_info = tf.config.experimental.get_device_details(gpu)
            print(f"GPU details: {gpu_info}")
            
            # Enable mixed precision for faster training
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)
            print("Mixed precision policy set to mixed_float16")
            
            return True
        else:
            print("No GPU detected")
            return False
            
    except Exception as e:
        print(f"Error configuring GPU: {e}")
        return False

class TabCNN:
    
    def __init__(self, 
                 batch_size=128, 
                 epochs=8,
                 con_win_size = 9,
                 spec_repr="c",
                 data_path="f:/newrepos/olitunes/backend/models/tab-cnn/data/spec_repr/",
                 id_file="f:/newrepos/olitunes/backend/models/tab-cnn/data/id.csv",
                 save_path="f:/newrepos/olitunes/backend/models/tab-cnn/model/saved/"):   
        
        self.batch_size = batch_size
        self.epochs = epochs
        self.con_win_size = con_win_size
        self.spec_repr = spec_repr
        self.data_path = data_path
        self.id_file = id_file
        self.save_path = save_path
        
        # Configure GPU
        self.gpu_available = configure_gpu()
        
        self.load_IDs()
        self.validate_data_files()
        
        self.save_folder = self.save_path + self.spec_repr + " " + datetime.datetime.now().strftime("%Y-%m-%d %H%M%S") + "/"
        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)
        self.log_file = self.save_folder + "log.txt"
        
        self.metrics = {}
        self.metrics["pp"] = []
        self.metrics["pr"] = []
        self.metrics["pf"] = []
        self.metrics["tp"] = []
        self.metrics["tr"] = []
        self.metrics["tf"] = []
        self.metrics["tdr"] = []
        self.metrics["data"] = ["g0","g1","g2","g3","g4","g5","mean","std dev"]
        
        if self.spec_repr == "c":
            self.input_shape = (192, self.con_win_size, 1)
        elif self.spec_repr == "m":
            self.input_shape = (128, self.con_win_size, 1)
        elif self.spec_repr == "cm":
            self.input_shape = (320, self.con_win_size, 1)
        elif self.spec_repr == "s":
            self.input_shape = (1025, self.con_win_size, 1)
            
        # these probably won't ever change
        self.num_classes = 21
        self.num_strings = 6

    def load_IDs(self):
        csv_file = self.id_file
        self.list_IDs = list(pd.read_csv(csv_file, header=None)[0])
        
    def validate_data_files(self):
        """Pre-scan all data files to identify which ones are valid and can be used for training."""
        print("Validating data files...")
        valid_IDs = []
        invalid_count = 0
        
        import numpy as np
        
        for ID in self.list_IDs:
            try:
                # determine filename and path
                data_dir = self.data_path + self.spec_repr + "/"
                filename = ID + ".npz"
                filepath = data_dir + filename
                
                # Check if file exists
                import os
                if not os.path.exists(filepath):
                    print(f"File not found: {filepath}")
                    invalid_count += 1
                    continue
                
                # Try to load the file to check if it's valid
                try:
                    # Just try to open the file to check if it's not corrupted
                    # Don't validate the content structure as that will be handled in DataGenerator
                    with np.load(filepath, allow_pickle=True) as data:
                        pass
                    
                    # If we reach here, at least the file can be opened
                    valid_IDs.append(ID)
                    
                except Exception as e:
                    print(f"Error loading {filepath}: {str(e)}")
                    invalid_count += 1
                    continue
                    
            except Exception as e:
                print(f"Error validating {ID}: {str(e)}")
                invalid_count += 1
                continue
        
        self.validated_IDs = valid_IDs
        print(f"Data validation complete: {len(valid_IDs)} valid files, {invalid_count} invalid/missing files.")
        
        # Update the original list_IDs to only contain valid IDs
        if len(valid_IDs) > 0:
            self.list_IDs = valid_IDs
            print("Using validated IDs for training.")
        else:
            print("WARNING: No valid files found! Using original ID list instead.")
            
    def split_data(self):
        """
        Splits list_IDs into five parts for five-fold cross-validation
        """
        print(f"Splitting data with {len(self.list_IDs)} IDs")
        
        # If we have no IDs, log an error
        if len(self.list_IDs) == 0:
            print("ERROR: No IDs available for splitting data. Training cannot proceed.")
            raise ValueError("No IDs found for training. Please check your data files.")
            
        n_folds = 5
        self.n_ids = len(self.list_IDs)
        self.folds = []
        skf = KFold(n_splits=n_folds, shuffle=True)
        
        self.indices = np.arange(self.n_ids)
        for train, test in skf.split(self.indices):
            self.folds.append({
                "training": train,
                "validation": test
            })
            
    def partition_data(self, fold):
        self.partition = {}
        self.partition["training"] = [self.list_IDs[i] for i in self.folds[fold]["training"]]
        self.partition["validation"] = [self.list_IDs[i] for i in self.folds[fold]["validation"]]
        
        self.training_generator = DataGenerator(self.partition['training'], 
                                                data_path=self.data_path, 
                                                batch_size=self.batch_size,  
                                                shuffle=True,
                                                spec_repr=self.spec_repr, 
                                                con_win_size=self.con_win_size)
        
        self.validation_generator = DataGenerator(self.partition['validation'], 
                                                 data_path=self.data_path, 
                                                 batch_size=self.batch_size,  
                                                 shuffle=False,
                                                 spec_repr=self.spec_repr, 
                                                 con_win_size=self.con_win_size)
        
        self.split_folder = self.save_folder + str(fold) + "/"
        if not os.path.exists(self.split_folder):
            os.makedirs(self.split_folder)
                
    def log_model(self):
        with open(self.log_file, 'w', encoding='utf-8') as fh:
            fh.write("\nbatch_size: " + str(self.batch_size))
            fh.write("\nepochs: " + str(self.epochs))
            fh.write("\nspec_repr: " + str(self.spec_repr))
            fh.write("\ndata_path: " + str(self.data_path))
            fh.write("\ncon_win_size: " + str(self.con_win_size))
            fh.write("\nid_file: " + str(self.id_file) + "\n")
            self.model.summary(print_fn=lambda x: fh.write(x + '\n'))
       
    def softmax_by_string(self, t):
        sh = K.shape(t)
        string_sm = []
        for i in range(self.num_strings):
            string_sm.append(K.expand_dims(K.softmax(t[:,i,:]), axis=1))
        return K.concatenate(string_sm, axis=1)
    
    def catcross_by_string(self, target, output):
        loss = 0
        for i in range(self.num_strings):
            loss += K.categorical_crossentropy(target[:,i,:], output[:,i,:])
        return loss
    
    def avg_acc(self, y_true, y_pred):
        return K.mean(K.equal(K.argmax(y_true, axis=-1), K.argmax(y_pred, axis=-1)))
           
    def build_model(self):
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3),
                             activation='relu',
                             input_shape=self.input_shape))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))   
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self.num_classes * self.num_strings)) # no activation
        model.add(Reshape((self.num_strings, self.num_classes)))
        model.add(Activation(self.softmax_by_string))

        # Use a more efficient optimizer when GPU is available
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001) if self.gpu_available else tf.keras.optimizers.Adadelta()
        
        model.compile(loss=self.catcross_by_string,
                      optimizer=optimizer,
                      metrics=[self.avg_acc])
        
        self.model = model

    def train(self):
        # Use the original batch_size from initialization
        print(f"Training with batch size: {self.batch_size}")
        
        # Datasets - Use the proper batch size
        training_generator = DataGenerator(self.partition['training'], 
                                           data_path=self.data_path, 
                                           batch_size=self.batch_size,  # Use the class batch_size
                                           shuffle=True,
                                           spec_repr=self.spec_repr, 
                                           con_win_size=self.con_win_size)
        validation_generator = DataGenerator(self.partition['validation'], 
                                             data_path=self.data_path, 
                                             batch_size=self.batch_size,  # Use the class batch_size
                                             shuffle=False,
                                             spec_repr=self.spec_repr, 
                                             con_win_size=self.con_win_size)
        
        # Track how many valid samples we're using
        print(f"Training with {len(self.partition['training'])} samples, validating with {len(self.partition['validation'])} samples")
        print(f"Note: Some samples may be skipped due to file issues (see warnings above)")
        
        # Configure callbacks for better training
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=3,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=2,
                min_lr=0.0001
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=self.split_folder + "best_model.weights.h5",
                save_best_only=True,
                monitor='val_loss'
            )
        ]
        
        # Train model on dataset
        model_history = self.model.fit(
            x=training_generator,
            validation_data=validation_generator,
            epochs=self.epochs,
            verbose=1,
            callbacks=callbacks
        )
        
        # Print training summary
        print("\nTraining completed.")
        if hasattr(model_history, 'history'):
            # Check if we have any metrics to report
            if 'avg_acc' in model_history.history:
                print(f"Final training accuracy: {model_history.history['avg_acc'][-1]:.4f}")
            if 'val_avg_acc' in model_history.history:
                print(f"Final validation accuracy: {model_history.history['val_avg_acc'][-1]:.4f}")
        
        # Save the model weights after training
        print("Saving model weights...")
        self.save_weights()
        print("Model weights saved successfully.")

    def save_weights(self):
        # In newer Keras versions, the filename must end in .weights.h5
        self.model.save_weights(self.split_folder + "model.weights.h5")
        
    def test(self):
        self.X_test, self.y_gt = self.validation_generator[0]
        self.y_pred = self.model.predict(self.X_test)
        
    def save_predictions(self):
        np.savez(self.split_folder + "predictions.npz", y_pred=self.y_pred, y_gt=self.y_gt)
        
    def evaluate(self):
        self.metrics["pp"].append(pitch_precision(self.y_pred, self.y_gt))
        self.metrics["pr"].append(pitch_recall(self.y_pred, self.y_gt))
        self.metrics["pf"].append(pitch_f_measure(self.y_pred, self.y_gt))
        self.metrics["tp"].append(tab_precision(self.y_pred, self.y_gt))
        self.metrics["tr"].append(tab_recall(self.y_pred, self.y_gt))
        self.metrics["tf"].append(tab_f_measure(self.y_pred, self.y_gt))
        self.metrics["tdr"].append(tab_disamb(self.y_pred, self.y_gt))
        
    def save_results_csv(self):
        output = {}
        for key in self.metrics.keys():
            if key != "data":
                vals = self.metrics[key]
                mean = np.mean(vals)
                std = np.std(vals)
                output[key] = vals + [mean, std]
        output["data"] =  self.metrics["data"]
        df = pd.DataFrame.from_dict(output)
        df.to_csv(self.save_folder + "results.csv") 
        
##################################
########### EXPERIMENT ###########
##################################
def main():
    try:
        print("Starting TabCNN training process...")
        
        # Check if GPU is available
        gpu_available = configure_gpu()
        print(f"GPU acceleration available: {gpu_available}")
        
        tabcnn = TabCNN()
        tabcnn.split_data()
        print("logging model...")
        tabcnn.build_model()
        tabcnn.log_model()
        
        for fold in range(5):
            try:
                print("\nfold " + str(fold))
                tabcnn.partition_data(fold)
                print("building model...")
                tabcnn.build_model()  
                print("training...")
                tabcnn.train()
                print("testing...")
                tabcnn.test()
                tabcnn.save_predictions()
                print(f"Fold {fold} completed successfully.")
            except Exception as e:
                print(f"Error in fold {fold}: {str(e)}")
                print("Continuing with next fold...")
                continue
        
        print("TabCNN training process completed.")
    except Exception as e:
        print(f"Error in TabCNN training process: {str(e)}")
        print("Training interrupted.")
        
if __name__ == '__main__':
    main()