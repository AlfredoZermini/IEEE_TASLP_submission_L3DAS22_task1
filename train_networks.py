import sys
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # suppress warnings
import time
import datetime
import random
import csv
from copy import deepcopy
from pathlib import Path
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.utils import plot_model, to_categorical
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import TensorBoard, CSVLogger
from tensorflow.python.keras.backend import set_session
from tensorflow.keras.models import load_model
from models import *
from generator import *


# initialize seed
SEED=0
tf.random.set_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)


# avoid memory GPU pre-allocation
config = tf.compat.v1.ConfigProto()
#config.gpu_options.per_process_gpu_memory_fraction = 0.8
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)


# custom cost function
def cost(y_true, y_pred):
   
   #return K.mean(K.sigmoid())  (K.square(y_true-y_pred)) )
   return K.mean( K.square(y_true-y_pred ) ) 


# train
def train(args, config):

   # number of frames expanded
   args.n_frames = args.frame_neigh*2+1 

   # get files list  
   args.train_files_list = [i for i in os.listdir(args.load_train_data_path) if not i == 'stats.h5'][:18] #os.listdir(args.load_train_data_path)[:30]
   args.val_files_list = os.listdir(args.load_val_data_path)[:300]

   # define input parameters
   [args.n_train_samples, args.n_freq, args.n_features] = [len(args.train_files_list),1025, 3]
   input_img = Input(shape=(args.n_freq, args.n_frames, args.n_features ) )
   
   # switch between models
   if args.DNN_type == 'MLP': # Total params: 38,853,633
      args.model = MLP_model(args, input_img)
      args.model.summary()
      
   
   elif args.DNN_type == 'CNN': # Total params: 85,284,737
      args.model = CNN_model(args, input_img)
      args.model.summary()
   
   # compile
   #adam = Adam(lr=1e-03, beta_1=0.9, beta_2=0.999, epsilon=0.95, decay=0.0)
   sgd = SGD(lr=0.1, momentum=0.9, decay=0.0, nesterov=False)
   args.model.compile(optimizer=sgd, loss=cost, metrics=[cost])

   # this solved a tricky graph error message
   args.sess = tf.compat.v1.Session()
   args.graph = tf.compat.v1.get_default_graph()
   set_session(args.sess)

   # model path
   checkpoint_filepath= os.path.join(args.models_path,'model_epoch{epoch:02d}.hdf5') #-val_loss{val_loss:.2f}

   # save after every epoch
   checkpoint = ModelCheckpoint(checkpoint_filepath, monitor='val_loss', verbose=1, save_best_only=False, \
      mode='min', save_weights_only = False, save_freq='epoch')

   # tensorboard
   log_dir = "logs/fit" + '_' + args.DNN_type + '_' + args.n_sources_string + '/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

   if not os.path.exists(log_dir):
      os.makedirs(log_dir)

   tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
   
   # define steps
   args.steps_per_epoch = args.n_train_samples  // args.batch_size
   args.validation_steps = len(args.val_files_list) // args.validation_batch_size

   # define generators
   train_generator = generator(args, args.load_train_data_path, args.train_files_list, args.batch_size, 'train')
   val_generator = generator(args, args.load_val_data_path, args.val_files_list, args.validation_batch_size, 'val')
   args.val_check_generator = generator(args, args.load_val_data_path, args.val_files_list[:args.val_samples_check], args.validation_batch_size, 'val')

   # sort models by date
   models_list = sorted(Path(args.models_path).iterdir(), key=os.path.getmtime)
   
   # remove any .log file from models_list
   models_list = [i for i in models_list if '.log' not in os.path.basename(i) and os.path.isdir(i) == False]
   
   # define log file
   csv_logger_path = os.path.join(args.models_path, 'training.log')
   
   # load model if any has been saved
   if models_list != []:

      print('Loading model ', models_list[-1])
      args.model = load_model( models_list[-1], compile=True, custom_objects={'cost': cost}) # compile=False restore optimizer state

      with open(csv_logger_path) as csvfile:
         epochs = []
         csvReader = csv.reader(csvfile, delimiter=',')
         next(csvReader) # jump title row
         for line in csvReader:
            if type(int(line[0])) != str:
               epochs.append(int(line[0]))

      # get last epoch  
      last_epoch = epochs[-1]

   else:
      last_epoch = 0

   # keep track of training history
   csv_logger = CSVLogger(csv_logger_path, separator=',', append=True)
   print(csv_logger)


   # fit model
   args.model.fit(train_generator,\
      validation_data=val_generator, validation_steps=args.validation_steps, validation_freq=1,\
      epochs=args.n_epochs, steps_per_epoch=args.steps_per_epoch, initial_epoch=last_epoch, \
      #use_multiprocessing = True, workers=1, \
      callbacks=[tensorboard_callback, checkpoint, csv_logger])
    
   print("Finished training", time.ctime())
   
   return None
       
if __name__ == '__main__':
   print("Training")
   K.clear_session()