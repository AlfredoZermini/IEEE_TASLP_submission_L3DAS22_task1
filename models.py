from tensorflow.keras.layers import Input, Dense, Flatten, Convolution2D, LeakyReLU, MaxPooling2D, BatchNormalization
from tensorflow.keras.models import Model


# MLP model
def MLP_model(args, input_img):

   x = ( Flatten())(input_img)
   
   for i in range(0,4):
      x = ( Dense(1024*args.n_sources))(x)
      x = ( BatchNormalization() )(x)
      x = ( LeakyReLU())(x)
   if args.n_sources == 1:
      o0 = ( Dense(args.n_freq, activation='sigmoid'))(x)
   elif args.n_sources == 2:
      o0 = ( Dense(args.n_freq, activation='sigmoid'))(x)
      o1 = ( Dense(args.n_freq, activation='sigmoid'))(x)
   elif args.n_sources == 3:
      o0 = ( Dense(args.n_freq, activation='sigmoid'))(x)
      o1 = ( Dense(args.n_freq, activation='sigmoid'))(x)
      o2 = ( Dense(args.n_freq, activation='sigmoid'))(x)
   elif args.n_sources == 4:
      o0 = ( Dense(args.n_freq, activation='sigmoid'))(x)
      o1 = ( Dense(args.n_freq, activation='sigmoid'))(x)
      o2 = ( Dense(args.n_freq, activation='sigmoid'))(x)
      o3 = ( Dense(args.n_freq, activation='sigmoid'))(x)

   if args.n_sources == 1:
      MLP = Model(input_img, [o0])
   elif args.n_sources == 2:
      MLP = Model(input_img, [o0,o1])
   elif args.n_sources == 3:
      MLP = Model(input_img, [o0,o1,o2])
   elif args.n_sources == 4:
      MLP = Model(input_img, [o0,o1,o2,o3])

   return MLP


# CNN model
def CNN_model(args, input_img):
   x = ( Convolution2D(64, kernel_size=(5, 5), activation='linear', padding='same' ))(input_img)
   x = ( BatchNormalization() )(x)
   x = ( LeakyReLU())(x)
   x = ( MaxPooling2D(pool_size=(2, 2)))(x)
   x = ( Convolution2D(128, kernel_size=(3, 3), activation='linear', padding='same' ) )(x)
   x = ( BatchNormalization() )(x)
   x = ( LeakyReLU())(x)
   x = ( MaxPooling2D(pool_size=(2, 1)))(x)
   x = ( Convolution2D(256, kernel_size=(8, 1), activation='linear', padding='same' ) )(x)
   x = ( BatchNormalization() )(x)
   x = ( LeakyReLU())(x)
   x = ( MaxPooling2D(pool_size=(4, 1)))(x)
   x = ( Flatten())(x)
   x = ( Dense(1024*args.n_sources, activation='relu'))(x)
   x = ( BatchNormalization() )(x)
   x = ( LeakyReLU())(x)
   if args.n_sources == 1:
      o0 = ( Dense(args.n_freq, activation='sigmoid'))(x)
   elif args.n_sources == 2:
      o0 = ( Dense(args.n_freq, activation='sigmoid'))(x)
      o1 = ( Dense(args.n_freq, activation='sigmoid'))(x)
   elif args.n_sources == 3:
      o0 = ( Dense(args.n_freq, activation='sigmoid'))(x)
      o1 = ( Dense(args.n_freq, activation='sigmoid'))(x)
      o2 = ( Dense(args.n_freq, activation='sigmoid'))(x)
   elif args.n_sources == 4:
      o0 = ( Dense(args.n_freq, activation='sigmoid'))(x)
      o1 = ( Dense(args.n_freq, activation='sigmoid'))(x)
      o2 = ( Dense(args.n_freq, activation='sigmoid'))(x)
      o3 = ( Dense(args.n_freq, activation='sigmoid'))(x)

   if args.n_sources == 1:
      CNN = Model(input_img, [o0])
   elif args.n_sources == 2:
      CNN = Model(input_img, [o0,o1])
   elif args.n_sources == 3:
      CNN = Model(input_img, [o0,o1,o2])
   elif args.n_sources == 4:
      CNN = Model(input_img, [o0,o1,o2,o3])

   return CNN