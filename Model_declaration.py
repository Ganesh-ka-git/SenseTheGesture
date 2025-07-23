# Libraries for Model Building
from keras.models import Sequential
from keras.layers import BatchNormalization, Flatten, Dropout, Dense
from keras.layers import Conv3D, MaxPooling3D, GlobalAveragePooling3D

def create_model3DCNN(video_len, height, width):
    # Building Model
    model = Sequential()
    
    # Block 1
    model.add(Conv3D(16, kernel_size=(3, 3, 3), activation='relu', padding='same', input_shape=(video_len, height, width, 3)))
    model.add(BatchNormalization())
    model.add(MaxPooling3D(pool_size=(1, 2, 2)))  # Temporal dimension preserved
    
    # Block 2
    model.add(Conv3D(32, kernel_size=(3, 3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))  # Temporal downsample
    
    # Block 3
    model.add(Conv3D(64, kernel_size=(3, 3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    
    # Block 4 (optional but adds power)
    model.add(Conv3D(128, kernel_size=(3, 3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    
    # Global Feature Compression
    model.add(GlobalAveragePooling3D())
    
    # Fully Connected Layers
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.4))
    
    # Output Layer
    model.add(Dense(5, activation='softmax'))  # 5 gesture classes
    return model