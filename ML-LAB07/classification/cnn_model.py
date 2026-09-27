import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import json
import os

def build_cnn_model(input_shape, num_classes):
    """
    สร้างโครงสร้างโมเดล Convolutional Neural Network (CNN)
    """
    model = Sequential([
       
        Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        
     
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        
      
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        
     
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        
      
        Dense(1 if num_classes == 2 else num_classes, 
              activation='sigmoid' if num_classes == 2 else 'softmax')
    ])
    
    loss_fn = 'binary_crossentropy' if num_classes == 2 else 'sparse_categorical_crossentropy'
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss=loss_fn,
        metrics=['accuracy']
    )
    
    return model

def train_model(model, X_train, y_train, X_val, y_val, epochs=15, batch_size=32, output_dir="outputs"):
    """
    ฝึกสอนโมเดล CNN พร้อมบันทึกประวัติการเทรน
    """
    os.makedirs(output_dir, exist_ok=True)
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )
    
  
    model_path = os.path.join(output_dir, "cnn_model.keras")
    model.save(model_path)
    
    history_path = os.path.join(output_dir, "history.json")
    with open(history_path, "w") as f:
        json.dump(history.history, f)
        
    print(f"Model saved to {model_path}")
    return model, history