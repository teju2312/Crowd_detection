import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Concatenate, Input
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import resnet50, inception_v3
from keras.callbacks import ReduceLROnPlateau
import concurrent.futures

# Set random seed for reproducibility
np.random.seed(2)

# Set seaborn style
sns.set(style='white', context='notebook', palette='deep')

# Load data
df = pd.read_csv("Dataset/labels.csv")
df['image_name'] = df['id'].map(lambda x: 'seq_{:06d}.jpg'.format(x))

# Setup some constants
size = 224
batch_size = 64
num_epochs = 20

# ImageDataGenerator with preprocessing function
datagen = ImageDataGenerator(
    rescale=1./255,
    preprocessing_function=resnet50.preprocess_input
)

# Function to add one-to-one correlation line to plots
def add_one_to_one_correlation_line(ax, min_factor=1, max_factor=1, **plot_kwargs):
    lim_min, lim_max = pd.DataFrame([ax.get_ylim(), ax.get_xlim()]).agg({0: 'min', 1: 'max'})
    lim_min *= min_factor
    lim_max *= max_factor
    plot_kwargs_internal = dict(color='grey', ls='--')
    plot_kwargs_internal.update(plot_kwargs)
    ax.plot([lim_min, lim_max], [lim_min, lim_max], **plot_kwargs_internal)
    ax.set_ylim([lim_min, lim_max])
    ax.set_xlim([lim_min, lim_max])

# Function to build the combined model
def build_combined_model():
    input_resnet = Input(shape=(size, size, 3))
    input_inception = Input(shape=(size, size, 3))

    base_model_resnet = resnet50.ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=(size, size, 3),
        pooling='avg'
    )

    base_model_inception = inception_v3.InceptionV3(
        weights='imagenet',
        include_top=False,
        input_shape=(size, size, 3),
        pooling='avg'
    )

    x_resnet = base_model_resnet(input_resnet)
    x_resnet = Dense(1024, activation='relu')(x_resnet)
    x_resnet = Dropout(0.5)(x_resnet)

    x_inception = base_model_inception(input_inception)
    x_inception = Dense(1024, activation='relu')(x_inception)
    x_inception = Dropout(0.5)(x_inception)

    combined_features = Concatenate()([x_resnet, x_inception])

    predictions = Dense(1, activation='linear')(combined_features)

    model = Model(inputs=[input_resnet, input_inception], outputs=predictions)

    for layer in base_model_resnet.layers[:-10]:
        layer.trainable = False

    for layer in base_model_inception.layers[:-10]:
        layer.trainable = False

    model.compile(
        optimizer=SGD(learning_rate=0.00096),
        loss="mean_squared_error",
        metrics=['mean_absolute_error', 'mean_squared_error']
    )

    return model

# Function to train the combined model
def train_combined_model(train_data):
    train_features_resnet, train_features_inception = train_data[0][0], train_data[0][1]
    train_labels = train_data[1]

    model = build_combined_model()

    model.fit(
        [train_features_resnet, train_features_inception],
        train_labels,
        epochs=num_epochs,
        verbose=0,
    )

    return model

# Split data into training and validation sets
train_df, valid_df = train_test_split(df, test_size=0.2, random_state=42)

# Create data generators
train_generator = datagen.flow_from_dataframe(
    dataframe=train_df,
    directory='Dataset/frames/frames',
    x_col="image_name",
    y_col="count",
    target_size=(size, size),
    batch_size=batch_size,
    class_mode='raw',
)

valid_generator = datagen.flow_from_dataframe(
    dataframe=valid_df,
    directory='Dataset/frames/frames',
    x_col="image_name",
    y_col="count",
    target_size=(size, size),
    batch_size=batch_size,
    class_mode='raw',
)

# Prepare data for parallel processing
train_data = [(next(train_generator), next(train_generator)) for _ in range(len(train_generator))]

# Train models in parallel
with concurrent.futures.ThreadPoolExecutor() as executor:
    models = list(executor.map(train_combined_model, train_data))

# Predict on validation set using the combined model
valid_features_resnet, valid_features_inception = next(valid_generator)[0], next(valid_generator)[0]
predictions_combined = np.mean([model.predict([valid_features_resnet, valid_features_inception])[:, 0] for model in models], axis=0)

# Calculate Mean Absolute Error (MAE) and Mean Squared Error (MSE)
valid_labels_resnet = next(valid_generator)[1]
mae_combined = np.mean(np.abs(valid_labels_resnet - predictions_combined))
mse_combined = mean_squared_error(valid_labels_resnet, predictions_combined)
print(f'Mean Absolute Error (MAE) of the combined model: {mae_combined:.2f}')
print(f'Mean Squared Error (MSE) of the combined model: {mse_combined:.2f}')
