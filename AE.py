import pdb
from pandas import read_csv
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torchvision

def get_adult_train_test():
    full_path = 'data/UCI_adult.csv'
    df = read_csv(full_path, na_values='?')
    df = df.dropna()
    X, y = df.drop(df.columns[-1], axis=1), df[df.columns[-1]]
    y = LabelEncoder().fit_transform(y)

    cat_cols = X.select_dtypes(include=['object', 'bool']).columns
    num_cols = X.select_dtypes(include=['int64', 'float64']).columns

    ct = ColumnTransformer([('cat', OneHotEncoder(), cat_cols), ('num', MinMaxScaler(), num_cols)])
    X_transf = ct.fit_transform(X)
    X = X_transf.toarray()

    return train_test_split(X, y, test_size=0.33, random_state=42)  # X_train, X_test, y_train, y_test


X_train, X_test, y_train, y_test = get_adult_train_test()

class AE(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.encoder_layer = nn.Linear(
            in_features=kwargs["input_shape"], out_features=32
        )
        self.decoder_output_layer = nn.Linear(
            in_features=32, out_features=kwargs["input_shape"]
        )

    def forward(self, features):
        activation = self.encoder_layer(features)
        code = torch.relu(activation)
        decoded = self.decoder_output_layer(code)
        reconstructed = torch.relu(decoded)
        return reconstructed


#  use gpu if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# create a model from `AE` autoencoder class
# load it to the specified device, either gpu or cpu
model = AE(input_shape=104).to(device)

# create an optimizer object
# Adam optimizer with learning rate 1e-3
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# mean-squared error loss
criterion = nn.MSELoss()

transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

train_loader = torch.utils.data.DataLoader(
    X_train, batch_size=128, shuffle=True, num_workers=4, pin_memory=True
)

epochs = 50
for epoch in range(epochs):
    loss = 0
    for batch_features in train_loader:
        # reshape mini-batch data to [N, 784] matrix
        # load it to the active device
        batch_features = batch_features.view(-1, 104).to(device).type(torch.float32)

        # reset the gradients back to zero
        # PyTorch accumulates gradients on subsequent backward passes
        optimizer.zero_grad()

        # compute reconstructions
        outputs = model(batch_features)

        # compute training reconstruction loss
        train_loss = criterion(outputs, batch_features)

        # compute accumulated gradients
        train_loss.backward()

        # perform parameter update based on current gradients
        optimizer.step()

        # add the mini-batch training loss to epoch loss
        loss += train_loss.item()

    # compute the epoch training loss
    loss = loss / len(train_loader)

    # display the epoch training loss
    print("epoch : {}/{}, loss = {:.6f}".format(epoch + 1, epochs, loss))

################################################
# # This is the size of our encoded representations
# encoding_dim = 32  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats
# 
# # This is our input image
# input_img = keras.Input(shape=(104,))
# # "encoded" is the encoded representation of the input
# encoded = layers.Dense(encoding_dim, activation='relu')(input_img)
# # "decoded" is the lossy reconstruction of the input
# decoded = layers.Dense(104, activation='sigmoid')(encoded)
# 
# # This model maps an input to its reconstruction
# autoencoder = keras.Model(input_img, decoded)
# 
# # This is our encoded (32-dimensional) input
# encoded_input = keras.Input(shape=(encoding_dim,))
# # Retrieve the last layer of the autoencoder model
# decoder_layer = autoencoder.layers[-1]
# # Create the decoder model
# decoder = keras.Model(encoded_input, decoder_layer(encoded_input))
# 
# autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
# autoencoder.fit(X_train, X_train,
#                 epochs=50,
#                 batch_size=256,
#                 shuffle=True,
#                 validation_data=(X_test, X_test))
