import os
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
import matplotlib.pyplot as plt

from datasets import *
from sklearn.model_selection import train_test_split

EPOCHS = 80
PATH = "models/ae_credit_model_{}.pt".format(EPOCHS)
TRAIN = 0
LOSS_SCALE = 100
LR = 1e-3

# credit_dict = get_credit()
# x, y = credit_dict["x_y"]
x, y = get_adult(npy=True)
in_dim = x.shape[1]

AE_dims = [in_dim, 24, 8]

class AE(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(AE_dims[0], AE_dims[1]),
            nn.ReLU(True))

        self.decoder = nn.Sequential(
            nn.Linear(AE_dims[1], AE_dims[0]),
            nn.ReLU(True))

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)

#  use gpu if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# create a model from `AE` autoencoder class
# load it to the specified device, either gpu or cpu
model = AE(input_shape=in_dim).to(device)

# create an optimizer object
# Adam optimizer with learning rate 1e-3
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# mean-squared error loss
criterion = nn.MSELoss()

x_train = torch.Tensor(x_train)

train_loader = torch.utils.data.DataLoader(
    x_train, batch_size=128, shuffle=True, num_workers=4, pin_memory=True
)


if not os.path.exists(PATH) or TRAIN:
    losses = []
    for epoch in range(EPOCHS):
        loss = 0
        for batch_features in train_loader:
            # load it to the active device
            batch_features = batch_features.view(-1, in_dim).to(device).type(torch.float32)

            # reset the gradients back to zero
            # PyTorch accumulates gradients on subsequent backward passes
            optimizer.zero_grad()

            # compute reconstructions
            outputs = model(batch_features)

            # compute training reconstruction loss
            train_loss = criterion(outputs, batch_features) * LOSS_SCALE  # For readibility

            # compute accumulated gradients
            train_loss.backward()

            # perform parameter update based on current gradients
            optimizer.step()

            # add the mini-batch training loss to epoch loss
            loss += train_loss.item()

        # compute the epoch training loss
        loss = loss / len(train_loader)

        losses.append(loss)

        # display the epoch training loss
        print("epoch : {}/{}, loss = {:.6f}".format(epoch + 1, EPOCHS, loss))

    print("[+]: Saved model in " + PATH)
    torch.save(model, PATH)
    plt.plot(range(len(losses)), losses)
    plt.show()

else:
    model = AE(input_shape=in_dim).to(device)
    model = torch.load(PATH)

x_train = x_train.to(device)
print("Train MSE: ", criterion(x_train, model(x_train)).item()*LOSS_SCALE)

x_test = torch.Tensor(x_test).to(device)
print("Test MSE: ", criterion(x_test, model(x_test)).item()*LOSS_SCALE)
# print(test.shape)
# print(model(test))

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
