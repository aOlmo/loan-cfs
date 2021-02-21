import pdb
import torch
import torch.nn as nn
import torchvision

from datasets import *
from sklearn.model_selection import train_test_split

class AE(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.encoder_layer = nn.Linear(
            in_features=kwargs["input_shape"], out_features=16
        )
        self.decoder_output_layer = nn.Linear(
            in_features=16, out_features=kwargs["input_shape"]
        )

    def forward(self, features):
        activation = self.encoder_layer(features)
        code = torch.relu(activation)
        decoded = self.decoder_output_layer(code)
        reconstructed = torch.relu(decoded)
        return reconstructed

credit_dict = get_credit()

#TODO: Should we separate x and y?
x, y = credit_dict["x_y"]
in_dim = x.shape[1]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

#  use gpu if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# create a model from `AE` autoencoder class
# load it to the specified device, either gpu or cpu
model = AE(input_shape=in_dim).to(device)

# create an optimizer object
# Adam optimizer with learning rate 1e-3
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# mean-squared error loss
criterion = nn.MSELoss()

transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

x_train = torch.Tensor(x_train)

train_loader = torch.utils.data.DataLoader(
    x_train, batch_size=32, shuffle=True, num_workers=4, pin_memory=True
)

epochs = 50
for epoch in range(epochs):
    loss = 0
    for batch_features in train_loader:
        # reshape mini-batch data to [N, 784] matrix
        # load it to the active device
        batch_features = batch_features.view(-1, in_dim).to(device).type(torch.float32)

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

print("1")

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
