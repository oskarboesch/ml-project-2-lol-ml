import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class VAE(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=200):
        super(VAE, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, latent_dim),
            nn.LeakyReLU(0.2)
        )
        
        # Latent mean and variance
        self.mean_layer = nn.Linear(latent_dim, 2)
        self.logvar_layer = nn.Linear(latent_dim, 2)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(2, latent_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )
     
    def encode(self, x):
        x = self.encoder(x)
        mean, logvar = self.mean_layer(x), self.logvar_layer(x)
        return mean, logvar

    def reparameterization(self, mean, logvar):
        epsilon = torch.randn_like(logvar).to(device)
        z = mean + torch.exp(0.5 * logvar) * epsilon
        return z

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterization(mean, logvar)
        x_hat = self.decode(z)
        return x_hat, mean, logvar

# Define the loss function as an external function
def loss_function(x, x_hat, mean, logvar):
    # Reconstruction loss
    reconstruction_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction="sum")
    # KL divergence loss
    KLD = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
    return reconstruction_loss + KLD
def train(model, train_loader, optimizer, epochs, device):
    model.train()  # Set model to training mode
    for epoch in range(epochs):
        overall_loss = 0
        for batch_idx, (x,) in enumerate(train_loader):  # Single input tensor
            x = x[0] if isinstance(x, tuple) else x  # Handle tuple if exists
            x = x.view(x.size(0), -1).to(device)  # Flatten input for VAE

            optimizer.zero_grad()

            x_hat, mean, logvar = model(x)
            loss = loss_function(x, x_hat, mean, logvar)  # Use external loss function
            
            overall_loss += loss.item()
            
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}/{epochs}, Average Loss: {overall_loss / len(train_loader.dataset):.4f}")
    return overall_loss

# Exemple d'utilisation
# Assurez-vous que `train_loader` est défini et contient vos données d'entraînement
# input_dim = 784  # Par exemple, pour les images MNIST
# latent_dim = 2
# model = VAE(input_dim=input_dim, latent_dim=latent_dim).to(device)
# optimizer = optim.Adam(model.parameters(), lr=1e-3)
# train(model, train_loader, optimizer, epochs=50, device=device)