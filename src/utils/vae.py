import torch
import torch.nn as nn
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class VAE(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=200):
        super(VAE, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),  # BatchNorm after first linear layer
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, latent_dim),
            nn.BatchNorm1d(latent_dim),  # BatchNorm after second linear layer
            nn.LeakyReLU(0.2)
        )
        
        # Latent mean and variance
        self.mean_layer = nn.Linear(latent_dim, latent_dim)
        self.logvar_layer = nn.Linear(latent_dim, latent_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.BatchNorm1d(latent_dim),  # BatchNorm after latent space input
            nn.LeakyReLU(0.2),
            nn.Linear(latent_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),  # BatchNorm after second decoder layer
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, input_dim),
            nn.Tanh()
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
    reconstruction_loss = nn.functional.mse_loss(x_hat, x)
    # KL divergence loss
    KLD = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
    return reconstruction_loss + KLD
def train(model, train_loader, val_loader, optimizer, epochs, device):
    model.train()
    for epoch in range(epochs):
        train_loss = 0
        for batch_idx, (x,) in enumerate(train_loader):
            x = x.view(x.size(0), -1).to(device)  # Flatten input
            
            optimizer.zero_grad()
            
            x_hat, mean, logvar = model(x)
            loss = loss_function(x, x_hat, mean, logvar)
            train_loss += loss.item()
            
            loss.backward()
            optimizer.step()
        
        # Validation
        val_loss = 0
        model.eval()
        with torch.no_grad():
            for x, in val_loader:
                x = x.view(x.size(0), -1).to(device)
                x_hat, mean, logvar = model(x)
                val_loss += loss_function(x, x_hat, mean, logvar).item()
        
        train_loss /= len(train_loader.dataset)
        val_loss /= len(val_loader.dataset)
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

def plot_latent_space(latent_features, labels):
    """
    Plot the latent features in 2D.
    Args:
        latent_features: Tensor of latent features (2D).
        labels: Corresponding labels for coloring.
    """
    z = latent_features.numpy()  # Convert to NumPy for plotting
    labels = labels.numpy() if isinstance(labels, torch.Tensor) else labels  # Ensure labels are NumPy
    
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(z[:, 0], z[:, 1], c=labels, cmap='viridis', alpha=0.7)
    plt.colorbar(scatter, label='Drug Response (AAC)')
    plt.title("VAE Latent Space")
    plt.xlabel("Latent Dimension 1")
    plt.ylabel("Latent Dimension 2")
    plt.grid(True)
    plt.show()