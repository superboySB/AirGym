import torch
import os
from lib.network.VAE import VAE


def clean_state_dict(state_dict):
    clean_dict = {}
    for key, value in state_dict.items():
        if "module." in key:
            key = key.replace("module.", "")
        if "dronet." in key:
            key = key.replace("dronet.", "encoder.")
        clean_dict[key] = value
    return clean_dict


class VAEImageEncoder:
    """
    Class that wraps around the VAE class for efficient inference for the aerial_gym class
    """

    def __init__(self, config, device="cuda:0"):
        self.config = config
        self.latent_dim = self.config.latent_dims
        self.vae_model = VAE(input_dim=1, latent_dim=self.config.latent_dims).to(device)
        # combine module path with model file name
        weight_file_path = os.path.join(self.config.model_folder, self.config.model_file)
        # load model weights
        print("Loading weights from file: ", weight_file_path)
        state_dict = clean_state_dict(torch.load(weight_file_path))
        self.vae_model.load_state_dict(state_dict)
        self.vae_model.eval()

    def encode(self, image_tensors):
        """
        Class to encode the set of images to a latent space. We can return both the means and sampled latent space variables.
        """
        with torch.no_grad():
            x_res, y_res = image_tensors.shape[-2], image_tensors.shape[-1]
            if self.config.image_res[0] != x_res or self.config.image_res[1] != y_res:
                interpolated_image = torch.nn.functional.interpolate(
                    image_tensors,
                    self.config.image_res,
                    mode=self.config.interpolation_mode,
                )
            else:
                interpolated_image = image_tensors
            z_sampled, means, *_ = self.vae_model.encode(interpolated_image)
        if self.config.return_sampled_latent:
            returned_val = z_sampled
        else:
            returned_val = means
        return returned_val

    def decode(self, latent_spaces):
        """
        Decode a latent space to reconstruct full images
        """
        with torch.no_grad():
            if latent_spaces.shape[-1] != self.config.latent_dims:
                print(
                    f"ERROR: Latent space size of {latent_spaces.shape[-1]} does not match network size {self.config.latent_dims}"
                )
            decoded_image = self.vae_model.decode(latent_spaces)
        return decoded_image

    def get_latent_dims_size(self):
        """
        Function to get latent space dims
        """
        return self.config.latent_dims