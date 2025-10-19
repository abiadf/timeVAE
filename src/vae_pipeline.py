import os, warnings
import numpy as np
import time
import torch
import yaml

from data_utils import (
    load_yaml_file,
    load_data,
    split_data,
    scale_data,
    inverse_transform_data,
    save_scaler,
    save_data,)
import paths
from vae.vae_utils import (
    instantiate_vae_model,
    train_vae,
    save_vae_model,
    get_posterior_samples,
    get_prior_samples,
    load_vae_model)
from visualize import plot_samples, plot_latent_space_samples, visualize_and_save_tsne


def run_vae_pipeline(dataset_name: str, vae_type: str):
    # ----------------------------------------------------------------------------------
    # Load data, perform train/valid split, scale data

    data = load_data(data_dir=paths.DATASETS_DIR, dataset=dataset_name)

    # split data into train/valid splits
    train_data, temp_data = split_data(data, valid_perc=0.4, shuffle=False) #train 60%, temp 40%
    valid_data, test_data = split_data(temp_data, valid_perc=0.5, shuffle=False) # split temp into valid & test 50/50

    # scale data
    scaled_train_data, scaled_valid_data, scaler = scale_data(train_data, valid_data)
    n_samples, n_timesteps, n_features = test_data.shape
    scaled_test_data = scaler.transform(test_data.reshape(-1, n_features)).reshape(n_samples, n_timesteps, n_features)

    print(f"{train_data.shape=}, {valid_data.shape=}, {test_data.shape=}")

    scaled_train_data = scaled_train_data.astype(np.float32)
    scaled_valid_data = scaled_valid_data.astype(np.float32)
    scaled_test_data  = scaled_test_data.astype(np.float32)

    print(f"Train mean/std: {scaled_train_data.mean():.3f}, {scaled_train_data.std():.3f} ({scaled_train_data.nbytes/1024**2:.2f} MB)")
    print(f"Valid mean/std: {scaled_valid_data.mean():.3f}, {scaled_valid_data.std():.3f} ({scaled_valid_data.nbytes/1024**2:.2f} MB)")
    print(f"Test mean/std: {scaled_test_data.mean():.3f}, {scaled_test_data.std():.3f} ({scaled_test_data.nbytes/1024**2:.2f} MB)")

    # ----------------------------------------------------------------------------------
    # Instantiate and train the VAE Model

    # load hyperparameters from yaml file
    hyperparameters = load_yaml_file(paths.HYPERPARAMETERS_FILE_PATH)[vae_type]

    # instantiate the model
    _, sequence_length, feature_dim = scaled_train_data.shape
    vae_model = instantiate_vae_model(
        vae_type=vae_type,
        sequence_length=sequence_length,
        feature_dim=feature_dim,
        **hyperparameters,)

    def read_params(file_path: str) -> dict:
        """Read parameters from a YAML file."""
        with open(file_path, "r") as f:
            return yaml.safe_load(f)
    params       = read_params("/Users/fouadabiad/Projects/asm_ML/src/params2.yaml")
    train_epochs = params["timevae"]["train_epochs"]


    start = time.time()
    # train vae
    train_vae(
        vae=vae_model,
        train_data=scaled_train_data,
        max_epochs=train_epochs,
        verbose=1,)
    end = time.time()
    print(f"Training time for {train_epochs} epochs: {(end - start):.2f} s")

    # @@@@@@@@@@ code added below this line is F's addition @@@@@@@@@@@

    # Get and save latent representations
    def get_latents(encoder, data: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (z_mean, z_log_var, z) from encoder output."""
        outputs = encoder.predict(data)
        if len(outputs) == 3:
            z_mean, z_log_var, z = outputs
        else: # only mean & logvar returned
            z_mean, z_log_var = outputs
            z = z_mean # deterministic mean
        return z_mean, z_log_var, z

    # After training
    _, _, z_train = get_latents(vae_model.encoder, scaled_train_data)
    _, _, z_valid = get_latents(vae_model.encoder, scaled_valid_data)
    _, _, z_test  = get_latents(vae_model.encoder, scaled_test_data)
    print(f"{z_train.shape=}, {z_valid.shape=}, {z_test.shape=}")

    os.makedirs("vae_output", exist_ok=True)
    np.save(f"/Users/fouadabiad/Projects/asm_ML/interim_data/z_train_{dataset_name}.npy", z_train)
    np.save(f"/Users/fouadabiad/Projects/asm_ML/interim_data/z_valid_{dataset_name}.npy", z_valid)
    np.save(f"/Users/fouadabiad/Projects/asm_ML/interim_data/z_test_{dataset_name}.npy", z_test)

    model_name = f"timevae_encoder_{dataset_name}.keras"
    vae_model.encoder.save(f"/Users/fouadabiad/Projects/asm_ML/interim_data/timevae/{model_name}")

    # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

    # # ----------------------------------------------------------------------------------
    # # Save scaler and model
    # model_save_dir = os.path.join(paths.MODELS_DIR, dataset_name)
    # # save scaler
    # save_scaler(scaler=scaler, dir_path=model_save_dir)
    # # Save vae
    # save_vae_model(vae=vae_model, dir_path=model_save_dir)

    # # ----------------------------------------------------------------------------------
    # # Visualize posterior samples
    # x_decoded = get_posterior_samples(vae_model, scaled_train_data)
    # plot_samples(
    #     samples1=scaled_train_data,
    #     samples1_name="Original Train",
    #     samples2=x_decoded,
    #     samples2_name="Reconstructed Train",
    #     num_samples=5,
    # )
    # # ----------------------------------------------------------------------------------
    # # Generate prior samples, visualize and save them

    # # Generate prior samples
    # prior_samples = get_prior_samples(vae_model, num_samples=train_data.shape[0])
    # # Plot prior samples
    # plot_samples(
    #     samples1=prior_samples,
    #     samples1_name="Prior Samples",
    #     num_samples=5,
    # )

    # # visualize t-sne of original and prior samples
    # visualize_and_save_tsne(
    #     samples1=scaled_train_data,
    #     samples1_name="Original",
    #     samples2=prior_samples,
    #     samples2_name="Generated (Prior)",
    #     scenario_name=f"Model-{vae_type} Dataset-{dataset_name}",
    #     save_dir=os.path.join(paths.TSNE_DIR, dataset_name),
    #     max_samples=2000,
    # )

    # # inverse transformer samples to original scale and save to dir
    # inverse_scaled_prior_samples = inverse_transform_data(prior_samples, scaler)
    # save_data(
    #     data=inverse_scaled_prior_samples,
    #     output_file=os.path.join(
    #         os.path.join(paths.GEN_DATA_DIR, dataset_name),
    #         f"{vae_type}_{dataset_name}_prior_samples.npz",
    #     ),
    # )

    # # ----------------------------------------------------------------------------------
    # # If latent_dim == 2, plot latent space
    # if hyperparameters["latent_dim"] == 2:
    #     plot_latent_space_samples(vae=vae_model, n=8, figsize=(15, 15))

    # # ----------------------------------------------------------------------------------
    # # later.... load model
    # loaded_model = load_vae_model(vae_type, model_save_dir)

    # # Verify that loaded model produces same posterior samples
    # new_x_decoded = loaded_model.predict(scaled_train_data)
    # print(
    #     "Preds from orig and loaded models equal: ",
    #     np.allclose(x_decoded, new_x_decoded, atol=1e-5),
    # )

    # # ----------------------------------------------------------------------------------


if __name__ == "__main__":
    # check `/data/` for available datasets
    dataset = "germany_catchment_frac0.5_0.3_0.015"

    # models: vae_dense, vae_conv, timeVAE
    model_name = "timeVAE"

    run_vae_pipeline(dataset, model_name)
