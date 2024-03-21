from util import *
from generate import *
from constants import *
from SongRNN import *
import torch
from train import *
# from generate import *
import json
import argparse
import gc
import numpy as np

with open(INPUT_TRAIN_PATH, 'r') as f:
    char_set = sorted(set(f.read()))

char_idx_map = {character: index for index, character in enumerate(char_set)}

# TODO determine which device to use (cuda or cpu)
device = "cpu"
if torch.cuda.is_available():
    device = torch.device("cuda")   


if __name__ == "__main__":
    #python3 main.py --config config.json  -> To Run the code
    set_seed()

    #Parse the input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.json', help='Specify the config file')
    args = parser.parse_args()

    
    print("Training on device: ", device)

    # Load the configuration from the specified config file
    with open(f'configs/{args.config}', "r") as config_file:
        config = json.load(config_file)

    # Extract configuration parameters
    MAX_GENERATION_LENGTH = config["max_generation_length"]
    TEMPERATURE = config["temperature"]
    SHOW_HEATMAP = config["show_heatmap"]
    
    learning_rate = config["learning_rate"]
    sequence_size = config["sequence_size"]
    no_epochs = config["no_epochs"]
    no_layers = config["no_layers"]
    hidden_size = config["hidden_size"]
    dropout = config["dropout"]
    model_type = config["model_type"]
   
    generated_song_file_path = config["generated_song_file_path"]
    evaluate_model_only = config["evaluate_model_only"]
    model_path = config["model_path"]

    # Load training and validation data
    data = load_data(INPUT_TRAIN_PATH, config)
    data_val = load_data(INPUT_VAL_PATH, config)

    print('==> Building model..')

    in_size, out_size = len(char_set), len(char_set)
    # Initialize the SongRNN model
    model = SongRNN(in_size, out_size, config)

    # If evaluating model only and trained model path is provided:
    if(evaluate_model_only and model_path != ""):
        # Load the checkpoint from the specified model path
        checkpoint = torch.load(model_path)

        # Load the model's state dictionary from the checkpoint
        model.load_state_dict(checkpoint['model_state_dict'])
        print('==> Model loaded from checkpoint..')
    else:
        # Train the model and get the training and validation losses
        losses, v_losses = train(model, data, data_val, char_idx_map, config, device)
        
        loss_plot_file_name = f"model={model_type} epochs={no_epochs} layers={no_layers} hidden_size={hidden_size} dropout={dropout} lr={learning_rate} max_len={MAX_GENERATION_LENGTH} temp={TEMPERATURE} seq_size={sequence_size} train_loss={np.round(losses[np.argmin(v_losses)],4)} val_loss={np.round(np.min(v_losses),4)}" 
        
        # Plot the training and validation losses
        plot_losses(losses, v_losses, loss_plot_file_name)

        
    # As a fun exercise, after your model is well-trained you can see how the model extends Beethoven's famous fur-elise tune 
#     with open("./data/fur_elise.txt", 'r') as file:
#         prime_str = file.read()
#     print(">>>>>>>>>> Extending Fur Elise <<<<<<<<<<<")
    
#     generated_song = generate_song(model, device, char_idx_map, max_len=MAX_GENERATION_LENGTH, temp=TEMPERATURE, 
#                                     prime_str=prime_str, show_heatmap=SHOW_HEATMAP)

#     with open(generated_fur_elise_file_path, "w") as file:
#         file.write(generated_song)

#     print("Generated song is written to : ", generated_fur_elise_file_path)

    

    # Generate a song from scratch using the trained model
    prime_str = '<start>'
    generated_song = generate_song(model, device, char_idx_map, max_len=MAX_GENERATION_LENGTH, temp=TEMPERATURE, 
                                    prime_str=prime_str, show_heatmap=SHOW_HEATMAP)

    with open(generated_song_file_path, "w") as file:
        file.write(generated_song)

    print("Generated song is written to : ", generated_song_file_path)

    # housekeeping
    gc.collect()
    torch.cuda.empty_cache()


