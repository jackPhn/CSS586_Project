import click
from musiclearn.sequential_models import *
from musiclearn.single_note_processing import *


@click.command()
@click.option('--model-type',
              type=click.Choice(['lstm', 'bidirect', 'att', 'wavenet'], case_sensitive=False),
              help='Type of model')
@click.option('--data-path', type=click.STRING, help='Path to folder stores dataset')
@click.option('--logs-dir', type=click.STRING, help='Folder that stores training logs')
@click.option('--sequence_length', type=click.INT, default=100,
              help='Length of the sequences of notes used in training')
def fit_sequential(model_type,
                   data_path,
                   logs_dir,
                   sequence_length):
    """ Fit a sequential model of choice on the specified dataset """
    notes = read_midi(data_path)
    
    n_vocab = get_num_unique_notes(notes)
    
    network_input, network_output = prepare_sequences(sequence_length, notes)
    
    if model_type == 'lstm':
        model = lstm_model(network_input.shape, n_vocab)
    elif model_type == 'bidirect':
        model = bidirectional_lstm_model(network_input.shape, n_vocab)
    elif model_type == 'att':
        model = attention_lstm_model(network_input.shape, n_vocab)
    elif model_type == 'wavenet':
        model = simplified_wavenet(network_input.shape, n_vocab)
    else:
        raise ValueError("Invalid model type")
    
    train_model(model, sequence_length, model_type, 10, 64)           


if __name__ == "__main__":
    fit_sequential()