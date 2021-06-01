import click
from musiclearn.sequential_models import *
from musiclearn.single_note_processing import *


@click.command()
@click.argument("output-name", type=click.STRING)
@click.option('--data-path', type=click.STRING, help='Path to folder stores dataset')
@click.option('--model-type',
              type=click.Choice(['lstm', 'bidirect', 'att', 'wavenet'], case_sensitive=False),
              help='Type of model')
@click.option('--weights-path', type=click.STRING, help='Path to saved weights of the model')
@click.option('--num-notes', type=click.INT, help='Number of notes to generate')
def generate_music(output_name,
                   data_path,
                   model_type,
                   weights_path,
                   num_notes):
    """ Generate a short piece of music with a fixed number of notes """
    # load the model
    notes = read_midi(data_path)
    n_vocab = get_num_unique_notes(notes)
    network_input, _ = prepare_sequences(SEQUENCE_LENGTH, notes)
    if model_type == 'lstm' or model_type == 'bidirect':
        model = load_lstm_model(weights_path)
    elif model_type == 'att':
        model = load_attention_lstm_model(weights_path, network_input.shape, n_vocab)
    elif model_type == 'wavenet':
        model = load_wavenet_model(weights_path)
    else:
        raise ValueError("Invalid model type")
    
    generate_midi_sample(model, data_path, output_name, num_notes)

    
if __name__ == "__main__":
    generate_music()