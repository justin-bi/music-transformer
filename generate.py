import torch
import torch.nn as nn
import os
import random
import pickle

# from third_party.midi_processor.processor import decode_midi, encode_midi

from utilities.argument_funcs import parse_generate_args, print_generate_args
from model.music_transformer import MusicTransformer
from dataset.e_piano import create_epiano_datasets

from utilities.constants import *
from utilities.device import get_device, use_cuda

# main
def main():
    """
    Author: Damon Gwinn
    Entry point. Generates music from a model specified by command line arguments
    """

    args = parse_generate_args()
    print_generate_args(args)

    if(args.force_cpu):
        use_cuda(False)
        print("WARNING: Forced CPU usage, expect model to perform slower\n")

    os.makedirs(args.output_dir, exist_ok=True)

    model = MusicTransformer(n_layers=args.n_layers, num_heads=args.num_heads,
                             d_model=args.d_model, dim_feedforward=args.dim_feedforward,
                             max_sequence=args.max_sequence, rpr=args.rpr).to(get_device())

    # model.load_state_dict(torch.load(args.model_weights))
    model.load_state_dict(torch.load(args.model_weights), strict=False)

    # Grabbing dataset if needed
    _, _, dataset = create_epiano_datasets(
        args.midi_root, args.num_prime, random_seq=False, test=True)

    # Can be None, an integer index to dataset, or a file path
    # FOR NOW, always default to a random primer file.
    if(args.primer_file is None):
        f = str(random.randrange(len(dataset)))
    # else:
    #     f = args.primer_file

    if(f.isdigit()):
        idx = int(f)
        primer, _ = dataset[idx]
        primer = primer.to(get_device())

        print("Using primer index:", idx, "(", dataset.data_files[idx], ")")

    # else:
    #     raw_mid = encode_midi(f)
    #     if(len(raw_mid) == 0):
    #         print("Error: No midi messages in primer file:", f)
    #         return

    #     primer, _ = process_midi(raw_mid, args.num_prime, random_seq=False)
    #     primer = torch.tensor(
    #         primer, dtype=TORCH_LABEL_TYPE, device=get_device())

    #     print("Using primer file:", f)

    # Saving primer first
    f_path = os.path.join(args.output_dir, "primer.mid")
    # decode_midi(primer[:args.num_prime].cpu().numpy(), file_path=f_path)

    # GENERATION
    model.eval()
    with torch.set_grad_enabled(False):
        if(args.beam > 0):
            print("BEAM:", args.beam)
            beam_seq = model.generate(
                primer[:args.num_prime], args.target_seq_length, beam=args.beam)

            # f_path = os.path.join(args.output_dir, "beam.mid")
            f_path = os.path.join(args.output_dir, "beam.pickle")
            with open(f_path, 'wb') as dump_file:
                pickle.dump(beam_seq[0].cpu(), dump_file)
            # decode_midi(beam_seq[0].cpu().numpy(), file_path=f_path)
        else:
            print("RAND DIST")
            rand_seq = model.generate(
                primer[:args.num_prime], args.target_seq_length, beam=0)

            # f_path = os.path.join(args.output_dir, "rand.mid")
            f_path = os.path.join(args.output_dir, "rand.pickle")
            with open(f_path, 'wb') as dump_file:
                pickle.dump(rand_seq[0].cpu(), dump_file)
            # decode_midi(rand_seq[0].cpu().numpy(), file_path=f_path)


if __name__ == "__main__":
    print("generate.py main function")
    main()
