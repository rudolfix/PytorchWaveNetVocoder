#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2017 Tomoki Hayashi (Nagoya University)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import argparse
import logging
import os
import sys

import numpy as np
import soundfile as sf
import torch

from torchvision import transforms
from pathlib import Path

from wavenet_vocoder.nets import decode_mu_law
from wavenet_vocoder.nets import encode_mu_law
from wavenet_vocoder.nets import WaveNet
from wavenet_vocoder.utils import find_files, parse_wave_file_name
from wavenet_vocoder.utils import read_txt


def decode_from_wav(x,
                    n_samples,
                    wav_transform=None,
                    speaker_code=0):
    """GENERATE DECODING BATCH.

    Args:
        x (ndarray): waveform to feed to wavenet as seed
        n_samples: new samples to generate
        wav_transform (func): Preprocessing function for waveform.
        speaker_code (int): int speaker code

    Returns:
        generator: Generator instance.

    """
    # ---------------------------
    # sample-by-sample generation
    # ---------------------------

    # new

    # feature file with just speaker code extended over sample size
    h = np.empty((x.shape[0] + n_samples, 0), dtype=np.float32)
    sc = np.tile(speaker_code, [h.shape[0], 1])
    h = np.concatenate([h, sc], axis=1)

    # perform pre-processing
    if wav_transform is not None:
        x = wav_transform(x)

    # convert to torch variable
    x = torch.from_numpy(x).long()
    h = torch.from_numpy(h).float()
    x = x.unsqueeze(0)  # 1 => 1 x 1
    h = h.transpose(0, 1).unsqueeze(0)  # T x C => 1 x C x T

    # send to cuda
    if torch.cuda.is_available():
        x = x.cuda()
        h = h.cuda()

    return x, h


def main():
    """RUN DECODING."""
    parser = argparse.ArgumentParser()
    # decode setting
    parser.add_argument("--seed_waveforms", required=True,
                        type=str, help="directory or list of wav files")
    parser.add_argument("--checkpoint", required=True,
                        type=str, help="model file")
    parser.add_argument("--outdir", required=True,
                        type=str, help="directory to save generated samples")
    parser.add_argument("--config", default=None,
                        type=str, help="configure file")
    parser.add_argument("--fs", default=16000,
                        type=int, help="sampling rate")
    parser.add_argument("--batch_size", default=1,
                        type=int, help="number of batches to decode of batch_length")
    parser.add_argument("--speaker_code",
                        type=int, help="speaker code")
    # other setting
    parser.add_argument("--intervals", default=1000,
                        type=int, help="log interval")
    parser.add_argument("--seed", default=1,
                        type=int, help="seed number")
    parser.add_argument("--verbose", default=1,
                        type=int, help="log level")
    args = parser.parse_args()

    # set log level
    if args.verbose > 0:
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s',
                            datefmt='%m/%d/%Y %I:%M:%S')
    elif args.verbose > 1:
        logging.basicConfig(level=logging.DEBUG,
                            format='%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s',
                            datefmt='%m/%d/%Y %I:%M:%S')
    else:
        logging.basicConfig(level=logging.WARNING,
                            format='%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s',
                            datefmt='%m/%d/%Y %I:%M:%S')
        logging.warning("logging is disabled.")

    # show arguments
    for key, value in vars(args).items():
        logging.info("%s = %s" % (key, str(value)))

    # check arguments
    if args.config is None:
        args.config = os.path.dirname(args.checkpoint) + "/model.conf"
    if not os.path.exists(args.config):
        raise FileNotFoundError("config file is missing (%s)." % (args.config))

    # check directory existence
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    if os.path.isdir(args.seed_waveforms):
        filenames = sorted(find_files(args.seed_waveforms, "*.wav", use_dir_name=False))
        wav_list = [args.seed_waveforms + "/" + filename for filename in filenames]
    elif os.path.isfile(args.seed_waveforms):
        wav_list = read_txt(args.seed_waveforms)
    else:
        logging.error("--seed_waveforms should be directory or list.")
        sys.exit(1)

    # if speaker code is specified then filter out all other speakers if files conform to wave set pattern
    if not args.speaker_code is None:
        try:
            wav_list =  [file for file in wav_list if parse_wave_file_name(file)[1] == args.speaker_code]
        except ValueError:
            # files not in wave set format, then original list will be kept
            pass

    # fix seed
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # fix slow computation of dilated conv
    # https://github.com/pytorch/pytorch/issues/15054#issuecomment-450191923
    torch.backends.cudnn.benchmark = True

    # load config
    config = torch.load(args.config)

    wav_transform = transforms.Compose([
        lambda x: encode_mu_law(x, config.n_quantize)])

    # set default gpu and do not track gradient
    torch.set_grad_enabled(False)

    # define model and load parameters
    model = WaveNet(
        n_quantize=config.n_quantize,
        n_aux=1,
        n_resch=config.n_resch,
        n_skipch=config.n_skipch,
        dilation_depth=config.dilation_depth,
        dilation_repeat=config.dilation_repeat,
        kernel_size=config.kernel_size)

    checkpoint = torch.load(args.checkpoint, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    if torch.cuda.is_available():
        model.cuda()

    # extract train iterations
    train_iterations = checkpoint["iterations"]

    # choose seed sample
    wav_id = np.random.randint(0, high=len(wav_list))
    wav_file = wav_list[wav_id]
    seed_x, _rate = sf.read(wav_file, dtype=np.float32)
    assert args.fs == _rate, f"expected sample rate is {args.fs}, {wav_file} is {_rate}"
    tot_sample_length = config.batch_length + model.receptive_field
    max_sample_id = seed_x.shape[0] - tot_sample_length
    assert max_sample_id >= 0
    sample_id = np.random.randint(0, max_sample_id)
    seed_sample = seed_x[sample_id:sample_id + tot_sample_length]

    # take code from filename if none
    speaker_code = parse_wave_file_name(wav_file)[1] if args.speaker_code is None else args.speaker_code

    # log decoding info
    output_fn_info = f"I{train_iterations}-R{args.seed}-S{speaker_code}-W{Path(wav_file).stem}"
    length_sec = args.batch_size * config.batch_length / args.fs
    logging.info(f"generating {args.batch_size} batches of {config.batch_length} samples = {length_sec} seconds"
                 f" into {output_fn_info} set")

    # write seed sample
    sf.write(args.outdir + f"/seed_sample-{output_fn_info}.wav", seed_sample, args.fs, "PCM_16")
    logging.info("wrote seed_sample.wav in %s." % args.outdir)

    # decode
    new_samples = args.batch_size*config.batch_length
    x, h = decode_from_wav(
        seed_sample,
        new_samples,
        wav_transform=wav_transform,
        speaker_code=speaker_code
    )

    def save_samples(samples, file_name):
        wav_data = decode_mu_law(samples, config.n_quantize)
        sf.write(file_name, wav_data, args.fs, "PCM_16")

    def progress_callback(samples, no_samples, elapsed):
        save_samples(samples, args.outdir + "/" + f"decoded.t.all-{output_fn_info}.wav")
        save_samples(samples[-no_samples:], args.outdir + "/" + f"decoded.t.new-{output_fn_info}.wav")

    logging.info("decoding (length = %d)" % h.shape[2])
    samples = model.fast_generate(x, h, new_samples, args.intervals, callback=progress_callback)
    # samples = model.generate(x, h, new_samples, args.intervals)
    logging.info(f"decoded {len(seed_sample)}")
    save_samples(samples, args.outdir + "/" + f"decoded-{output_fn_info}.wav")
    logging.info("wrote decoded.wav in %s." % args.outdir)


if __name__ == "__main__":
    main()
