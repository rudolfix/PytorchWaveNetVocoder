#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2017 Tomoki Hayashi (Nagoya University)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import argparse
import logging
import os
import sys
import time

from dateutil.relativedelta import relativedelta
from typing import Dict, Tuple, List
from functools import reduce

import numpy as np
import six
import soundfile as sf
import torch

from torch import nn
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

from wavenet_vocoder.nets import encode_mu_law, decode_mu_law
from wavenet_vocoder.nets import initialize
from wavenet_vocoder.nets import WaveNet
from wavenet_vocoder.utils import background
from wavenet_vocoder.utils import find_files, parse_wave_file_name
from wavenet_vocoder.utils import read_txt


def create_speaker_code_feature(speaker_code, batch_length, receptive_field):
    # we just need one short feature tensor as speaker_code is global
    # empty feature file
    h = np.empty((batch_length + receptive_field, 0), dtype=np.float32)
    sc = np.tile(speaker_code, [h.shape[0], 1])
    h = np.concatenate([h, sc], axis=1)
    h_ = torch.from_numpy(h).float()

    return h_[:-1].transpose(0, 1)


def load_flat_wav_set(file_list, speaker_code=0):
    # create list of following tuples [[(wav_files, speaker_code)]]
    return [[(f, speaker_code)] for f in file_list]


def load_speaker_code_set(file_list, same_speakers=True) -> List[List[Tuple[str, int]]]:
    # loads and groups files of following format Unn_Snnn_*.wav where
    # Snn gives speaker code nn, Unnn gives utterance number nnn
    # grouping by utterance number - all utterance must have same speaker sets
    utterances: Dict[int, List[Tuple[str, int]]] = {}
    for file in file_list:
        utterance_no, speaker_code = parse_wave_file_name(file)

        if utterance_no in utterances:
            utterances[utterance_no].append((file, speaker_code))
        else:
            utterances[utterance_no] = [(file, speaker_code)]
    # validate if utterances have same speakers
    if same_speakers:
        # yes we could just group and make intersection on all sets
        speaker_sets = [set([t[1] for t in u]) for u in utterances.values()]
        intersection = reduce(lambda x, y: x.intersection(y), speaker_sets, speaker_sets[0])
        if intersection != speaker_sets[0]:
            difference = speaker_sets[0].symmetric_difference(intersection)
            logging.error(f"not all speaker sets are equal {difference}")
            raise ValueError(difference)

    return list(utterances.values())


@background(max_prefetch=1)
def train_generator(wav_sets, sample_rate, receptive_field, batch_length, batch_size,
                    max_batches_per_file=None,
                    wav_transform=None,
                    shuffle=True):
    """GENERATE TRAINING BATCH.

    Args:
        wav_sets (list(list(tuple)): List of sets of wav files grouped into utterances with same speaker sets.
        receptive_field (int): Size of receptive filed.
        batch_length (int): Batch length (if set None, utterance batch will be used.).
        batch_size (int): Batch size (if batch_length = None, batch_size will be 1.).
        max_batches_per_file (int):  maximum batches sampled from a single file
        wav_transform (func): Preprocessing function for waveform.
        shuffle (bool): Whether to shuffle the file list and samples

    Returns:
        generator: Generator instance.

    """
    # shuffle list
    def shuffle_set():
        n_files = len(wav_sets)
        idx = np.random.permutation(n_files)
        return [wav_sets[i] for i in idx]

    if shuffle:
        wav_sets = shuffle_set()

    if batch_length is None:
        raise NotImplemented("utterance mode not supported")

    speaker_codes = {}

    def get_speaker_features(speaker_code):
        return speaker_codes.setdefault(speaker_code,
                                        create_speaker_code_feature(speaker_code, batch_length, receptive_field))

    while True:
        # process over all of files
        for wav_set in wav_sets:
            logging.debug(f"processing utterance set with {len(wav_set)} speakers")
            loaded_waves = []
            for wavfile, speaker_code in wav_set:
                x, _rate = sf.read(wavfile, dtype=np.float32)
                assert sample_rate == _rate, f"expected sample rate is {sample_rate}, {wavfile} is {_rate}"
                h_ = get_speaker_features(speaker_code)
                epoch_len = (len(x) - receptive_field - batch_length) // batch_length
                logging.debug(f"loading {wavfile} with speaker code {speaker_code} and {epoch_len} batches")
                loaded_waves.append((x, h_, epoch_len))

            # use minimum epoch_len. we assume that same utterances from different speakers are more or less same size
            min_epoch_len = min(t[2] for t in loaded_waves)

            # make array of batches
            file_epoch_range = np.arange(min_epoch_len)
            if shuffle:
                np.random.shuffle(file_epoch_range)
            # limit epoch range of a single file
            if max_batches_per_file:
                logging.debug(f"limiting batches from {len(file_epoch_range)} to {max_batches_per_file}")
                file_epoch_range = file_epoch_range[:max_batches_per_file]

            def sample_speaker():
                return loaded_waves[np.random.randint(len(loaded_waves))]

            # new set - new batch, that discards incomplete batch leftover
            batch_x, batch_h, batch_t = [], [], []
            x, h_, _ = sample_speaker()

            for idx in file_epoch_range:
                # get pieces
                start_sample = idx * batch_length
                x_ = x[start_sample:start_sample + receptive_field + batch_length]

                # perform pre-processing
                if wav_transform is not None:
                    x_ = wav_transform(x_)

                # convert to torch variable
                x_ = torch.from_numpy(x_).long()

                # remove the last and first sample for training
                batch_x += [x_[:-1]]  # (T)
                batch_h += [h_]  # (D x T)
                batch_t += [x_[1:]]  # (T)

                # return mini batch
                if len(batch_x) == batch_size:
                    batch_x = torch.stack(batch_x)
                    batch_h = torch.stack(batch_h)
                    batch_t = torch.stack(batch_t)

                    # send to cuda
                    if torch.cuda.is_available():
                        batch_x = batch_x.cuda()
                        batch_h = batch_h.cuda()
                        batch_t = batch_t.cuda()

                    yield (batch_x, batch_h), batch_t

                    # new batch - new speaker
                    batch_x, batch_h, batch_t = [], [], []
                    x, h_, _ = sample_speaker()

        # re-shuffle
        if shuffle:
            wav_sets = shuffle_set()


def save_checkpoint(checkpoint_dir, model, optimizer, iterations):
    """SAVE CHECKPOINT.

    Args:
        checkpoint_dir (str): Directory to save checkpoint.
        model (torch.nn.Module): Pytorch model instance.
        optimizer (torch.optim.optimizer): Pytorch optimizer instance.
        iterations (int): Number of current iterations.

    """
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iterations": iterations}
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    torch.save(checkpoint, checkpoint_dir + "/checkpoint-%d.pkl" % iterations)
    logging.info("%d-iter checkpoint created." % iterations)


def main():
    """RUN TRAINING."""
    parser = argparse.ArgumentParser()
    # path setting
    parser.add_argument("--waveforms", required=True,
                        type=str, help="directory or list of wav files")
    parser.add_argument("--fs", default=16000,
                        type=int, help="sampling rate")
    parser.add_argument("--expdir", required=True,
                        type=str, help="directory to save the model")
    # network structure setting
    parser.add_argument("--n_quantize", default=256,
                        type=int, help="number of quantization")
    parser.add_argument("--n_resch", default=512,
                        type=int, help="number of channels of residual output")
    parser.add_argument("--n_skipch", default=256,
                        type=int, help="number of channels of skip output")
    parser.add_argument("--dilation_depth", default=10,
                        type=int, help="depth of dilation")
    parser.add_argument("--dilation_repeat", default=1,
                        type=int, help="number of repeating of dilation")
    parser.add_argument("--kernel_size", default=2,
                        type=int, help="kernel size of dilated causal convolution")
    parser.add_argument("--speaker_code",
                        type=int, help="speaker code")
    # network training setting
    parser.add_argument("--lr", default=1e-4,
                        type=float, help="learning rate")
    parser.add_argument("--weight_decay", default=0.0,
                        type=float, help="weight decay coefficient")
    parser.add_argument("--batch_length", default=20000,
                        type=int, help="batch length (if set 0, utterance batch will be used)")
    parser.add_argument("--batch_size", default=1,
                        type=int, help="batch size (if use utterance batch, batch_size will be 1.")
    parser.add_argument("--max_batches_per_file",
                        type=int, help="Maximum batches generated from a single wav file.")
    parser.add_argument("--iters", default=200000,
                        type=int, help="number of iterations")
    # other setting
    parser.add_argument("--checkpoint_interval", default=10000,
                        type=int, help="how frequent saving model")
    parser.add_argument("--intervals", default=100,
                        type=int, help="log interval")
    parser.add_argument("--seed", default=1,
                        type=int, help="seed number")
    parser.add_argument("--resume", default=None, nargs="?",
                        type=str, help="model path to restart training")
    parser.add_argument("--n_gpus", default=1,
                        type=int, help="number of gpus")
    parser.add_argument("--verbose", default=1,
                        type=int, help="log level")
    args = parser.parse_args()

    # set log level
    if args.verbose == 1:
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

    # make experimental directory
    if not os.path.exists(args.expdir):
        os.makedirs(args.expdir)

    # fix seed
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # fix slow computation of dilated conv
    # https://github.com/pytorch/pytorch/issues/15054#issuecomment-450191923
    torch.backends.cudnn.benchmark = True

    # save args as conf
    torch.save(args, args.expdir + "/model.conf")

    # define network
    model = WaveNet(
        n_quantize=args.n_quantize,
        n_aux=1,  # speaker code
        n_resch=args.n_resch,
        n_skipch=args.n_skipch,
        dilation_depth=args.dilation_depth,
        dilation_repeat=args.dilation_repeat,
        kernel_size=args.kernel_size)
    logging.info(model)
    model.apply(initialize)
    model.train()

    if args.n_gpus > 1:
        device_ids = range(args.n_gpus)
        model = torch.nn.DataParallel(model, device_ids)
        model.receptive_field = model.module.receptive_field
        if args.n_gpus > args.batch_size:
            logging.warning("batch size is less than number of gpus.")

    # define optimizer and loss
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    # define transforms
    wav_transform = transforms.Compose([
        lambda x: encode_mu_law(x, args.n_quantize)])

    # define generator
    if os.path.isdir(args.waveforms):
        filenames = sorted(find_files(args.waveforms, "*.wav", use_dir_name=False))
        wav_list = [args.waveforms + "/" + filename for filename in filenames]
    elif os.path.isfile(args.waveforms):
        wav_list = read_txt(args.waveforms)
    else:
        logging.error("--waveforms should be directory or list.")
        sys.exit(1)

    if args.speaker_code is None:
        wav_set = load_speaker_code_set(wav_list, same_speakers=True)
    else:
        wav_set = load_flat_wav_set(wav_list, args.speaker_code)

    logging.info("number of training sets = %d with %d files." % (len(wav_set), len(wav_list)))
    generator = train_generator(
        wav_set,
        args.fs,
        model.receptive_field,
        args.batch_length,
        args.batch_size,
        max_batches_per_file=args.max_batches_per_file,
        wav_transform=wav_transform,
        shuffle=True)

    # charge minibatch in queue
    while not generator.queue.full():
        time.sleep(0.1)

    # resume model and optimizer
    if args.resume is not None and len(args.resume) != 0:
        checkpoint = torch.load(args.resume, map_location=lambda storage, loc: storage)
        iterations = checkpoint["iterations"]
        if args.n_gpus > 1:
            model.module.load_state_dict(checkpoint["model"])
        else:
            model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        logging.info("restored from %d-iter checkpoint." % iterations)
    else:
        iterations = 0

    # check gpu and then send to gpu
    if torch.cuda.is_available():
        model.cuda()
        criterion.cuda()
        for state in optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.cuda()
    else:
        logging.error("gpu is not available. please check the setting. GOD SPED")

    # train
    loss = 0
    total = 0
    writer = SummaryWriter()

    for i in six.moves.range(iterations, args.iters):
        start = time.time()
        (batch_x, batch_h), batch_t = generator.next()
        batch_output = model(batch_x, batch_h)
        batch_loss = criterion(
            batch_output[:, model.receptive_field:].contiguous().view(-1, args.n_quantize),
            batch_t[:, model.receptive_field:].contiguous().view(-1))
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()
        loss += batch_loss.item()
        total += time.time() - start
        logging.debug("batch loss = %.3f (%.3f sec / batch) shape x %s h %s t %s" % (
            batch_loss.item(), time.time() - start, str(batch_x.shape), str(batch_h.shape), str(batch_t.shape)))

        # report progress
        if (i + 1) % args.intervals == 0:
            # log
            logging.info("(iter:%d) average loss = %.6f (%.3f sec / batch)" % (
                i + 1, loss / args.intervals, total / args.intervals))
            logging.info("estimated required time = "
                         "{0.days:02}:{0.hours:02}:{0.minutes:02}:{0.seconds:02}"
                         .format(relativedelta(
                             seconds=int((args.iters - (i + 1)) * (total / args.intervals)))))
            # write summary
            writer.add_scalar("Loss", loss / args.intervals, i)
            writer.add_scalar("Batch Time", total / args.intervals, i)

            source_sample = decode_mu_law(batch_x.cpu().numpy().flatten('C'), args.n_quantize)
            writer.add_audio("Input Sample", source_sample, i, args.fs)
            writer.flush()

            loss = 0
            total = 0

        # save intermediate model
        if (i + 1) % args.checkpoint_interval == 0:
            if args.n_gpus > 1:
                save_checkpoint(args.expdir, model.module, optimizer, i + 1)
            else:
                save_checkpoint(args.expdir, model, optimizer, i + 1)

    # save final model
    if args.n_gpus > 1:
        torch.save({"model": model.module.state_dict()}, args.expdir + "/checkpoint-final.pkl")
    else:
        torch.save({"model": model.state_dict()}, args.expdir + "/checkpoint-final.pkl")
    logging.info("final checkpoint created.")


if __name__ == "__main__":
    main()
