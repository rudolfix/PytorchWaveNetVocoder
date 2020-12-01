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
from wavenet_vocoder.utils import find_files
from wavenet_vocoder.utils import read_txt


def validate_length(x, y, upsampling_factor=None):
    """VALIDATE LENGTH.

    Args:
        x (ndarray): ndarray with x.shape[0] = len_x.
        y (ndarray): ndarray with y.shape[0] = len_y.
        upsampling_factor (int): Upsampling factor.

    Returns:
        ndarray: Length adjusted x with same length y.
        ndarray: Length adjusted y with same length x.

    """
    if upsampling_factor is None:
        if x.shape[0] < y.shape[0]:
            y = y[:x.shape[0]]
        if x.shape[0] > y.shape[0]:
            x = x[:y.shape[0]]
        assert len(x) == len(y)
    else:
        if x.shape[0] > y.shape[0] * upsampling_factor:
            x = x[:y.shape[0] * upsampling_factor]
        if x.shape[0] < y.shape[0] * upsampling_factor:
            mod_y = y.shape[0] * upsampling_factor - x.shape[0]
            mod_y_frame = mod_y // upsampling_factor + 1
            y = y[:-mod_y_frame]
            x = x[:y.shape[0] * upsampling_factor]
        assert len(x) == len(y) * upsampling_factor

    return x, y


@background(max_prefetch=16)
def train_generator(wav_list, sample_rate, receptive_field,
                    batch_length=None,
                    batch_size=1,
                    wav_transform=None,
                    feat_transform=None,
                    shuffle=True,
                    speaker_code=0):
    """GENERATE TRAINING BATCH.

    Args:
        wav_list (list): List of wav files.
        receptive_field (int): Size of receptive filed.
        batch_length (int): Batch length (if set None, utterance batch will be used.).
        batch_size (int): Batch size (if batch_length = None, batch_size will be 1.).
        wav_transform (func): Preprocessing function for waveform.
        shuffle (bool): Whether to shuffle the file list and samples
        speaker_code (bool): Pass speaker code

    Returns:
        generator: Generator instance.

    """
    # shuffle list
    if shuffle:
        n_files = len(wav_list)
        idx = np.random.permutation(n_files)
        wav_list = [wav_list[i] for i in idx]

    # show warning
    if batch_length is None and batch_size > 1:
        logging.warning("in utterance batch mode, batchsize will be 1.")

    while True:
        batch_x, batch_h, batch_t = [], [], []
        # process over all of files
        for wavfile in wav_list:
            # load waveform and aux feature
            x, _rate = sf.read(wavfile, dtype=np.float32)
            assert sample_rate == _rate, f"expected sample rate is {sample_rate}, {wavfile} is {_rate}"
            # empty feature file
            h = np.empty((x.shape[0], 0), dtype=np.float32)
            # h = extend_time(h, upsampling_factor)
            # speaker code is mandatory
            sc = np.tile(speaker_code, [h.shape[0], 1])
            h = np.concatenate([h, sc], axis=1)

            # check both lengths are same
            logging.debug("before x length = %d" % x.shape[0])
            logging.debug("before h length = %d" % h.shape[0])
            x, h = validate_length(x, h)
            logging.debug("after x length = %d" % x.shape[0])
            logging.debug("after h length = %d" % h.shape[0])

            # ---------------------------------------
            # use mini batch without upsampling layer
            # ---------------------------------------
            if batch_length is not None:
                # make buffer array
                if "x_buffer" not in locals():
                    x_buffer = np.empty((0), dtype=np.float32)
                    h_buffer = np.empty((0, h.shape[1]), dtype=np.float32)
                x_buffer = np.concatenate([x_buffer, x], axis=0)
                h_buffer = np.concatenate([h_buffer, h], axis=0)

                epoch_range = np.arange((len(x) - receptive_field - batch_length) // batch_length)
                if shuffle:
                    np.random.shuffle(epoch_range)

                for idx in epoch_range:
                    # get pieces
                    start_sample = idx * batch_length
                    x_ = x_buffer[start_sample:start_sample + receptive_field + batch_length]
                    h_ = h_buffer[start_sample:start_sample + receptive_field + batch_length]

                    # perform pre-processing
                    if wav_transform is not None:
                        x_ = wav_transform(x_)

                    # convert to torch variable
                    x_ = torch.from_numpy(x_).long()
                    h_ = torch.from_numpy(h_).float()

                    # remove the last and first sample for training
                    batch_x += [x_[:-1]]  # (T)
                    batch_h += [h_[:-1].transpose(0, 1)]  # (D x T)
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

                        batch_x, batch_h, batch_t = [], [], []

            # --------------------------------------------
            # use utterance batch without upsampling layer
            # --------------------------------------------
            elif batch_length is None:
                # perform pre-processing
                if wav_transform is not None:
                    x = wav_transform(x)

                # convert to torch variable
                x = torch.from_numpy(x).long()
                h = torch.from_numpy(h).float()

                # remove the last and first sample for training
                batch_x = x[:-1].unsqueeze(0)  # (1 x T)
                batch_h = h[:-1].transpose(0, 1).unsqueeze(0)  # (1 x D x T)
                batch_t = x[1:].unsqueeze(0)  # (1 x T)

                # send to cuda
                if torch.cuda.is_available():
                    batch_x = batch_x.cuda()
                    batch_h = batch_h.cuda()
                    batch_t = batch_t.cuda()

                yield (batch_x, batch_h), batch_t

        # re-shuffle
        if shuffle:
            idx = np.random.permutation(n_files)
            wav_list = [wav_list[i] for i in idx]


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
    parser.add_argument("--speaker_code", default=0,
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

    logging.info("number of training data = %d." % len(wav_list))
    generator = train_generator(
        wav_list,
        args.fs,
        receptive_field=model.receptive_field,
        batch_length=args.batch_length,
        batch_size=args.batch_size,
        wav_transform=wav_transform,
        shuffle=True,
        speaker_code=args.speaker_code)

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
            writer.add_scalar("Batch Time", total / args.intervals / args.intervals, i)

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
