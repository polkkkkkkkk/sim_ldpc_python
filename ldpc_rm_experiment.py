"""LDPC simulation workflow with simplified rate matching."""

import dataclasses
import os

import numpy as np

from codec_rm_impl import instantiate_codec
from simulator_rm_awgn_python.channel import AwgnQAMChannel
from simulator_rm_awgn_python.data_storage import DataEntry
from simulator_rm_awgn_python.settings import HLINE_STR
from simulator_rm_awgn_python.tools import dir_exists


@dataclasses.dataclass
class LdpcRmDataEntry(DataEntry):
    """Simulation result with decoder iteration statistics."""

    n_iter: int
    iter_pdf: np.array

    def __str__(self):
        msg = super().__str__()
        avg_iter = self.n_iter / self.tests if self.tests else 0
        return msg + f', {avg_iter:1.3f} avg. iterations'


@dataclasses.dataclass
class LdpcRmExperimentSettings:
    """Rate-matched LDPC experiment parameters."""

    modulation: str
    channel_output: str
    codec: dict
    data_dir: str = 'data'
    filename: str = ''
    title: str = ''
    inf_bits_count: int = 0
    codec_info: str = ''

    def __post_init__(self):
        dir_exists(self.data_dir)
        if self.channel_output not in ('soft', 'hard'):
            raise ValueError(
                f'Channel output {self.channel_output} is not supported'
            )
        codec_instance = instantiate_codec(**self.codec)
        self.filename = os.path.join(
            self.data_dir,
            codec_instance.get_filename_template()
            + '_' + self.modulation + '_' + self.channel_output + '.pickle'
        )
        self.title = (
            codec_instance.get_title_template()
            + ', modulation ' + self.modulation + '-' + self.channel_output
        )
        self.codec_info = str(codec_instance)
        self.inf_bits_count = codec_instance.get_inf_bits_count()

    def __str__(self):
        msg = HLINE_STR + '\n'
        msg += 'Channel parameters:\n'
        msg += f'  Modulation:                        {self.modulation.upper()}\n'
        msg += f'  Channel output:                    {self.channel_output}\n'
        msg += HLINE_STR + '\n' + self.codec_info + '\n' + HLINE_STR + '\n'
        return msg + f'Output filename: {self.filename}'


class LdpcRmExperimentInstance:
    """Run one LDPC experiment using precomputed RM metadata."""

    def __init__(self, settings):
        self.settings = settings
        self.codec = instantiate_codec(**settings.codec)

        block_length = self.codec.block_length()
        dtype = getattr(np, self.codec.llr_type)
        self.tx_bits = np.zeros(block_length, dtype=np.uint8)
        if not self.codec.is_azcw():
            self.iwd = np.zeros(
                self.codec.get_inf_bits_count(),
                dtype=np.uint8
            )
        self.llr_in = np.zeros(block_length, dtype=dtype)
        self.llr_out = np.zeros(block_length, dtype=dtype)

        channel_slice = slice(
            self.codec.channel_start,
            self.codec.channel_stop
        )
        self.channel_llr = self.llr_in[channel_slice]
        self.channel = AwgnQAMChannel(
            self.settings.modulation,
            self.tx_bits[channel_slice],
            self.channel_llr,
            self.codec.is_azcw(),
            self.codec.last_repeat_index,
            self.codec.number_repetitions
        )
        self.is_channel_hard = self.settings.channel_output == 'hard'

    def run_channel(self, snr_db, rng):
        in_ber, in_ser = self.channel.run(snr_db, rng)
        if self.is_channel_hard:
            self.channel_llr[:] = np.sign(self.channel_llr)
        return in_ber, in_ser

    def run(self, snr_db, rng):
        if not self.codec.is_azcw():
            self.codec.generate(rng, self.iwd, self.tx_bits)
        in_ber, in_ser = self.run_channel(snr_db, rng)
        n_iter = self.codec.decode(self.llr_in, self.llr_out)
        out_ber = self.output_ber(n_iter)
        return LdpcRmDataEntry(
            in_be_cum=in_ber,
            in_se_cum=in_ser,
            be_cum=out_ber,
            fe_cum=out_ber > 0,
            n_iter=n_iter,
            iter_pdf=self.one_hot(n_iter),
            tests=1
        )

    def output_ber(self, n_iter):
        if self.codec.is_azcw() and n_iter < self.codec.n_iterations:
            return 0.0
        return self.codec.decoder_impl.output_ber(
            self.llr_out,
            self.tx_bits,
            n_iter
        )

    def one_hot(self, n_iter):
        vec = np.zeros(self.codec.n_iterations + 1, dtype=np.int32)
        vec[n_iter] = 1
        return vec
