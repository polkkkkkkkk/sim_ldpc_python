"""LDPC codec adapter for the simplified rate-matching simulation path."""

import os
import sys

from codec_impl import BinarySoftCodecBase
from simulator_rm_awgn_python.tools import load_json
from ldpc_soft_py.bin_ldpc_soft import BinLdpcSoftDecoder


class BinaryLdpcRmSoftCodec(BinarySoftCodecBase):
    """Binary LDPC codec with precomputed sequential rate matching metadata."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        config = load_json(os.path.join(kwargs['src_dir'], kwargs['code']))
        self.punctured_tail = int(config.get('punctured_tail', 0))
        self.max_multiplicity = int(config.get('max_multiplicity', 1))
        self.last_max_multiplicity_index = int(
            config.get('last_max_multiplicity_index', -1)
        )
        self._validate_rate_matching()

        self.decoder_impl, self.dec_fcn = self.get_decoder_instance(
            BinLdpcSoftDecoder
        )

    def _validate_rate_matching(self):
        block_length = self.block_length()
        if not 0 <= self.punctured < block_length:
            raise ValueError('punctured is outside the codeword')
        if not 0 <= self.punctured_tail <= block_length - self.punctured:
            raise ValueError('punctured_tail is outside the circular buffer')
        if self.max_multiplicity < 1:
            raise ValueError('max_multiplicity must be at least 1')

        channel_stop = block_length - self.punctured_tail
        if channel_stop <= self.punctured:
            raise ValueError('Rate matching leaves no bits for the channel')
        if not self.punctured <= self.last_max_multiplicity_index < channel_stop:
            raise ValueError('last_max_multiplicity_index is outside transmitted bits')
        if self.max_multiplicity > 1 and self.punctured_tail:
            raise ValueError('Repetition and tail puncturing cannot be combined')

    @property
    def channel_start(self):
        return self.punctured

    @property
    def channel_stop(self):
        return self.block_length() - self.punctured_tail

    @property
    def last_repeat_index(self):
        if self.max_multiplicity == 1:
            return -1
        return self.last_max_multiplicity_index - self.channel_start

    @property
    def number_repetitions(self):
        return self.max_multiplicity

    def get_inf_bits_count(self):
        return self.pcm_shape[1] - self.pcm_shape[0]

    def get_transmitted_bits_count(self):
        circular_buffer_length = self.block_length() - self.punctured
        max_prefix_length = (
            self.last_max_multiplicity_index - self.punctured + 1
        )
        return (
            circular_buffer_length * (self.max_multiplicity - 1)
            + max_prefix_length
        )

    def decode(self, llr_in, llr_out):
        return self.dec_fcn(llr_in, llr_out)

    def get_block_length_str(self):
        block_length = self.get_transmitted_bits_count()
        msg = f'{block_length} (simplified rate matching)'
        return msg, block_length

    def get_filename_template(self):
        return 'bin_ldpc_rm_' + super().get_filename_template()

    def get_title_template(self):
        return 'Binary LDPC RM, ' + super().get_title_template()

    def __str__(self):
        msg = 'Binary LDPC code with simplified rate matching\n'
        msg += super().__str__() + '\n'
        msg += 'Rate-matching parameters:\n'
        msg += f'  Punctured prefix:          {self.punctured}\n'
        msg += f'  Punctured tail:            {self.punctured_tail}\n'
        msg += f'  Maximum multiplicity:      {self.max_multiplicity}\n'
        msg += (
            '  Last maximum-multiplicity '
            f'index: {self.last_max_multiplicity_index}'
        )
        return msg


def instantiate_codec(**kwargs):
    """Instantiate an RM codec from experiment settings."""
    codec_type = kwargs.pop('type')
    class_type = getattr(sys.modules[__name__], codec_type)
    return class_type(**kwargs)
