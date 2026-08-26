"""
LDPC codecs implementation
"""

import os
import sys

import numpy as np

from simulator_awgn_python.tools import load_json
from simulator_awgn_python.channel import random_bits

from lbc_encoder.lbc_encoder import LBCEncoder

from ldpc_soft_py.bin_ldpc_soft import Alist
from ldpc_soft_py.bin_ldpc_soft import BinLdpcSoftDecoder, BinGldpcSoftDecoder


class BinaryCodecBase:
    """
    Binary (G)LDPC codec base class. To specify the codec,
    the following parameters must be provided:
      - Source directory containing a general code data
      - Path to json file (relative to the source directory) that contains:
        * file specifying the parity check matrix (alist format), obligatory parameter
        * file specifying generator matrix (txt format, loaded by np.loadtxt(). If not specified,
          the simulation will be performed on all-zero codewords (AZCW)
    Coding rate will be calculated in accordance with shortened positions.
    """
    def __init__(self, **kwargs):
        src_dir = kwargs.get('src_dir')
        code_filename = kwargs.get('code')
        if not (isinstance(src_dir, str) and isinstance(code_filename, str)):
            raise ValueError('Please specify \'src_dir\' and \'code\' parameters')
        config = load_json(os.path.join(src_dir, code_filename))
        # Load generator and parity check matrices

        self.is_nr = 'payload_length' in config

        # Get a unique name for a dedicated filename
        self.name = config.get('name', '')

        # Punctured positions: always assumed to be first
        self.punctured = int(
            config.get('punctured', config.get('punctured_prefix', 0))
        )

        # For systematic code, evaluate FER using the first k positions
        # Use a whole codeword otherwise
        is_systematic = config.get('is_systematic', False)
        if isinstance(is_systematic, str):
            is_systematic = is_systematic.lower() == 'true'
        self.is_systematic = bool(is_systematic)

        # Read the shape of the parity check matrix
        self.alist_path = os.path.join(src_dir, config.get('pcm'))
        self.pcm_shape = Alist.read_shape(self.alist_path)

        if self.is_nr:
            self.base_graph = int(config['base_graph'])
            self.lifting_size = int(config['lifting_size'])
            self.payload_length = int(config['payload_length'])
            self.systematic_length = int(config['systematic_length'])
            self.mother_length = int(config['mother_length'])
            self.punctured_prefix = int(
                config.get('punctured_prefix', self.punctured)
            )
            self.rm_input_length = int(
                config.get(
                    'rm_input_length',
                    self.mother_length - self.punctured_prefix
                )
            )
            self.filler_mask = np.asarray(config['filler_mask'], dtype=bool)
            self.payload_indices = np.arange(
                self.payload_length,
                dtype=np.int64
            )
            if self.payload_length <= 0:
                raise ValueError('Payload length must be positive')
            if self.base_graph not in (1, 2) or self.lifting_size <= 0:
                raise ValueError('Invalid NR base graph parameters')
            if not self.payload_length <= self.systematic_length <= self.mother_length:
                raise ValueError('Invalid NR code lengths')
            if self.mother_length != self.pcm_shape[1]:
                raise ValueError('NR mother length / parity check matrix mismatch')
            expected_mother_length = (
                68 if self.base_graph == 1 else 52
            ) * self.lifting_size
            expected_systematic_length = (
                22 if self.base_graph == 1 else 10
            ) * self.lifting_size
            expected_prefix = 2 * self.lifting_size
            if self.mother_length != expected_mother_length:
                raise ValueError('NR mother length / BG and Zc mismatch')
            if self.systematic_length != expected_systematic_length:
                raise ValueError('NR systematic length / BG and Zc mismatch')
            if self.punctured_prefix != expected_prefix:
                raise ValueError('NR punctured prefix / Zc mismatch')
            if self.punctured != self.punctured_prefix:
                raise ValueError('NR punctured metadata mismatch')
            if self.rm_input_length != self.mother_length - self.punctured_prefix:
                raise ValueError('NR rate-matching input length mismatch')
            if self.filler_mask.shape != (self.mother_length,):
                raise ValueError('NR filler mask / mother length mismatch')
            expected_filler = np.zeros(self.mother_length, dtype=bool)
            expected_filler[self.payload_length:self.systematic_length] = True
            if not np.array_equal(self.filler_mask, expected_filler):
                raise ValueError('NR filler mask has unexpected positions')
            filler = config.get('filler', {})
            if (
                int(filler.get('start', self.payload_length)) != self.payload_length
                or int(filler.get('stop', self.systematic_length)) != self.systematic_length
            ):
                raise ValueError('NR filler metadata has unexpected range')

        # Read the generator matrix and initialize the encoder:
        generator_path = config.get('generator', None)
        if self.is_nr and generator_path is None:
            raise ValueError('NR code metadata requires a generator matrix')
        if generator_path is not None:
            self.encoder = LBCEncoder(os.path.join(src_dir, generator_path))
            if self.pcm_shape[1] != self.encoder.cwd_length:
                raise ValueError('Generator / parity check matrices shape mismatch')
            if self.is_nr and self.encoder.inf_bits != self.systematic_length:
                raise ValueError('NR generator / systematic length mismatch')

        self.decoder_impl = None  # To be instantiated by subclass

    def generate(self, rng, iwd, cwd):
        """
        Generate the information word and codeword
        return: codeword
        """
        if not self.is_nr:
            random_bits(iwd, rng)
            self.encoder.encode(iwd, cwd)
            return

        if iwd.shape != (self.payload_length,):
            raise ValueError('NR payload buffer has an unexpected length')
        if cwd.shape != (self.mother_length,):
            raise ValueError('NR codeword buffer has an unexpected length')

        random_bits(iwd, rng)
        systematic = np.zeros(self.systematic_length, dtype=np.uint8)
        systematic[:self.payload_length] = iwd
        self.encoder.encode(systematic, cwd)

    def get_payload_length(self):
        """Return the number of payload bits used for BER evaluation."""
        if self.is_nr:
            return self.payload_length
        return self.get_inf_bits_count()

    def get_payload_indices(self):
        """Return payload positions in the systematic codeword."""
        if self.is_nr:
            return self.payload_indices.copy()
        return np.arange(self.get_payload_length(), dtype=np.int64)

    def block_length(self):
        """
        Get a full codeword length (including punctured bits)
        """
        return self.pcm_shape[1]

    def get_inf_bits_count(self):
        """
        Get the information bit count for correct coding rate evaluation.
        Must be implemented by a subclass,
        information bits are calculated differently for LDPC and GLDPC
        """
        raise NotImplementedError('Must be implemented by subclass')

    def decode(self, llr_in, llr_out):
        """
        Run decoder implementation
        :param llr_in input LLR vector with puncturing being applied (if needed)
        :param llr_out output LLR vector placeholder
        :return The number of decoding iterations
        """
        raise NotImplementedError('Must be implemented by subclass')

    def get_filename_template(self):
        """
        A unique filename template that is generated from code parameters
        """
        # Encode information bit count and block length into the filename template
        if self.name:
            template = self.name + '_'
        else:
            template = ''
        return template + f'k{self.get_inf_bits_count()}_n{self.get_block_length_str()[1]}_'

    def get_title_template(self):
        """
        Get title string template for live-plot
        """
        return f'k = {self.get_inf_bits_count()}, n = {self.get_block_length_str()[0]}'

    def get_block_length_str(self):
        """
        Block length string and value with punctured/shortened bits taken into account
        """
        block_len_val = int(self.pcm_shape[1]) - self.punctured
        msg = f'{block_len_val}'
        if self.punctured:
            msg += f' (first {self.punctured} are punctured)'

        return msg, block_len_val

    def __str__(self):
        msg = 'Code parameters:\n'
        msg += f'  Alist path:               \'{self.alist_path}\'\n'
        msg += f'  Parity check matrix shape: {self.pcm_shape[0]} x {self.pcm_shape[1]}\n'
        msg += '  Generator matrix path:    '

        if hasattr(self, 'encoder'):
            generator_path = self.encoder.generator_path
            msg += f'\'{generator_path}\'\n'
            msg += '  Generator matrix shape:    '
            msg += f'{self.encoder.cwd_length} x {self.encoder.inf_bits}\n'
        else:
            msg += ' None. All-zero-codewords simulation\n'

        block_len_str, block_len_val = self.get_block_length_str()
        msg += f'  Block length:              {block_len_str} bits\n'
        msg += f'  Information bits count:    {self.get_inf_bits_count()}\n'
        msg += f'  Coding rate:               {self.get_inf_bits_count() / block_len_val:1.4f}\n'
        msg += '  Estimating FER by:         '
        if self.is_systematic:
            msg += 'first positions (systematic code)'
        else:
            msg += 'a whole codeword'
        return msg


class BinarySoftCodecBase(BinaryCodecBase):
    """
    Binary (G)LDPC codec with soft decoding algorithm.
    The following additional parameters must be provided:
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.llr_type = kwargs.get('llr_type')
        self.n_iterations = kwargs.get('n_iterations')
        self.llr_scale = kwargs.get('llr_scale')
        self.algorithm = kwargs.get('algorithm')
        # Sanity checks
        if not isinstance(self.llr_scale, float) or self.llr_scale < 0:
            raise ValueError('LLR scale must be non-negative')
        if not isinstance(self.n_iterations, int) or self.n_iterations <= 0:
            raise ValueError('The number of decoding iterations must be positive')
        if self.llr_type not in ('float32', 'float64'):
            raise ValueError(f'LLR type {self.llr_type} is not supported')


    def get_inf_bits_count(self):
        """
        Get the information bit count for correct coding rate evaluation.
        Must be implemented by a subclass,
        information bits are calculated differently for LDPC and GLDPC
        """
        raise NotImplementedError('Must be implemented by subclass')

    def decode(self, llr_in, llr_out):
        """
        Run decoder implementation
        """
        raise NotImplementedError('Must be implemented by subclass')

    def get_filename_template(self):
        """
        A unique filename template that is generated from code parameters
        """
        # Encode information bit count and block length into the filename template
        template = super().get_filename_template()
        template += f'{self.algorithm}_iter_{self.n_iterations}_llr_{self.llr_type}'
        if self.algorithm != 'sum_product':
            template += f'_scale_{self.llr_scale:1.3f}'
        return template

    def get_title_template(self):
        """
        Get title string template for live-plot
        """
        return super().get_title_template() + f', {self.algorithm}, {self.n_iterations} iterations'

    def get_decoder_instance(self, dec_type):
        """
        Instantiate particular decoder implementation.
        Pass all fields required to initialize decoding settings structure
        """
        dec_instance = dec_type(
            self.alist_path,
            block_length=self.pcm_shape[1],
            n_checks=self.pcm_shape[0],
            llr_type=self.llr_type,
            llr_scale=self.llr_scale,
            n_iterations=self.n_iterations,
            is_systematic=self.is_systematic,
            is_azcw=self.is_azcw()
            )
        try:
            dec_fcn = getattr(dec_instance, self.algorithm)
        except AttributeError as exc:
            raise ValueError(
                f'Decoding algorithm {self.algorithm} ' +
                f'is not supported by {self.__class__.__name__}'
            ) from exc
        return dec_instance, dec_fcn

    def is_azcw(self):
        """
        If there is no encoder specified, the simulation is run using all-zeros-codewords
        """
        return not hasattr(self, 'encoder')

    def __str__(self):
        msg = super().__str__() + '\n'
        msg += 'Decoder-specific parameters:\n'
        msg += f'  Decoding algorithm:       \'{self.algorithm}\'\n'
        msg += f'  The number of iterations:  {self.n_iterations}\n'
        msg += f'  LLR type:                 \'{self.llr_type}\'\n'
        if self.algorithm != 'sum_product':
            msg += f'  LLR scale:                 {self.llr_scale:1.3f}'
        return msg


class BinaryGldpcSoftCodec(BinarySoftCodecBase):
    """
    Binary GLDPC codec with soft decoding algorithm
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.decoder_impl, self.dec_fcn = self.get_decoder_instance(BinGldpcSoftDecoder)
        if hasattr(self, 'encoder'):
            raise ValueError(f'Encoding function is not supported by {self.__class__.__name__}')

    def get_inf_bits_count(self):
        """
        Get the information bit count for correct coding rate evaluation.
        Warning: parity check matrix is assumed to be of full rank
        """
        return self.pcm_shape[1] - 2 * self.pcm_shape[0]

    def decode(self, llr_in, llr_out):
        """
        Decoding function implementation
        """
        return self.dec_fcn(llr_in, llr_out)

    def get_filename_template(self):
        """
        A unique filename template that is generated from code parameters
        """
        # Encode information bit count and block length into the filename template
        return 'bin_gldpc_' + super().get_filename_template()

    def get_title_template(self):
        """
        Get title string template for live-plot
        """
        return 'Binary GLDPC, ' + super().get_title_template()

    def __str__(self):
        msg = 'Binary GLDPC (Cordaro-Wagner ext.) code with soft decoding algorithm\n'
        return msg + super().__str__()


class BinaryLdpcSoftCodec(BinarySoftCodecBase):
    """
    Binary LDPC codec with soft decoding algorithm
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.decoder_impl, self.dec_fcn = self.get_decoder_instance(BinLdpcSoftDecoder)

    def get_inf_bits_count(self):
        """
        Get the information bit count for correct coding rate evaluation
        Warning: parity check matrix is assumed to be of full rank
        """
        if self.is_nr:
            return self.payload_length
        return self.pcm_shape[1] - self.pcm_shape[0]

    def decode(self, llr_in, llr_out):
        """
        Decoding function implementation
        """
        return self.dec_fcn(llr_in, llr_out)

    def get_filename_template(self):
        """
        A unique filename template that is generated from code parameters
        """
        # Encode information bit count and block length into the filename template
        return 'bin_ldpc_' + super().get_filename_template()

    def get_title_template(self):
        """
        Get title string template for live-plot
        """
        return 'Binary LDPC, ' + super().get_title_template()

    def __str__(self):
        msg = 'Binary LDPC code with soft decoding algorithm\n'
        return msg + super().__str__()


def instantiate_codec(**kwargs):
    """
    Generate codec instance from configuration provided by experiment file
    """
    codec_type = kwargs.pop('type')
    class_type = getattr(sys.modules[__name__], codec_type)
    return class_type(**kwargs)
