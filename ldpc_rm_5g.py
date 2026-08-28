"""
5G LDPC codes constructor
"""
import json
import os
import argparse
import bisect

import numpy as np
import galois

from ldpc_soft_py.bin_ldpc_soft import Alist
from ldpc_soft_py.bin_pcm_tools import expand_pcm

CODES_DIR = 'codes'

ALLOWED_LIFTING_SIZES = [
    2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 
    15, 16, 18, 20, 22, 24, 26, 28, 30, 32, 36, 
    40, 44, 48, 52, 56, 60, 64, 72, 80, 88, 96, 
    104, 112, 120, 128, 144, 160, 176, 192, 208, 
    224, 240, 256, 288, 320, 352, 384, 576
]


def factor_to_index(lifting_size):
    """
    Convert factor to circulant set index
    Table 5.3.2-1: Sets of LDPC lifting size Z encoded as a function
    """
    if lifting_size in [2, 4, 8, 16, 32, 64, 128, 256]:
        return 1
    if lifting_size in [3, 6, 12, 24, 48, 96, 192, 384]:
        return 2
    if lifting_size in [5, 10, 20, 40, 80, 160, 320]:
        return 3
    if lifting_size in [7, 14, 28, 56, 112, 224]:
        return 4
    if lifting_size in [9, 18, 36, 72, 144, 288, 576]:
        return 5
    if lifting_size in [11, 22, 44, 88, 176, 352]:
        return 6
    if lifting_size in [13, 26, 52, 104, 208]:
        return 7
    if lifting_size in [15, 30, 60, 120, 240]:
        return 8
    raise ValueError('Can not get set index for factor ', lifting_size)

def choose_lifting_size(n_inf_bits, base_graph):
    """
    Choses Zc
    """
    if type(n_inf_bits) != int:
        raise TypeError(f"n_inf_bits must be int, but it is {type(n_inf_bits)}")
    if n_inf_bits <= 0:
        raise ValueError(f"n_inf_bits must be positive, but it is {n_inf_bits}")

    if base_graph not in [1, 2]:
        raise ValueError('Unknown base graph value')

    if base_graph == 1:
        selection_k_base = 22
    elif n_inf_bits > 640:
        selection_k_base = 10
    elif n_inf_bits > 560:
        selection_k_base = 9
    elif n_inf_bits > 192:
        selection_k_base = 8
    else:
        selection_k_base = 6

    lifting_size_idx = bisect.bisect_left(
        ALLOWED_LIFTING_SIZES, 
        n_inf_bits / selection_k_base
    )
    if lifting_size_idx >= len(ALLOWED_LIFTING_SIZES):
        raise ValueError("Unable to find lifting_size")
    lifting_size = ALLOWED_LIFTING_SIZES[lifting_size_idx]

    print(f'Zc: {lifting_size}')

    return lifting_size


def gen_pcm(n_inf_bits, base_graph):
    """
    Generate parity check matrix
    """
    pcm_exp, k_base, lifting_size = get_pcm_expander(n_inf_bits, base_graph)
    pcm_full = np.array(expand_pcm(lifting_size, pcm_exp), dtype=np.uint8)
    return pcm_full, k_base, lifting_size


def get_generator(pcm, k_base, factor):
    """
    Generator matrix construction
    """
    inv = np.array(np.linalg.inv(galois.GF2(
        pcm[:(4 * factor), (k_base * factor): (k_base + 4) * factor]
    ))).astype(np.uint8)
    inf_part_1 = pcm[:4 * factor, :k_base * factor]
    inf_part_2 = pcm[4 * factor:, :(k_base + 4) * factor]
    generator_l = np.hstack([np.eye(k_base * factor), np.mod(inv @ inf_part_1, 2).T])
    generator_r = np.hstack([np.eye((k_base + 4) * factor), inf_part_2.T])
    return np.mod(generator_l @ generator_r, 2).astype(np.uint8)


def get_pcm_expander(n_inf_bits, base_graph):
    """
    Get parity check matrix in the expander form
    """
    # Load base-graph
    if base_graph not in [1, 2]:
        raise ValueError('Unknown base graph value')
    with open(f'ldpc_5g_data/bg{base_graph}.json', 'r', encoding='utf-8') as filedesc:
        bg_data = json.load(filedesc)

    pcm_base = np.array(bg_data['H'])
    kb_max = pcm_base.shape[1] - pcm_base.shape[0]
    lifting_size = choose_lifting_size(n_inf_bits, base_graph)

    lift_index = factor_to_index(lifting_size)
    print(f'Index: {lift_index}.')

    pcm_expander = np.array(bg_data['sets'][str(lift_index)])
    return pcm_expander, kb_max, lifting_size


def get_filename_template(pcm, factor):
    """
    Generate filename template for LDPC code:
    - generator matrix (np.savetxt(), space-separated) (*_generator.txt)
    - parity check matrix in the ALIST format (*_pcm.alist)
    - JSON file that also keeps (*.json):
        - Information bits indices
        - Punctured indices
    """
    n_checks, block_len = pcm.shape
    n_inf_bits = block_len - n_checks
    block_len -= 2 * factor
    return f'ldpc_5g_k{n_inf_bits}_n{block_len}'

def remove_filler_columns(
    pcm_full,
    n_inf_bits,
    k_base,
    lifting_size
):
    """
    Removes filler columns
    """
    length = k_base * lifting_size
    if n_inf_bits not in range(1, length + 1):
        raise ValueError(
            f"n_inf_bits must be in [{1}, {length}], "
            f"but it is {n_inf_bits}"
        )

    if pcm_full.shape[1] < length:
        raise ValueError(
            f"pcm_full.shape[1] must be >= {length}, "
            f"but it is {pcm_full.shape[1]}"
        )

    kept_columns = np.concat((
        np.arange(0, n_inf_bits), 
        np.arange(length, pcm_full.shape[1])
    ))
    pcm_reduced = pcm_full[:, kept_columns]

    return pcm_reduced, kept_columns


def remove_filler_from_generator(
    generator_full,
    n_inf_bits,
    kept_columns
):
    """
    Remove filler input rows and the corresponding codeword columns
    from a full generator matrix.
    """
    if not isinstance(generator_full, np.ndarray) or generator_full.ndim != 2:
        raise ValueError("generator_full must be a two-dimensional numpy array")
    if not isinstance(n_inf_bits, (int, np.integer)) or not (
        0 < n_inf_bits <= generator_full.shape[0]
    ):
        raise ValueError(
            "n_inf_bits must be in "
            f"[1, {generator_full.shape[0]}], but it is {n_inf_bits}"
        )

    kept_columns = np.asarray(kept_columns)
    if kept_columns.ndim != 1:
        raise ValueError("kept_columns must be one-dimensional")
    if not np.issubdtype(kept_columns.dtype, np.integer):
        raise TypeError("kept_columns must contain integer indices")
    if kept_columns.size and (
        np.any(kept_columns < 0)
        or np.any(kept_columns >= generator_full.shape[1])
    ):
        raise ValueError("kept_columns contains an out-of-range index")

    return np.asarray(
        generator_full[:n_inf_bits, kept_columns],
        dtype=np.uint8
    )


def validate_reduced_matrices(pcm, generator, n_inf_bits):
    """Validate dimensions, systematic form, and H * G^T = 0 over GF(2)."""
    if pcm.ndim != 2 or generator.ndim != 2:
        raise ValueError("PCM and generator must be two-dimensional")
    if pcm.shape[1] - pcm.shape[0] != n_inf_bits:
        raise ValueError("Reduced PCM dimension does not match n_inf_bits")
    if generator.shape != (n_inf_bits, pcm.shape[1]):
        raise ValueError("Reduced generator / PCM shape mismatch")
    if not np.array_equal(
        generator[:, :n_inf_bits],
        np.eye(n_inf_bits, dtype=np.uint8)
    ):
        raise ValueError("Reduced generator matrix is not systematic")
    if np.any((pcm @ generator.T) % 2):
        raise ValueError("Reduced PCM and generator matrix are inconsistent")


def build_code_config(
    pcm_file,
    generator_file,
    pcm,
    n_inf_bits,
    coding_rate,
    punctured_prefix
):
    """Build the complete code configuration, including rate matching."""
    if not isinstance(coding_rate, (int, float, np.integer, np.floating)) or (
        coding_rate <= 0
    ):
        raise ValueError("coding_rate must be positive")
    if not isinstance(punctured_prefix, (int, np.integer)) or not (
        0 <= punctured_prefix < pcm.shape[1]
    ):
        raise ValueError("punctured_prefix is outside the codeword")

    channel_length = int(np.round(n_inf_bits / coding_rate))
    if channel_length <= 0:
        raise ValueError("coding_rate produces an empty channel buffer")

    circular_buffer_length = pcm.shape[1] - punctured_prefix
    punctured_tail = max(circular_buffer_length - channel_length, 0)
    complete_cycles, remainder = divmod(
        channel_length,
        circular_buffer_length
    )

    if complete_cycles == 0:
        max_multiplicity = 1
        last_max_multiplicity_index = punctured_prefix + remainder - 1
    elif remainder == 0:
        max_multiplicity = complete_cycles
        last_max_multiplicity_index = pcm.shape[1] - 1
    else:
        max_multiplicity = complete_cycles + 1
        last_max_multiplicity_index = punctured_prefix + remainder - 1

    code = {
        'pcm': pcm_file,
        'punctured': int(punctured_prefix),
        'punctured_tail': int(punctured_tail),
        'max_multiplicity': int(max_multiplicity),
        'last_max_multiplicity_index': int(last_max_multiplicity_index),
        'is_systematic': True
    }
    if generator_file is not None:
        code['generator'] = generator_file
    return code


def generate_5g_code(inf_bits_count, coding_rate, base_graph):
    """
    Main function
    """
    # Generate parity check matrix:
    print('Creating the parity check matrix...')
    n_inf_bits = int(inf_bits_count)
    pcm, k_base, factor = gen_pcm(
        n_inf_bits,
        base_graph
    )
    pcm_reduced, kept_columns = remove_filler_columns(
        pcm,
        n_inf_bits,
        k_base,
        factor
    )
    print('Creating the generator matrix...')
    try:
        generator_full = get_generator(pcm, k_base, factor)
        generator_reduced = remove_filler_from_generator(
            generator_full,
            n_inf_bits,
            kept_columns
        )
        validate_reduced_matrices(
            pcm_reduced,
            generator_reduced,
            n_inf_bits
        )
    except KeyboardInterrupt:
        print('Interrupted. Generator matrix will not be created')
        generator_reduced = None

    filename_template = get_filename_template(pcm_reduced, factor)
    pcm_file = filename_template + '_pcm.alist'
    Alist.write(pcm_reduced, os.path.join(CODES_DIR, pcm_file))

    if generator_reduced is not None:
        gen_mtx_file = filename_template + '_generator.txt'
        np.savetxt(
            os.path.join(CODES_DIR, gen_mtx_file),
            generator_reduced,
            delimiter=' ',
            fmt='%d'
        )
    else:
        gen_mtx_file = None

    code = build_code_config(
        pcm_file,
        gen_mtx_file,
        pcm_reduced,
        n_inf_bits,
        coding_rate,
        2 * factor
    )

    json_file = os.path.join(CODES_DIR, filename_template + '.json')
    with open(json_file, 'w', encoding='utf-8') as filedesc:
        json.dump(code, filedesc, indent=2)
    print(f'Successfully generated {json_file}')
    return json_file


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate 5G-LDPC codes')
    parser.add_argument('--k', help='The number of information bits')
    parser.add_argument('--rate', help='Intended coding rate')
    parser.add_argument('--BG', help='Base graph: \'1\' or \'2\'')
    args = parser.parse_args()
    try:
        generate_5g_code(
            inf_bits_count=float(args.k),  # Intended information bits count
            coding_rate=float(args.rate),  # Intended coding rate
            base_graph=int(args.BG)
        )
    except TypeError:
        print(f'Usage: {__file__} -h')
    except ValueError:
        print('Code was not created. Try to vary the number of information bits')
        print('Factor must be [2, 3, 5, 7, 9, 11, 13, 15] X 2**N')
