"""
5G LDPC codes constructor
"""
import json
import os
import argparse

import numpy as np
import galois

from ldpc_soft_py.bin_ldpc_soft import Alist
from ldpc_soft_py.bin_pcm_tools import expand_pcm

CODES_DIR = 'codes'

ALLOWED_ZC = sorted({
    2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
    18, 20, 22, 24, 26, 28, 30, 32, 36, 40, 44, 48, 52, 56,
    60, 64, 72, 80, 88, 96, 104, 112, 120, 128, 144, 160,
    176, 192, 208, 224, 240, 256, 288, 320, 352, 384, 576
})


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


def choose_lifting_size(payload_length, base_graph):
    """
    Select the smallest supported Zc for the requested payload length K'.
    """
    if not isinstance(payload_length, (int, np.integer)) or payload_length <= 0:
        raise ValueError('Payload length must be a positive integer')
    if base_graph == 1:
        k_base = 22
    elif base_graph == 2:
        if payload_length <= 192:
            k_base = 6
        elif payload_length <= 560:
            k_base = 8
        elif payload_length <= 640:
            k_base = 9
        else:
            k_base = 10
    else:
        raise ValueError('Unknown base graph value')

    for lifting_size in ALLOWED_ZC:
        if k_base * lifting_size >= payload_length:
            return lifting_size

    raise ValueError(
        f'Payload length {payload_length} is too large for BG{base_graph}'
    )


def gen_pcm(payload_length, base_graph):
    """
    Generate parity check matrix
    """
    bg_data, _ = get_pcm_expander(base_graph)
    expected_shape = (46, 68) if base_graph == 1 else (42, 52)
    pcm_base = np.asarray(bg_data['H'])
    if pcm_base.shape != expected_shape:
        raise ValueError(
            f'Base graph {base_graph} must have shape {expected_shape}, '
            f'got {pcm_base.shape}'
        )

    k_base = 22 if base_graph == 1 else 10
    lifting_size = choose_lifting_size(payload_length, base_graph)
    lift_index = factor_to_index(lifting_size)
    pcm_exp = np.array(bg_data['sets'][str(lift_index)])
    if pcm_exp.shape != expected_shape:
        raise ValueError(
            f'Lifting set {lift_index} for BG{base_graph} must have shape '
            f'{expected_shape}, got {pcm_exp.shape}'
        )

    pcm = expand_pcm(lifting_size, pcm_exp).astype(np.uint8)
    systematic_length = k_base * lifting_size
    mother_length = pcm.shape[1]
    expected_pcm_shape = (
        (46 * lifting_size, 68 * lifting_size)
        if base_graph == 1
        else (42 * lifting_size, 52 * lifting_size)
    )
    if pcm.shape != expected_pcm_shape:
        raise ValueError(
            f'Expanded BG{base_graph} must have shape {expected_pcm_shape}, '
            f'got {pcm.shape}'
        )
    filler_mask = np.zeros(mother_length, dtype=bool)
    filler_mask[payload_length:systematic_length] = True
    return pcm, k_base, lifting_size, filler_mask


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


# def get_pcm_expander(n_inf_bits, base_graph):
#     """
#     Get parity check matrix in the expander form
#     """
#     # Load base-graph
#     if base_graph not in [1, 2]:
#         raise ValueError('Unknown base graph value')
#     with open(f'ldpc_5g_data/bg{base_graph}.json', 'r', encoding='utf-8') as filedesc:
#         bg_data = json.load(filedesc)

#     pcm_base = np.array(bg_data['H'])
#     kb_max = pcm_base.shape[1] - pcm_base.shape[0]
#     # Cases below correspond to base graph 2
#     if n_inf_bits > 640:
#         k_base = kb_max
#     elif n_inf_bits > 560:
#         k_base = 9
#     elif n_inf_bits > 192:
#         k_base = 8
#     else:
#         k_base = 6

#     # For base graph 1, k_base = 22 for all cases
#     if base_graph == 1:
#         k_base = kb_max
#     factor = int(np.round(n_inf_bits / k_base))

#     print(f'Factor: {factor}, K_b = {k_base}.')
#     print(f'K = {n_inf_bits}/{k_base * factor} (intended/actual)')
#     lift_index = factor_to_index(factor)
#     print(f'Index: {lift_index}.')

#     pcm_expander = np.array(bg_data['sets'][str(lift_index)])
#     return np.hstack([pcm_expander[:, :k_base], pcm_expander[:, kb_max:]]), k_base

def get_pcm_expander(base_graph):
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

    return bg_data, kb_max





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


def generate_5g_code(inf_bits_count, coding_rate, base_graph):
    """
    Main function
    """
    # Generate parity check matrix:
    print('Creating the parity check matrix...')
    pcm, k_base, lifting_size, filler_mask = gen_pcm(
        int(inf_bits_count),
        base_graph
    )
    filename_template = get_filename_template(pcm, lifting_size)

    pcm_file = filename_template + '_pcm.alist'
    Alist.write(pcm, os.path.join(CODES_DIR, pcm_file))
    print('Creating the generator matrix...')
    try:
        generator_mtx = get_generator(pcm, k_base, lifting_size)
        gen_mtx_file = filename_template + '_generator.txt'
        np.savetxt(os.path.join(CODES_DIR, gen_mtx_file), generator_mtx, delimiter=' ', fmt='%d')
    except KeyboardInterrupt:
        print('Interrupted. Generator matrix will not be created')
        gen_mtx_file = None

    systematic_length = k_base * lifting_size
    mother_length = pcm.shape[1]
    punctured_prefix = 2 * lifting_size
    code = {
        'pcm': filename_template + '_pcm.alist',
        'punctured': punctured_prefix,
        'is_systematic': True,
        'base_graph': base_graph,
        'lifting_size': lifting_size,
        'payload_length': int(inf_bits_count),
        'systematic_length': systematic_length,
        'mother_length': mother_length,
        'rm_input_length': mother_length - punctured_prefix,
        'punctured_prefix': punctured_prefix,
        'filler': {
            'start': int(inf_bits_count),
            'stop': systematic_length
        },
        'filler_mask': filler_mask.tolist()
    }
    if gen_mtx_file:
        code['generator'] = gen_mtx_file

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
