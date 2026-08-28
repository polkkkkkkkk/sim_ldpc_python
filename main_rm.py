"""Entry point for LDPC simulations with simplified rate matching."""

import numpy as np

from lbc_encoder.lbc_encoder import lib_compile as lbc_compile
from ldpc_soft_py.bin_ldpc_soft import lib_compile as ldpc_compile
from ldpc_rm_experiment import (
    LdpcRmDataEntry,
    LdpcRmExperimentInstance,
    LdpcRmExperimentSettings,
)
from simulator_rm_awgn_python.channel import lib_compile as chan_compile
from simulator_rm_awgn_python.tools import load_json


def compile_all():
    chan_compile()
    ldpc_compile()
    lbc_compile()


def single_run():
    config = load_json('experiment_rm.json')
    settings = LdpcRmExperimentSettings(**config['experiment'])
    instance = LdpcRmExperimentInstance(settings)
    data = instance.run(-8.0, np.random.default_rng(seed=1))
    print(data)


if __name__ == '__main__':
    from simulator_rm_awgn_python.simulator import run_all_experiments

    compile_all()
    run_all_experiments(
        LdpcRmExperimentSettings,
        LdpcRmExperimentInstance,
        LdpcRmDataEntry
    )
