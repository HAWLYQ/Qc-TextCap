# Copyright (c) Facebook, Inc. and its affiliates.
from pythia.common.registry import Registry
from pythia.datasets.captioning.m4c_control_textcaps.dataset import M4CControlTextCapsDataset
from pythia.datasets.vqa.m4c_textvqa.builder import M4CTextVQABuilder


@Registry.register_builder("m4c_control_textcaps")
class M4CControlTextCapsBuilder(M4CTextVQABuilder):
    def __init__(self):
        print('pythia/datasets/captioning/m4c_control_textcaps/builder.py  M4CControlTextCapsBuilder init')
        super().__init__()
        self.dataset_name = "m4c_control_textcaps"
        self.set_dataset_class(M4CControlTextCapsDataset)
