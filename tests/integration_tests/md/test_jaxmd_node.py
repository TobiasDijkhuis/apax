import znh5md
import pytest
import ipsuite as ips
from apax.config.md_config import MDConfig
from apax.config.train_config import Config
import zntrack
from ase.io import read, write
from tests.conftest import load_config_and_run_training
import pathlib
import apax.nodes
import uuid
import yaml
import ipsuite.base
from ase import Atoms
import os

TEST_PATH = pathlib.Path(__file__).parent.resolve()


# @pytest.mark.parametrize(
#     "traj_path", ("../../../../CH3_project/nodes/cycle-00/ApaxJaxMD/md/md.h5",)
# )


class PrintEnergyNode(ips.base.IPSNode):
    data: list[Atoms] = zntrack.deps()

    def run(self):
        #  if isinstance(self.data, znh5md.IO):
        #      atoms_lst = self.data[:]
        #  else:
        #      atoms_lst = self.data

        print([atoms.get_potential_energy() for atoms in self.data])
        # self.frames = self.data


@pytest.mark.parametrize("num_data", (30,))
def test_read_stopped_md(get_tmp_path, example_dataset):
    model_confg_path = TEST_PATH / "config.yaml"
    updated_model_confg_path = TEST_PATH / "updated_config.yaml"
    md_confg_path = TEST_PATH / "md_config_threshold.yaml"
    updated_md_confg_path = TEST_PATH / "updated_md_config.yaml"

    working_dir = get_tmp_path / str(uuid.uuid4())
    data_path = get_tmp_path / "ds.extxyz"

    write(data_path, example_dataset)
    data = read(data_path, index=":")

    with open(md_confg_path, "r") as file:
        md_config_dict = yaml.safe_load(file)
    del md_config_dict["properties"]
    md_config_dict["n_inner"] = 5
    md_config_dict["sampling_rate"] = 2
    with open(updated_md_confg_path, "w") as file:
        yaml.safe_dump(md_config_dict, file)

    with open(model_confg_path, "r") as file:
        model_config_dict = yaml.safe_load(file)
    del model_config_dict["data"]["data_path"]
    del model_config_dict["data"]["experiment"]
    with open(updated_model_confg_path, "w") as file:
        yaml.safe_dump(model_config_dict, file)

    with ips.Project() as project:
        model = apax.nodes.Apax(
            data=data[:-2], config=updated_model_confg_path, validation_data=data[-2:]
        )

        md = apax.nodes.md.ApaxJaxMD(
            data=data,
            model=model,
            config=updated_md_confg_path,
        )

        print = PrintEnergyNode(data=md.frames[:])

    project.run()

    # frames = znh5md.IO("nodes/ApaxJaxMD/md/md.h5")[:]
    # print(frames[1].get_potential_energy())

    os.removedirs("nodes")
