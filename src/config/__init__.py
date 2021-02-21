import copy
import json
import hashlib
from typing import NamedTuple, List, Dict, Optional


class DataConfig(NamedTuple):
    raw_path: str
    cache_path: str
    split_names: List[str]
    split_portions: List[float]
    split_mode: str
    dataset_format: str
    target_column: str
    text_column: str


class ExtractorConfig(NamedTuple):
    ex_type: str
    ex_args: dict


class ModelConfig(NamedTuple):
    model_name: str
    problem_type: str
    batch_size: str
    model_conf: dict


class ExperimentConfig(NamedTuple):
    data_config: DataConfig
    extractor_config: ExtractorConfig
    model_config: ModelConfig
    label_names: List[str]
    seed: Optional[int]
    output_path: str
    model_path: Optional[str] = None
    pretrained_model_path: Optional[str] = None

    def _as_flat_dict(self):
        def flatten(value):
            if getattr(value, '_asdict', None) is None:
                return value
            return value._asdict()
        own_dict = self._asdict()
        return {k: flatten(v) for (k, v) in own_dict.items()}

    def hash(self):
        as_dict = self._as_flat_dict()
        as_json = json.dumps(as_dict, sort_keys=True)
        as_hash = hashlib.md5(as_json.encode('utf8')).hexdigest()
        return as_hash

    def output_file(self):
        experiment_hash = self.hash()
        result_path = '{}/{}.json'.format(self.output_path, experiment_hash)
        return result_path

    def model_file(self):
        if self.model_path is None:
            return None
        experiment_hash = self.hash()
        result_path = '{}/{}.pkl'.format(self.model_path, experiment_hash)
        return result_path

    @staticmethod
    def from_dict(config_dict):
        config_dict = copy.deepcopy(config_dict)
        config_dict['data_config'] = DataConfig(**config_dict['data_config'])
        config_dict['extractor_config'] = ExtractorConfig(**config_dict['extractor_config'])
        config_dict['model_config'] = ModelConfig(**config_dict['model_config'])
        return ExperimentConfig(**config_dict)
