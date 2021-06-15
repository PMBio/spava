import pandas as pd
import math
from typing import List, Dict, Union, Optional
import re
import numpy as np
import torch
import torch.nn as nn
from ignite.metrics import Metric
from ignite.exceptions import NotComputableError

# These decorators helps with distributed settings
from ignite.metrics.metric import sync_all_reduce, reinit__is_reduced

DISEASE_STATUSES = ['non-tumor', 'tumor']
CANCER_SUBTYPES = ['PR+ER+', 'PR-ER+', 'PR-ER-', 'PR+ER-']
CANCER_CLINICAL_TYPES = ['HR+HER2-', 'TripleNeg', 'HR+HER2+', 'HR-HER2+']
# using the labels proposed here: https://www.cancerresearchuk.org/about-cancer/breast-cancer/stages-types-grades/tnm-staging
PC_FLATTEN_PTNM_TN_LABELS = True
VALID_PTNM_T_LABELS = ['TX', 'T1', 'T1a', 'T1b', 'T1c', 'T2', 'T2a', 'T3', 'T4', 'T4b']
PTNM_T_LABELS_HIERARCHY = {
    'TX': [],
    'T1': ['T1a', 'T1b', 'T1c'],
    'T2': ['T2a'],
    'T3': [],
    'T4': ['T4b']
}
VALID_PTNM_N_LABELS = ['pNX', 'pN0', 'pN1', 'pN1a', 'pN1mi', 'pN2', 'pN2a', 'pN3', 'pN3a', 'pN3b']
PTNM_N_LABELS_HIERARCHY = {
    'pNX': [],
    'pN0': [],
    'pN1': ['pN1a', 'pN1mi'],
    'pN2': ['pN2a'],
    'pN3': ['pN3a', 'pN3b']
}
VALID_PTNM_M_LABELS = ['M0', 'cMo(i+)', 'pM1']


def get_description_of_cleaned_features():
    feature_description = {
        'image_level_features': ['FileName_FullStack', 'merged_pid', 'diseasestatus', 'Height_FullStack', 'Width_FullStack', 'area', 'sum_area_cells', 'Count_Cells'],
        'patient_level_features': ['PrimarySite', 'Subtype', 'clinical_type', 'PTNM_T', 'PTNM_N', 'PTNM_M',
                                   'DFSmonth', 'OSmonth', 'images_per_patient', 'images_per_patient_filtered', 'cohort']
    }
    return feature_description


def get_predictive_clustering_valid_features():
    feature_description = get_description_of_cleaned_features()
    columns = feature_description['image_level_features'] + feature_description['patient_level_features']
    columns = [s for s in columns if s not in ['FileName_FullStack', 'merged_pid', 'PrimarySite', 'Height_FullStack', 'Width_FullStack', 'images_per_patient']]
    return columns


def flatten_ptnm_tn_labels(my_class, ptnm_labels_hierarchy):
    if type(my_class) == float and math.isnan(my_class):
        return my_class
    big_classes_labels = list(ptnm_labels_hierarchy.keys())
    small_classes_labels = []
    for v in ptnm_labels_hierarchy.values():
        small_classes_labels.extend(v)

    if my_class in big_classes_labels:
        return my_class
    else:
        assert my_class in small_classes_labels
        parent_class = [k for k, v in ptnm_labels_hierarchy.items() for vv in v if vv == my_class]
        assert len(parent_class) == 1
        parent_class = parent_class[0]
        return parent_class


def clean_metadata(df_basel, df_zurich, verbose=False):
    if verbose:
        print('clearing metadata')
    df = pd.concat([df_basel, df_zurich])

    assert np.sum(df['FileName_FullStack'].isna()) == 0
    assert df['FileName_FullStack'].is_unique

    assert np.sum(df['PID'].isna()) == 0
    pids_basel = df['PID'].unique()
    # if not CONTINUOUS_INTEGRATION:
    #     assert len(pids_basel) == max(pids_basel)
    pids_zurich = df['PID'].unique()
    # if not CONTINUOUS_INTEGRATION:
    #     assert len(pids_zurich) == max(pids_zurich)
    df_basel = df_basel.rename(columns={'PID': 'merged_pid'})
    df_zurich['merged_pid'] = df_zurich['PID'].apply(lambda x: x + max(pids_basel))
    df_zurich.drop('PID', axis=1, inplace=True)
    df = pd.concat([df_basel, df_zurich])
    feature_description = get_description_of_cleaned_features()
    assert set(df.columns.to_list()) == set(feature_description['image_level_features'] + feature_description['patient_level_features'])
    pids = sorted(df['merged_pid'].unique())
    for my_pid in pids:
        dff = df.loc[df['merged_pid'] == my_pid, :]
        must_be_shared = feature_description['patient_level_features']
        dfff = dff[must_be_shared]
        dfff.eq(dfff.iloc[0, :], axis=1)
        pass

    assert np.sum(df_basel['diseasestatus'].isna()) == 0
    assert np.sum(df_zurich['diseasestatus'].isna()) == len(df_zurich)
    df_zurich['diseasestatus'] = ['tumor'] * len(df_zurich)
    df = pd.concat([df_basel, df_zurich])
    assert all([e in DISEASE_STATUSES for e in df['diseasestatus'].value_counts().to_dict().keys()])

    assert df_basel['PrimarySite'].value_counts().to_dict() == {'breast': len(df_basel)}
    assert np.sum(df_zurich['PrimarySite'].isna()) == len(df_zurich)
    df_zurich['PrimarySite'] = ['breast'] * len(df_zurich)

    def warn_on_na(df, df_name, column):
        n = np.sum(df[column].isna())
        if n > 0:
            if verbose:
                print(f'warning: {df_name}[{column}] contains {n} NAs out of {len(df)} values')

    assert all([e in CANCER_SUBTYPES for e in df['Subtype'].value_counts().to_dict().keys()])
    warn_on_na(df_basel, 'df_basel', 'Subtype')
    warn_on_na(df_zurich, 'df_zurich', 'Subtype')

    assert all([e in CANCER_CLINICAL_TYPES for e in df['clinical_type'].value_counts().to_dict().keys()])
    warn_on_na(df_basel, 'df_basel', 'clinical_type')
    warn_on_na(df_zurich, 'df_zurich', 'clinical_type')

    assert np.sum(df['Height_FullStack'].isna()) == 0

    assert np.sum(df['Width_FullStack'].isna()) == 0

    assert np.sum(df['area'].isna()) == 0

    assert np.sum(df['sum_area_cells'].isna()) == 0

    assert np.sum(df['Count_Cells'].isna()) == 0

    # PTNM_T
    assert np.sum(df['PTNM_T'].isna()) == 0

    def ptnm_t_renamer(bad_label):
        label = re.sub(r'^t', '', bad_label)
        if label == '[]':
            label = 'X'
        label = 'T' + label
        return label

    df_basel['PTNM_T'] = df_basel['PTNM_T'].apply(ptnm_t_renamer)
    df_zurich['PTNM_T'] = df_zurich['PTNM_T'].apply(ptnm_t_renamer)
    if PC_FLATTEN_PTNM_TN_LABELS:
        if verbose:
            print('flattening PTNM_T labels')
        df_basel['PTNM_T'] = df_basel['PTNM_T'].apply(lambda x: flatten_ptnm_tn_labels(x, PTNM_T_LABELS_HIERARCHY))
        df_zurich['PTNM_T'] = df_zurich['PTNM_T'].apply(lambda x: flatten_ptnm_tn_labels(x, PTNM_T_LABELS_HIERARCHY))
    df = pd.concat([df_basel, df_zurich])
    assert all([e in VALID_PTNM_T_LABELS for e in df['PTNM_T'].value_counts().to_dict().keys()])
    if verbose:
        # TODO: check if this interpretation is correct
        print('warning: interpreting the PTNM_T label "[]" as "TX"')

    # PTNM_N
    assert np.sum(df['PTNM_N'].isna()) == 0

    def ptnm_n_renamer(bad_label):
        label = re.sub(r'^n', '', bad_label)
        if label in ['0sl', '0sn']:
            label = '0'
        if label == 'x' or label == 'X' or label == '[]':
            label = 'X'
        label = 'pN' + label
        return label

    df_basel['PTNM_N'] = df_basel['PTNM_N'].apply(ptnm_n_renamer)
    df_zurich['PTNM_N'] = df_zurich['PTNM_N'].apply(ptnm_n_renamer)
    if PC_FLATTEN_PTNM_TN_LABELS:
        if verbose:
            print('flattening PTNM_N labels')
        df_basel['PTNM_N'] = df_basel['PTNM_N'].apply(lambda x: flatten_ptnm_tn_labels(x, PTNM_N_LABELS_HIERARCHY))
        df_zurich['PTNM_N'] = df_zurich['PTNM_N'].apply(lambda x: flatten_ptnm_tn_labels(x, PTNM_N_LABELS_HIERARCHY))
    df = pd.concat([df_basel, df_zurich])
    assert all([e in VALID_PTNM_N_LABELS for e in df['PTNM_N'].value_counts().to_dict().keys()])
    if verbose:
        # TODO: check if these interpretations are correct
        print('warning: interpreting the PTNM_N label "[]" as "pNX"')
        print('warning: interpreting the PTNM_N labels "0sl" and "0sn" as "pN0"')

    # PTNM_M
    assert np.sum(df['PTNM_M'].isna()) == 0

    def ptnm_m_renamer(bad_label):
        label = bad_label
        if label == '0' or label == 0:
            label = 'M0'
        if label == '1' or label == 1 or label == 'M1':
            label = 'pM1'
        if label == 'M0_IPLUS':
            label = 'cMo(i+)'
        return label

    df_basel['PTNM_M'] = df_basel['PTNM_M'].apply(ptnm_m_renamer)
    df_zurich['PTNM_M'] = df_zurich['PTNM_M'].apply(ptnm_m_renamer)
    df = pd.concat([df_basel, df_zurich])
    assert all([e in VALID_PTNM_M_LABELS for e in df['PTNM_M'].value_counts().to_dict().keys()])
    if verbose:
        # TODO: check if this interpretations is correct
        print('warning: interpreting the "M0_IPLUS" label as "cMo(i+)"')

    assert np.sum(df_basel['DFSmonth'].isna()) == 0
    assert np.sum(df_zurich['DFSmonth'].isna()) == len(df_zurich)

    assert np.sum(df_basel['OSmonth'].isna()) == 0
    assert np.sum(df_zurich['OSmonth'].isna()) == len(df_zurich)
    if verbose:
        print('metadata cleaned')
    return df_basel, df_zurich


def class_to_one_hot(my_class: Union[str, float], classes: List[str]):
    assert len(classes) == len(set(classes))
    if type(my_class) == float and math.isnan(my_class):
        return torch.tensor([float('nan')] * len(classes))
    row = None
    for i, one_class in enumerate(classes):
        if my_class == one_class:
            row = i
    if row is None:
        raise ValueError(f'class_to_one_hot: {my_class} not in {classes}')
    return torch.eye(len(classes), dtype=torch.long)[row, :]


def one_hot_to_class(t: torch.Tensor, classes: List[str]):
    if torch.isnan(t).any():
        raise ValueError(f'one_hot_to_class: found nan in the tensor {t}')
    t = t.view(-1)
    assert len(t) == len(classes), (len(t), len(classes), t, classes)
    # warning: note that here we are not checking for multiple maxes as we are doing in ptnm_tn_one_hot_to_class()
    i = torch.argmax(t).item()
    return classes[i]


def ptnm_tn_class_to_one_hot(my_class: Union[str, float], ptnm_labels_hierarchy: Dict[str, List[str]], valid_ptnm_labels: List[str]):
    assert len(valid_ptnm_labels) == len(set(valid_ptnm_labels))
    if type(my_class) == float and math.isnan(my_class):
        return torch.tensor([float('nan')] * len(valid_ptnm_labels))
    big_classes_labels = list(ptnm_labels_hierarchy.keys())
    small_classes_labels = []
    for v in ptnm_labels_hierarchy.values():
        small_classes_labels.extend(v)
    if my_class in big_classes_labels:
        return torch.eye(len(valid_ptnm_labels))[valid_ptnm_labels.index(my_class)]
    else:
        assert my_class in small_classes_labels
        i = valid_ptnm_labels.index(my_class)
        parent_class = [k for k, v in ptnm_labels_hierarchy.items() for vv in v if vv == my_class]
        assert len(parent_class) == 1
        parent_class = parent_class[0]
        parent_class_i = valid_ptnm_labels.index(parent_class)
        t = torch.zeros(len(valid_ptnm_labels), dtype=torch.long)
        t[i] = 1
        t[parent_class_i] = 1
        return t


def torchize_feature(k: str, v):
    if k in ['area', 'sum_area_cells', 'Count_Cells', 'DFSmonth', 'OSmonth']:
        if type(v) == float and math.isnan(v):
            return torch.tensor(float('nan'))
        else:
            return torch.tensor(v, dtype=torch.float)
    elif k == 'diseasestatus':
        return class_to_one_hot(v, DISEASE_STATUSES)
    elif k == 'Subtype':
        return class_to_one_hot(v, CANCER_SUBTYPES)
    elif k == 'clinical_type':
        return class_to_one_hot(v, CANCER_CLINICAL_TYPES)
    elif k == 'PTNM_T':
        return ptnm_tn_class_to_one_hot(v, PTNM_T_LABELS_HIERARCHY, VALID_PTNM_T_LABELS)
    elif k == 'PTNM_N':
        return ptnm_tn_class_to_one_hot(v, PTNM_N_LABELS_HIERARCHY, VALID_PTNM_N_LABELS)
    elif k == 'PTNM_M':
        return class_to_one_hot(v, VALID_PTNM_M_LABELS)
    else:
        raise ValueError(f'torchize_feature: k = {k}, v = {v}')


def ptnm_tn_one_hot_to_class(t: torch.Tensor, ptnm_labels_hierarchy: Dict[str, List[str]], valid_ptnm_labels: List[str]):
    if torch.isnan(t).any():
        raise ValueError(f'one_hot_to_class: found nan in the tensor {t}')
    t = t.view(-1)
    assert len(t) == len(valid_ptnm_labels)
    indexes_of_big_classes = [valid_ptnm_labels.index(s) for s in ptnm_labels_hierarchy.keys()]
    # indexes_of_small_classes = {k: [valid_ptnm_labels.index(vv)] for k, v in ptnm_labels_hierarchy.items() for vv in v}
    indexes_of_small_classes = [valid_ptnm_labels.index(vv) for v in ptnm_labels_hierarchy.values() for vv in v]

    epsilon = 0.001
    big_class_i = []
    big_class_max = -1
    for i, k in zip(indexes_of_big_classes, ptnm_labels_hierarchy.keys()):
        if t[i] >= big_class_max - epsilon:
            if abs(t[i] - big_class_max) <= epsilon:
                big_class_i.append(i)
            else:
                big_class_i = [i]
            big_class_max = t[i]
    assert len(big_class_i) == 1
    big_class_i = big_class_i[0]
    big_class_label = valid_ptnm_labels[big_class_i]

    small_class_i = []
    small_class_max = -1
    for i, k in zip(indexes_of_small_classes, ptnm_labels_hierarchy.keys()):
        if t[i] >= small_class_max - epsilon:
            if abs(t[i] - small_class_max) <= epsilon:
                small_class_i.append(i)
            else:
                small_class_i = [i]
            small_class_max = t[i]
    if all([t[i] < epsilon for i in small_class_i]):
        return big_class_label
    else:
        assert len(small_class_i) == 1
        small_class_i = small_class_i[0]
        small_class_label = valid_ptnm_labels[small_class_i]
        if small_class_label in ptnm_labels_hierarchy[big_class_label]:
            return small_class_label
        else:
            print(f'warning: possible ambiguous predictions between {big_class_label} and {small_class_label}')
            if small_class_max < big_class_max:
                return big_class_label
            else:
                return small_class_label


def untorchize_feature(k: str, t: torch.Tensor):
    if torch.isnan(t).any():
        return float('nan')
    elif k == 'diseasestatus':
        return one_hot_to_class(t, DISEASE_STATUSES)
    elif k == 'Subtype':
        return one_hot_to_class(t, CANCER_SUBTYPES)
    elif k == 'clinical_type':
        return one_hot_to_class(t, CANCER_CLINICAL_TYPES)
    elif k in ['area', 'sum_area_cells', 'Count_Cells', 'DFSmonth', 'OSmonth']:
        return t.item()
    elif k == 'PTNM_T':
        return ptnm_tn_one_hot_to_class(t, PTNM_T_LABELS_HIERARCHY, VALID_PTNM_T_LABELS)
    elif k == 'PTNM_N':
        return ptnm_tn_one_hot_to_class(t, PTNM_N_LABELS_HIERARCHY, VALID_PTNM_N_LABELS)
    elif k == 'PTNM_M':
        return one_hot_to_class(t, VALID_PTNM_M_LABELS)
    else:
        raise ValueError(f'torchize_feature: k = {k}, t = {t}')


def loss_factory(feature_name: str):
    valid_features = get_predictive_clustering_valid_features()
    assert feature_name in valid_features
    if feature_name in ['diseasestatus', 'Subtype', 'clinical_type', 'PTNM_M']:
        return nn.MultiLabelSoftMarginLoss()
    elif feature_name in ['area', 'sum_area_cells', 'Count_Cells', 'DFSmonth', 'OSmonth']:
        return nn.MSELoss()
    elif feature_name == 'PTNM_T':
        return nn.MultiLabelSoftMarginLoss()
    elif feature_name == 'PTNM_N':
        return nn.MultiLabelSoftMarginLoss()
    assert False, f'unexpected feature_name: {feature_name}'


def activation_factory(feature_name: str, in_channels: int):
    valid_features = get_predictive_clustering_valid_features()
    assert feature_name in valid_features
    if feature_name in ['diseasestatus', 'Subtype', 'clinical_type', 'PTNM_M']:
        return nn.LogSoftmax()
    elif feature_name in ['area', 'sum_area_cells', 'Count_Cells', 'DFSmonth', 'OSmonth']:
        return nn.Linear(in_channels, 1)
    elif feature_name == 'PTNM_T':
        return nn.LogSoftmax()
    elif feature_name == 'PTNM_N':
        return nn.LogSoftmax()
    assert False, f'unexpected feature_name: {feature_name}'


class CustomAccuracy(Metric):
    def __init__(self, begin: int, end: int):
        assert 0 <= begin < end
        self.begin = begin
        self.end = end
        self._num_correct = None
        self._num_examples = None
        super(CustomAccuracy, self).__init__()

    @reinit__is_reduced
    def reset(self):
        self._num_correct = 0
        self._num_examples = 0
        super(CustomAccuracy, self).reset()

    @reinit__is_reduced
    def update(self, output):
        y_pred, y = output
        y_pred = y_pred[:, self.begin:self.end]
        y = y[:, self.begin:self.end]

        indices_pred = torch.argmax(y_pred, dim=1)
        indices = torch.argmax(y, dim=1)

        correct = torch.eq(indices_pred, indices).view(-1)

        self._num_correct += torch.sum(correct).item()
        self._num_examples += correct.shape[0]

    @sync_all_reduce('_num_examples', '_num_correct')
    def compute(self):
        if self._num_examples == 0:
            raise NotComputableError('CustomAccuracy must have at least one example before it can be computed.')
        return self._num_correct / self._num_examples


class CustomMSE(Metric):
    def __init__(self, begin: int, end: int):
        assert 0 <= begin < end
        self.begin = begin
        self.end = end
        self._incremental_mse = None
        self._num_examples = None
        super(CustomMSE, self).__init__()

    @reinit__is_reduced
    def reset(self):
        self._incremental_mse = 0.
        self._num_examples = 0
        super(CustomMSE, self).reset()

    @reinit__is_reduced
    def update(self, output):
        y_pred, y = output
        y_pred = y_pred[:, self.begin:self.end]
        y = y[:, self.begin:self.end]
        partial_mse = torch.square((y - y_pred).view(-1)).sum()

        self._incremental_mse += partial_mse.item()
        self._num_examples += y.shape[0]

    @sync_all_reduce('_num_examples', '_incremental_mse')
    def compute(self):
        if self._num_examples == 0:
            raise NotComputableError('CustomAccuracy must have at least one example before it can be computed.')
        return self._incremental_mse / self._num_examples


def metrics_factory(feature_name: str, begin: int, end: int) -> Dict[str, Metric]:
    valid_features = get_predictive_clustering_valid_features()
    assert feature_name in valid_features
    if feature_name in ['diseasestatus', 'Subtype', 'clinical_type', 'PTNM_M']:
        return {'accuracy': CustomAccuracy(begin, end)}
    elif feature_name in ['area', 'sum_area_cells', 'Count_Cells', 'DFSmonth', 'OSmonth']:
        return {'mse': CustomMSE(begin, end)}
    elif feature_name == 'PTNM_T':
        return {'accuracy': CustomAccuracy(begin, end)}
    elif feature_name == 'PTNM_N':
        return {'accuracy': CustomAccuracy(begin, end)}
    assert False, f'unexpected feature_name: {feature_name}'


# from joblib import Memory
# memory = Memory(configs_uzh.paths.joblib_cache_folder)
# memory.clear()


# @memory.cache


def get_metadata(clean=True, clean_verbose=True):
    # import time
    # start = time.time()
    from ds import file_path_old_data
    f_basel = file_path_old_data('../Data_publication/BaselTMA/Basel_PatientMetadata.csv')
    f_zurich = file_path_old_data('../Data_publication/ZurichTMA/Zuri_PatientMetadata.csv')
    df_basel = pd.read_csv(f_basel)
    df_zurich = pd.read_csv(f_zurich)
    selected_columns = ['FileName_FullStack', 'PID', 'diseasestatus', 'PrimarySite', 'Subtype', 'clinical_type',
                        'Height_FullStack', 'Width_FullStack', 'area', 'sum_area_cells', 'Count_Cells',
                        'PTNM_T', 'PTNM_N', 'PTNM_M', 'DFSmonth', 'OSmonth']
    df_basel = df_basel[selected_columns]
    df_zurich = df_zurich[selected_columns]

    df_basel['images_per_patient'] = df_basel['FileName_FullStack'].groupby(df_basel['PID']).transform('count')
    df_zurich['images_per_patient'] = df_zurich['FileName_FullStack'].groupby(df_zurich['PID']).transform('count')
    from splits import train, validation, test
    valid_omes = train + validation + test

    dropped_basel = len(df_basel[~df_basel['FileName_FullStack'].isin(valid_omes)])
    df_basel = df_basel[df_basel['FileName_FullStack'].isin(valid_omes)]
    print(f'discarding {dropped_basel} omes from the Basel cohort, remaining: {len(df_basel)}')
    # assert dropped_basel == 0

    dropped_zurich = len(df_zurich[~df_zurich['FileName_FullStack'].isin(valid_omes)])
    df_zurich = df_zurich[df_zurich['FileName_FullStack'].isin(valid_omes)]
    print(f'discarding {dropped_basel} omes from the Zurich cohort, remaning: {len(df_zurich)}')
    # assert dropped_zurich == 0

    df_basel['images_per_patient_filtered'] = df_basel['FileName_FullStack'].groupby(df_basel['PID']).transform('count')
    df_zurich['images_per_patient_filtered'] = df_zurich['FileName_FullStack'].groupby(df_zurich['PID']).transform(
        'count')
    df_basel['cohort'] = 'basel'
    df_zurich['cohort'] = 'zurich'

    # better safe than sorry
    assert len(df_basel) + len(df_zurich) == len(valid_omes)

    if clean:
        df_basel, df_zurich = clean_metadata(df_basel, df_zurich, verbose=clean_verbose)
    # print(f'get_metadata(clean={clean}): {time.time() - start}')

    # a = df_basel.PID.tolist()
    # b = df_zurich.PID.tolist()
    # print(len(set(a)), len(set(b)), len(set(a) | set(b)))
    # print(min(a), max(a), min(b), max(b))
    #
    # df_zurich['PID'] = df_zurich['PID'].apply(lambda x: x + max(a))
    #
    # a = df_basel.PID.tolist()
    # b = df_zurich.PID.tolist()
    # print(len(set(a)), len(set(b)), len(set(a) | set(b)), len(set(a)) + len(set(b)), len(set(a).intersection(set(b))))
    # print(max(a), max(b))

    df = pd.concat([df_basel, df_zurich])
    # return df_basel, df_zurich
    return df
    # non_tumor = df_basel[['diseasestatus', 'PID']][df_basel['diseasestatus'] == 'non-tumor'].groupby(['PID']).count()
    # tumor = df_basel[['diseasestatus', 'PID']][df_basel['diseasestatus'] == 'tumor'].groupby(['PID']).count()


if __name__ == '__main__':
    # df_basel, df_zurich = get_metadata()
    # print(df_basel)
    df = get_metadata()
    print(df)
