from ai_backend.evaluators.metrics.metric_eval import Accuracy, F1, Precision, Recall
from ai_backend.evaluators.metrics.multi_label_metrics import MultiLabelAccuracy, MultiLabelRecall, MultiLabelPrecision, MultiLabelFBeta, MultiLabelConfusionMatrix
from ai_backend.evaluators.metrics.loss_metrics import CrossEntropyLossMetric, BCELossMetric, BCEWithLogitsLossMetric

metric_dict = {'accuracy' : Accuracy(),
                'f1_micro' : F1(average='micro'), 'precision_micro' : Precision(average='micro'), 'recall_micro' : Recall(average='micro'),
                'f1_macro' : F1(average='macro'), 'precision_macro' : Precision(average='macro'), 'recall_macro' : Recall(average='macro'),
                'cross_entropy_loss' : CrossEntropyLossMetric(), 'bce_loss' : BCELossMetric()}

multilabel_metric_dict = {'accuracy_micro' : MultiLabelAccuracy(averaging_type='micro'), 'accuracy_macro' : MultiLabelAccuracy(averaging_type='macro'),
                            'f1_micro' : MultiLabelFBeta(beta=1, averaging_type='micro'), 'precision_micro' : MultiLabelPrecision(averaging_type='micro'), 'recall_micro' : MultiLabelRecall(averaging_type='micro'),
                            'f1_macro' : MultiLabelFBeta(beta=1, averaging_type='macro'), 'precision_macro' : MultiLabelPrecision(averaging_type='macro'), 'recall_macro' : MultiLabelRecall(averaging_type='macro'),
                            'confusion_matrix' : MultiLabelConfusionMatrix(),
                            'bce_with_logits_loss' : BCEWithLogitsLossMetric()}

classwise_metric_dict = {'accuracy': MultiLabelAccuracy(averaging_type=None), 'f1': MultiLabelFBeta(beta=1, averaging_type=None),
                          'precision': MultiLabelPrecision(averaging_type=None), 'recall': MultiLabelRecall(averaging_type=None),
                          'confusion_matrix': MultiLabelConfusionMatrix()}

def get_metric_by_name(metric_name : str):
    return metric_dict[metric_name]

def get_metrics_by_names(metric_names : list):
    return [get_metric_by_name(metric_name) for metric_name in metric_names]

def get_multilabel_metric_by_name(metric_name : str):
    return multilabel_metric_dict[metric_name]

def get_multilabel_metrics_by_names(metric_names : list):
    return [get_multilabel_metric_by_name(metric_name) for metric_name in metric_names]

def get_classwise_metric_by_name(metric_name : str):
    return classwise_metric_dict[metric_name]

def get_classwise_metrics_by_names(metric_names : list):
    return [get_classwise_metric_by_name(metric_name) for metric_name in metric_names]