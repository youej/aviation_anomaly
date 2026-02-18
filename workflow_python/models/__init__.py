"""
Model architectures for aviation anomaly prediction.

Tier 1 — Aviation Domain Baselines:
    BaseGRU (DT-MIL), MultiHeadCnnRnn (MHCNN-RNN)

Tier 2 — Cross-Domain SOTA:
    MultiHeadAttention (Block 1 & 2), GatedTransformerNetwork (GTN), InceptionTime

Tier 3 — Early Classification:
    EARLIEST (LSTM + RL halting policy)
"""

from models.base_gru import BaseGRU
from models.mhcnn_rnn import MultiHeadCnnRnn
from models.multi_head_attention import MultiHeadAttention
from models.gated_transformer import GatedTransformerNetwork, PositionalEncodingLayer
from models.inception_time import InceptionTime, InceptionTimeEnsemble
from models.earliest import EARLIEST

__all__ = [
    'BaseGRU',
    'MultiHeadCnnRnn',
    'MultiHeadAttention',
    'GatedTransformerNetwork',
    'InceptionTime',
    'InceptionTimeEnsemble',
    'EARLIEST',
]
