import os
from pathlib import Path
import re
import random
import transformers
from tokenizers import BertWordPieceTokenizer
from transformers import BertTokenizer
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam
from torchtext.vocab import vocab
import itertools
import math
import numpy as np
import pandas as pd
from collections import Counter
import typing
import time
from datetime import datetime
from sklearn.model_selection import train_test_split


class PositionalEmbedding(torch.nn.Module):
    def __init__(self, d_model, max_len=128):
        super().__init__()
        # Initialize positional encodings
        pe = torch.zeros(max_len, d_model).float()
        pe.requires_grad = False

        for pos in range(max_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** (2 * i / d_model)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** (2 * (i + 1) / d_model)))

        # Add batch dimension
        self.register_buffer('pe', pe.unsqueeze(0))  # Use register_buffer to save on device

    def forward(self, x):
        # Ensure positional embeddings are on the same device as input
        return self.pe[:, :x.size(1), :].to(x.device)

class BERTEmbedding(nn.Module):
    """
    BERT Embedding includes:
    1. Token Embedding: Standard embedding matrix for tokens.
    2. Positional Embedding: Adds positional information via sine and cosine functions.
    3. Segment Embedding: Adds segment information (e.g., sentence A = 1, sentence B = 2).
    The sum of these embeddings is returned as the final output.
    """

    def __init__(self, vocab_size, embed_size, seq_len=64, dropout=0.1):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.segment_embedding = nn.Embedding(3, embed_size, padding_idx=0)
        self.position_embedding = PositionalEmbedding(d_model=embed_size, max_len=seq_len)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, sequence, segment_label):
        device = next(self.parameters()).device
        sequence = sequence.to(device)
        segment_label = segment_label.to(device)

        token_embeds = self.token_embedding(sequence)
        position_embeds = self.position_embedding(sequence)
        segment_embeds = self.segment_embedding(segment_label)

        embeddings = token_embeds + position_embeds + segment_embeds
        return self.dropout(embeddings)
    
class MultiHeadedAttention(nn.Module):
    def __init__(self, heads, d_model, dropout=0.1):
        super().__init__()

        assert d_model % heads == 0
        self.d_k = d_model // heads
        self.heads = heads
        self.dropout = nn.Dropout(dropout)

        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.output_linear = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, mask):
        query, key, value = self.query(query), self.key(key), self.value(value)

        # Reshape to (batch_size, heads, max_len, d_k)
        query = query.view(query.shape[0], -1, self.heads, self.d_k).permute(0, 2, 1, 3)
        key = key.view(key.shape[0], -1, self.heads, self.d_k).permute(0, 2, 1, 3)
        value = value.view(value.shape[0], -1, self.heads, self.d_k).permute(0, 2, 1, 3)

        # Compute attention scores
        scores = torch.matmul(query, key.permute(0, 1, 3, 2)) / math.sqrt(query.size(-1))

        # Apply mask
        scores = scores.masked_fill(mask == 0, -1e9)

        # Attention weights and dropout
        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)

        # Apply attention weights to value
        context = torch.matmul(weights, value)

        # Reshape back to (batch_size, max_len, d_model)
        context = context.permute(0, 2, 1, 3).contiguous().view(context.shape[0], -1, self.heads * self.d_k)

        return self.output_linear(context)


class FeedForward(nn.Module):
    def __init__(self, d_model, middle_dim=2048, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, middle_dim)
        self.fc2 = nn.Linear(middle_dim, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x):
        x = self.activation(self.fc1(x))
        return self.fc2(self.dropout(x))


class EncoderLayer(nn.Module):
    def __init__(self, d_model=768, heads=12, feed_forward_hidden=768 * 4, dropout=0.1):
        super().__init__()

        self.layernorm = nn.LayerNorm(d_model)
        self.self_multihead = MultiHeadedAttention(heads, d_model)
        self.feed_forward = FeedForward(d_model, middle_dim=feed_forward_hidden)
        self.dropout = nn.Dropout(dropout)

    def forward(self, embeddings, mask):
        # Self-attention
        attended = self.dropout(self.self_multihead(embeddings, embeddings, embeddings, mask))
        attended = self.layernorm(attended + embeddings)

        # Feed-forward network
        ff_out = self.dropout(self.feed_forward(attended))
        return self.layernorm(ff_out + attended)

class BERT(nn.Module):
    """
    BERT model: Bidirectional Encoder Representations from Transformers.
    """

    def __init__(self, vocab_size, d_model=768, n_layers=12, heads=12, dropout=0.1):
        """
        Initializes the BERT model.

        :param vocab_size: Total vocabulary size
        :param d_model: Hidden size of the BERT model
        :param n_layers: Number of transformer layers
        :param heads: Number of attention heads
        :param dropout: Dropout rate
        """
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers

        # Feed-forward network hidden size is 4 times the model's hidden size
        self.feed_forward_hidden = d_model * 4

        # Embedding layer: sum of token, segment, and positional embeddings
        self.embedding = BERTEmbedding(vocab_size=vocab_size, embed_size=d_model)

        # Multi-layer transformer blocks
        self.encoder_blocks = nn.ModuleList(
            [EncoderLayer(d_model, heads, self.feed_forward_hidden, dropout) for _ in range(n_layers)]
        )

    def forward(self, x, segment_info):
        # Attention mask for padded tokens
        mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)

        # Embed the input sequence
        x = self.embedding(x, segment_info)

        # Pass through multiple transformer layers
        for encoder in self.encoder_blocks:
            x = encoder(x, mask)
        return x


class NextSentencePrediction(nn.Module):
    """
    2-class classification model: predicts whether the second sentence follows the first.
    """

    def __init__(self, hidden_size):
        """
        Initializes the next sentence prediction model.

        :param hidden_size: The hidden size of the BERT model output
        """
        super().__init__()
        self.linear = nn.Linear(hidden_size, 2)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        # Use only the [CLS] token (first token) for classification
        return self.softmax(self.linear(x[:, 0]))


class MaskedLanguageModel(nn.Module):
    """
    Predicts the original token for each masked token in the input sequence.
    This is a multi-class classification problem where the number of classes is the vocabulary size.
    """

    def __init__(self, hidden_size, vocab_size):
        """
        Initializes the masked language model.

        :param hidden_size: The hidden size of the BERT model output
        :param vocab_size: The size of the vocabulary for the classification task
        """
        super().__init__()
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        return self.softmax(self.linear(x))


class BERTLM(nn.Module):
    """
    BERT Language Model combining Next Sentence Prediction (NSP) and Masked Language Modeling (MLM).
    """

    def __init__(self, bert: BERT, vocab_size):
        """
        Initializes the BERT Language Model with NSP and MLM.

        :param bert: The pre-trained BERT model
        :param vocab_size: The size of the vocabulary for the masked language model
        """
        super().__init__()
        self.bert = bert
        self.next_sentence = NextSentencePrediction(self.bert.d_model)
        self.mask_lm = MaskedLanguageModel(self.bert.d_model, vocab_size)

    def forward(self, x, segment_label):
        # Pass through the BERT model to obtain embeddings
        x = self.bert(x, segment_label)

        # Get outputs for both NSP and MLM tasks
        return self.next_sentence(x), self.mask_lm(x)
    
class ScheduledOptim():
    """A simple wrapper for learning rate scheduling."""

    def __init__(self, optimizer, d_model, n_warmup_steps):
        """
        Initializes the learning rate scheduler.

        :param optimizer: The optimizer for which learning rate is scheduled
        :param d_model: The dimension of the model (used to initialize learning rate)
        :param n_warmup_steps: Number of warm-up steps for learning rate scheduling
        """
        self._optimizer = optimizer
        self.n_warmup_steps = n_warmup_steps
        self.n_current_steps = 0
        self.init_lr = np.power(d_model, -0.5)

    def step_and_update_lr(self):
        """Step with the optimizer and update the learning rate."""
        self._update_learning_rate()
        self._optimizer.step()

    def zero_grad(self):
        """Zero out the gradients in the optimizer."""
        self._optimizer.zero_grad()

    def _get_lr_scale(self):
        """Calculates the scaling factor for the learning rate."""
        return np.min([
            np.power(self.n_current_steps, -0.5),
            np.power(self.n_warmup_steps, -1.5) * self.n_current_steps
        ])

    def _update_learning_rate(self):
        """Updates the learning rate based on the current step."""
        self.n_current_steps += 1
        lr = self.init_lr * self._get_lr_scale()

        # Update the learning rate for each parameter group in the optimizer
        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr