"""
ANAO Loss: Anatomy-Negation Aware Objective for medical cross-modal alignment.
Extends standard InfoNCE loss with anatomy location and negation awareness penalties.
"""

import torch
import torch.nn as nn
from torch.nn import functional as F

try:
    import torch.distributed.nn
    from torch import distributed as dist
    has_distributed = True
except ImportError:
    has_distributed = False

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None

from timm.loss import LabelSmoothingCrossEntropy


def gather_features(
        image_features,
        text_features,
        anat_labels=None,
        neg_labels=None,
        local_loss=False,
        gather_with_grad=False,
        rank=0,
        world_size=1,
):
    assert has_distributed, 'torch.distributed did not import correctly.'
    if gather_with_grad:
        all_image_features = torch.cat(torch.distributed.nn.all_gather(image_features), dim=0)
        all_text_features = torch.cat(torch.distributed.nn.all_gather(text_features), dim=0)
        if anat_labels is not None:
            all_anat_labels = torch.cat(torch.distributed.nn.all_gather(anat_labels), dim=0)
        else:
            all_anat_labels = None
        if neg_labels is not None:
            all_neg_labels = torch.cat(torch.distributed.nn.all_gather(neg_labels), dim=0)
        else:
            all_neg_labels = None
    else:
        gathered_image_features = [torch.zeros_like(image_features) for _ in range(world_size)]
        gathered_text_features = [torch.zeros_like(text_features) for _ in range(world_size)]
        dist.all_gather(gathered_image_features, image_features)
        dist.all_gather(gathered_text_features, text_features)
        if not local_loss:
            gathered_image_features[rank] = image_features
            gathered_text_features[rank] = text_features
        all_image_features = torch.cat(gathered_image_features, dim=0)
        all_text_features = torch.cat(gathered_text_features, dim=0)

        if anat_labels is not None:
            gathered_anat = [torch.zeros_like(anat_labels) for _ in range(world_size)]
            dist.all_gather(gathered_anat, anat_labels)
            if not local_loss:
                gathered_anat[rank] = anat_labels
            all_anat_labels = torch.cat(gathered_anat, dim=0)
        else:
            all_anat_labels = None

        if neg_labels is not None:
            gathered_neg = [torch.zeros_like(neg_labels) for _ in range(world_size)]
            dist.all_gather(gathered_neg, neg_labels)
            if not local_loss:
                gathered_neg[rank] = neg_labels
            all_neg_labels = torch.cat(gathered_neg, dim=0)
        else:
            all_neg_labels = None

    return all_image_features, all_text_features, all_anat_labels, all_neg_labels


class AnaOLoss(nn.Module):
    """
    Anatomy-Negation Aware Objective loss for medical image-text alignment.

    L_total = L_clip + lambda_anat * L_anatomy + lambda_neg * L_negation

    - L_anatomy: penalizes high-similarity cross-modal pairs when anatomical
      locations differ (e.g., "left lower lobe" report vs right-side image)
    - L_negation: penalizes high-similarity cross-modal pairs when negation
      status of key clinical findings differs (e.g., "no pneumothorax" vs
      image showing pneumothorax)
    """

    def __init__(
            self,
            anat_weight=0.1,
            neg_weight=0.1,
            local_loss=False,
            gather_with_grad=False,
            rank=0,
            world_size=1,
            smoothing=0.,
    ):
        super().__init__()
        self.anat_weight = anat_weight
        self.neg_weight = neg_weight
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.rank = rank
        self.world_size = world_size
        self.label_smoothing_cross_entropy = (
            LabelSmoothingCrossEntropy(smoothing=smoothing) if smoothing > 0 else None
        )
        self.prev_num_logits = 0
        self.labels = {}

    def _compute_clip_loss(self, logits_per_image, logits_per_text, device):
        num_logits = logits_per_image.shape[0]
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if self.world_size > 1 and self.local_loss:
                labels = labels + num_logits * self.rank
            self.labels[device] = labels
            self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]

        if self.label_smoothing_cross_entropy:
            total_loss = (
                self.label_smoothing_cross_entropy(logits_per_image, labels) +
                self.label_smoothing_cross_entropy(logits_per_text, labels)
            ) / 2
        else:
            total_loss = (
                F.cross_entropy(logits_per_image, labels) +
                F.cross_entropy(logits_per_text, labels)
            ) / 2

        i2t_acc = (logits_per_image.argmax(-1) == labels).float().mean()
        t2i_acc = (logits_per_text.argmax(-1) == labels).float().mean()
        return total_loss, {"i2t": i2t_acc, "t2i": t2i_acc}

    def _compute_anatomy_loss(self, similarity_matrix, anat_labels):
        """
        Penalize high similarity between pairs with mismatched anatomical locations.

        Args:
            similarity_matrix: [N, N] cosine similarity scaled by logit_scale
            anat_labels: [N] integer labels encoding anatomical location
                         0: left_upper, 1: left_lower, 2: right_upper,
                         3: right_lower, 4: bilateral, 5: unspecified/none
        Returns:
            scalar penalty
        """
        N = similarity_matrix.shape[0]
        anat_i = anat_labels.unsqueeze(0).expand(N, N)  # [N, N]
        anat_j = anat_labels.unsqueeze(1).expand(N, N)  # [N, N]

        # Mask: 1 when anatomical locations differ AND neither is "unspecified"
        valid_mask = (anat_i != 5) & (anat_j != 5)
        mismatch_mask = (anat_i != anat_j) & valid_mask

        if mismatch_mask.sum() == 0:
            return torch.tensor(0.0, device=similarity_matrix.device)

        # Penalize high-similarity mismatched pairs
        penalized = similarity_matrix[mismatch_mask]
        # Use relu to only penalize positive similarities (cosine similarity above 0)
        anatomy_loss = F.relu(penalized).mean()
        return anatomy_loss

    def _compute_negation_loss(self, similarity_matrix, neg_labels):
        """
        Penalize high similarity between pairs where negation states of key
        findings differ significantly.

        Args:
            similarity_matrix: [N, N] cosine similarity scaled by logit_scale
            neg_labels: [N, K] binary indicators for K key clinical findings
                       (1 = finding present, 0 = finding absent/negated)
        Returns:
            scalar penalty
        """
        if neg_labels.dim() == 1:
            return torch.tensor(0.0, device=similarity_matrix.device)

        N = similarity_matrix.shape[0]

        # Normalize shape: DataLoader may produce [N, K] or [K, N].
        # Always convert to [N, K] where N=batch, K=num_findings.
        if neg_labels.shape[0] != N and neg_labels.shape[1] == N:
            neg_labels = neg_labels.T
        K = neg_labels.shape[1]

        neg_i = neg_labels.unsqueeze(0).expand(N, N, K)  # [N, N, K]
        neg_j = neg_labels.unsqueeze(1).expand(N, N, K)  # [N, N, K]

        # Hamming distance normalized by K: fraction of findings with differing negation
        neg_diff = (neg_i != neg_j).float().mean(dim=-1)  # [N, N]

        # Weight by similarity: high-sim pairs with high negation difference get high penalty
        negation_loss = (neg_diff * F.relu(similarity_matrix)).mean()
        return negation_loss

    def forward(self, image_features, text_features, logit_scale=1.,
                anat_labels=None, neg_labels=None):
        device = image_features.device

        if self.world_size > 1:
            all_image_features, all_text_features, all_anat_labels, all_neg_labels = \
                gather_features(
                    image_features, text_features,
                    anat_labels=anat_labels,
                    neg_labels=neg_labels,
                    local_loss=self.local_loss,
                    gather_with_grad=self.gather_with_grad,
                    rank=self.rank,
                    world_size=self.world_size,
                )

            if self.local_loss:
                logits_per_image = logit_scale * image_features @ all_text_features.T
                logits_per_text = logit_scale * text_features @ all_image_features.T
                # For local loss, we use local anat/neg labels for penalty computation
                sim_for_anat = logit_scale * image_features @ all_text_features.T
                sim_for_neg = sim_for_anat
                anat_for_loss = anat_labels
                neg_for_loss = neg_labels
            else:
                logits_per_image = logit_scale * all_image_features @ all_text_features.T
                logits_per_text = logits_per_image.T
                sim_for_anat = logits_per_image
                sim_for_neg = logits_per_image
                anat_for_loss = all_anat_labels
                neg_for_loss = all_neg_labels
        else:
            logits_per_image = logit_scale * image_features @ text_features.T
            logits_per_text = logit_scale * text_features @ image_features.T
            sim_for_anat = logits_per_image
            sim_for_neg = logits_per_image
            anat_for_loss = anat_labels
            neg_for_loss = neg_labels

        clip_loss, acc = self._compute_clip_loss(logits_per_image, logits_per_text, device)

        anatomy_loss = torch.tensor(0.0, device=device)
        negation_loss = torch.tensor(0.0, device=device)

        if self.anat_weight > 0 and anat_for_loss is not None:
            anatomy_loss = self._compute_anatomy_loss(sim_for_anat, anat_for_loss)

        if self.neg_weight > 0 and neg_for_loss is not None:
            negation_loss = self._compute_negation_loss(sim_for_neg, neg_for_loss)

        total_loss = clip_loss + self.anat_weight * anatomy_loss + self.neg_weight * negation_loss

        acc["anatomy_loss"] = anatomy_loss
        acc["negation_loss"] = negation_loss

        return total_loss, acc
