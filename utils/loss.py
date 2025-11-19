import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
from collections.abc import Callable

from monai.networks import one_hot
from monai.utils import LossReduction

class FocalLoss(nn.Module):
    """
    Implémentation de la Focal Loss pour la segmentation multi-classes.
    """
    def __init__(self, gamma=2.0, weight=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.register_buffer('weight', weight)
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Calculer la perte Cross Entropy sans réduction, mais avec les poids de classe
        ce_loss = F.cross_entropy(inputs, targets, weight=self.weight, reduction='none', label_smoothing=0.1)
        
        # pt = p_t = exp(-ce_loss)
        pt = torch.exp(-ce_loss)
        
        # focal_loss = (1 - pt)^gamma * ce_loss
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        return focal_loss.sum() if self.reduction == 'sum' else focal_loss

class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.3, smooth=1e-6, class_weights=None):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        # Enregistrer les poids comme un buffer pour s'assurer qu'ils sont déplacés sur le bon appareil.
        self.register_buffer('class_weights', class_weights)

    def forward(self, inputs, targets):
        # inputs: (N, C, D, H, W) - logits
        # targets: (N, D, H, W) - long
        
        if targets.dtype == torch.long:
            # Convert hard labels to one-hot
            num_classes = inputs.shape[1]
            targets_one_hot = F.one_hot(targets, num_classes=num_classes).permute(0, 4, 1, 2, 3).float()

        else:
            # Assume targets are already soft/one-hot
            targets_one_hot = targets
        
        # Apply softmax to inputs to get probabilities
        probs = F.softmax(inputs, dim=1)

        # Flatten
        probs = probs.view(probs.shape[0], probs.shape[1], -1)
        targets_one_hot = targets_one_hot.view(targets_one_hot.shape[0], targets_one_hot.shape[1], -1)

        # True Positives, False Positives & False Negatives
        TP = (probs * targets_one_hot).sum(dim=2)
        FP = (probs * (1 - targets_one_hot)).sum(dim=2)
        FN = ((1 - probs) * targets_one_hot).sum(dim=2)

        tversky_index = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)

        # Nous voulons maximiser l'indice de Tversky, donc nous minimisons 1 - Tversky index
        per_class_loss = 1 - tversky_index

        if self.class_weights is not None:
            # Les poids devraient déjà être sur le bon appareil car ils sont un buffer.
            # Remodeler pour la diffusion (broadcasting): (C,) -> (1, C)
            weights = self.class_weights.view(1, -1)
            # Appliquer les poids à la perte par classe
            per_class_loss = per_class_loss * weights

        # We average over classes and batch
        return per_class_loss.mean()

class FocalTverskyPlusPlusLoss(nn.Module):
    """
    Compute the Tversky loss defined in:

        Sadegh et al. (2017) Tversky loss function for image segmentation
        using 3D fully convolutional deep networks. (https://arxiv.org/abs/1706.05721)

    Adapted from:
        https://github.com/NifTK/NiftyNet/blob/v0.6.0/niftynet/layer/loss_segmentation.py#L631

    """

    def __init__(
        self,
        include_background: bool = True,
        to_onehot_y: bool = False,
        sigmoid: bool = False,
        softmax: bool = False,
        other_act: Callable | None = None,
        alpha: float = 0.7,
        beta: float = 0.3,
        gamma_pp: float = 2.0,  # around 2-3 in the paper
        gamma_focal: float = 1.33,  # around 4/3 to 2 in the paper
        reduction: LossReduction | str = LossReduction.MEAN,
        smooth_nr: float = 1e-5,
        smooth_dr: float = 1e-5,
        batch: bool = False,
    ) -> None:
        super().__init__()
        if other_act is not None and not callable(other_act):
            raise TypeError(f"other_act must be None or callable but is {type(other_act).__name__}.")
        if int(sigmoid) + int(softmax) + int(other_act is not None) > 1:
            raise ValueError(
                "Incompatible values: more than 1 of [sigmoid=True, softmax=True, other_act is not None]."
            )
        self.include_background = include_background
        self.to_onehot_y = to_onehot_y
        self.sigmoid = sigmoid
        self.softmax = softmax
        self.other_act = other_act
        self.alpha = alpha
        self.beta = beta
        self.gamma_pp = gamma_pp
        self.gamma_focal = gamma_focal
        self.smooth_nr = float(smooth_nr)
        self.smooth_dr = float(smooth_dr)
        self.batch = batch
        self.reduction = reduction

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.sigmoid:
            input = torch.sigmoid(input)

        n_pred_ch = input.shape[1]
        if self.softmax:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `softmax=True` ignored.")
            else:
                input = torch.softmax(input, 1)

        if self.other_act is not None:
            input = self.other_act(input)

        if self.to_onehot_y:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `to_onehot_y=True` ignored.")
            else:
                # MONAI's one_hot function expects a channel dimension, so we add one.
                target = one_hot(target.unsqueeze(1), num_classes=n_pred_ch)

        if not self.include_background:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `include_background=False` ignored.")
            else:
                # if skipping background, removing first channel
                target = target[:, 1:]
                input = input[:, 1:]

        if target.shape != input.shape:
            raise AssertionError(
                f"ground truth has differing shape ({target.shape}) from input ({input.shape})"
            )

        p0 = input
        p1 = 1 - p0
        g0 = target
        g1 = 1 - g0

        # reducing only spatial dimensions (not batch nor channels)
        reduce_axis: list[int] = torch.arange(2, len(input.shape)).tolist()
        if self.batch:
            # reducing spatial dimensions and batch
            reduce_axis = [0] + reduce_axis

        tp = torch.sum(p0 * g0, reduce_axis)
        fp = self.alpha * torch.sum((p0 * g1) ** self.gamma_pp, reduce_axis)
        fn = self.beta * torch.sum((p1 * g0) ** self.gamma_pp, reduce_axis)
        numerator = tp + self.smooth_nr
        denominator = tp + fp + fn + self.smooth_dr

        score: torch.Tensor = (1.0 - numerator / denominator) ** self.gamma_focal

        if self.reduction == LossReduction.SUM.value:
            return torch.sum(score)  # sum over the batch and channel dims
        if self.reduction == LossReduction.NONE.value:
            return score  # returns [N, num_classes] losses
        if self.reduction == LossReduction.MEAN.value:
            return torch.mean(score)
        raise ValueError(
            f'Unsupported reduction: {self.reduction}, available options are ["mean", "sum", "none"].'
        )

class Loss(nn.Module):

    def __init__(self, loss_type='ce', tversky_alpha=0.7, tversky_beta=0.3, class_weights=None):
        super().__init__()
        self.loss_type = loss_type
        
        class_weight_tensor = None
        if class_weights is not None:
            class_weight_tensor = torch.tensor(class_weights, dtype=torch.float)

        if self.loss_type == 'focal_tversky':
            print("INFO: Utilisation de la perte hybride Focal + Tversky.")
            self.focal_loss = FocalLoss(gamma=2.0, weight=class_weight_tensor)
            self.tversky_loss = TverskyLoss(alpha=tversky_alpha, beta=tversky_beta, class_weights=class_weight_tensor)
        elif self.loss_type == 'ce_tversky':
            print("INFO: Utilisation de la perte hybride CE + Tversky (avec pondération fixe).")
            # NOTE: L'utilisation de `deterministic='warn'` permet d'utiliser le lissage de label.
            self.ce_loss = nn.CrossEntropyLoss(weight=class_weight_tensor, label_smoothing=0.1)
            self.tversky_loss = TverskyLoss(alpha=tversky_alpha, beta=tversky_beta, class_weights=class_weight_tensor)
        elif self.loss_type == 'tversky':
            print("INFO: Utilisation de la perte Tversky seule.")
            self.tversky_loss = TverskyLoss(alpha=tversky_alpha, beta=tversky_beta, class_weights=class_weight_tensor)
        elif self.loss_type == 'focal':
            print("INFO: Utilisation de la perte Focal seule.")
            self.focal_loss = FocalLoss(gamma=2.0, weight=class_weight_tensor)
        elif self.loss_type == 'focal_tversky_pp':
            print("INFO: Utilisation de la perte FocalTverskyPlusPlus.")
            self.ftpp_loss = FocalTverskyPlusPlusLoss(
                to_onehot_y=True,      # Les cibles sont des entiers (long)
                softmax=True,          # Le modèle retourne des logits bruts
                alpha=0.7,             # Valeur par défaut de Tversky
                beta=0.3,              # Valeur par défaut de Tversky
                gamma_pp=2.0,          # Valeur par défaut du papier
                gamma_focal=1.33       # Valeur par défaut du papier
            )
        elif self.loss_type == 'ce':
            print("INFO: Utilisation de la perte CrossEntropy standard.")
            # NOTE: L'utilisation de `deterministic='warn'` permet d'utiliser le lissage de label.
            self.ce_loss = nn.CrossEntropyLoss(weight=class_weight_tensor, label_smoothing=0.1)
        else:
            raise ValueError(f"Type de perte non supporté : '{self.loss_type}'.")

    def _compute_loss_dict(self, seg_preds, seg_targets):
        """Calcule le dictionnaire de pertes pour un seul tenseur de prédiction."""
        if self.loss_type == 'focal_tversky':
            loss_focal = self.focal_loss(seg_preds, seg_targets)
            loss_tversky = self.tversky_loss(seg_preds, seg_targets)
            return {'focal': loss_focal, 'tversky': loss_tversky}
        
        elif self.loss_type == 'ce_tversky':
            loss_ce = self.ce_loss(seg_preds, seg_targets)
            loss_tversky = self.tversky_loss(seg_preds, seg_targets)
            return {'ce': loss_ce, 'tversky': loss_tversky}
        
        elif self.loss_type == 'tversky':
            loss_tversky = self.tversky_loss(seg_preds, seg_targets)
            return {'tversky': loss_tversky}
        
        elif self.loss_type == 'focal':
            loss_focal = self.focal_loss(seg_preds, seg_targets)
            return {'focal': loss_focal}
        
        elif self.loss_type == 'ce':
            loss_ce = self.ce_loss(seg_preds, seg_targets)
            return {'ce': loss_ce}
        
        elif self.loss_type == 'focal_tversky_pp':
            loss_ftpp = self.ftpp_loss(seg_preds, seg_targets)
            return {'ftpp': loss_ftpp}
        return {}

    def forward(self, predictions, seg_targets):
        # Handle both dict and tensor predictions for robustness
        if isinstance(predictions, dict):
            seg_preds = predictions.get('seg')
            if seg_preds is None:
                raise ValueError("Prediction dictionary must contain a 'seg' key.")
        elif isinstance(predictions, (torch.Tensor, list)):
            seg_preds = predictions
        else:
            raise TypeError(f"Unsupported prediction format: {type(predictions)}. Expected dict, torch.Tensor, or list.")

        if isinstance(seg_preds, list):
            # Cas de la deep supervision
            # On ne prend que les deux dernières sorties (avant-dernière et dernière résolution)
            outputs_to_use = seg_preds[-2:]
            
            total_losses = {}
            # Poids fixes pour les deux sorties : 1.0 pour l'avant-dernière, 1.0 pour la dernière.
            weights = [1.0, 1.0]

            # Déterminer si seg_targets est déjà one-hot (e.g., après Mixup)
            # Si seg_targets a 5 dimensions (N, C, D, H, W) et est float, c'est probablement one-hot.
            is_seg_targets_one_hot = (seg_targets.ndim == seg_preds[0].ndim and seg_targets.dtype == torch.float)
            
            # Déterminer la forme spatiale de la cible originale
            target_spatial_shape = seg_targets.shape[2:] if is_seg_targets_one_hot else seg_targets.shape[1:]

            for i, pred in enumerate(outputs_to_use):
                # Sous-échantillonner la cible si nécessaire. Les prédictions vont de basse à haute résolution.
                if pred.shape[2:] != target_spatial_shape:
                    if is_seg_targets_one_hot:
                        # Si déjà one-hot, interpoler directement (mode trilinear pour les probabilités)
                        target_down = F.interpolate(seg_targets, size=pred.shape[2:], mode='trilinear', align_corners=False)
                    else:
                        # Si pas one-hot, unsqueeze channel et interpoler (mode nearest pour les labels entiers)
                        target_down = F.interpolate(seg_targets.unsqueeze(1).float(), size=pred.shape[2:], mode='nearest').squeeze(1).long()
                else:
                    target_down = seg_targets

                level_loss_dict = self._compute_loss_dict(pred, target_down)
                
                for loss_name, loss_value in level_loss_dict.items():
                    if loss_name not in total_losses:
                        total_losses[loss_name] = 0.0
                    total_losses[loss_name] += loss_value * weights[i]
            
            return total_losses
        else:
            # Cas d'une sortie unique
            return self._compute_loss_dict(seg_preds, seg_targets)