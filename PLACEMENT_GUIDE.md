# A7-P2 Paper-Style Regularization Drop-In

Extract this archive over the repository root. It updates four runtime files,
two existing tests, and adds the A7-P2 experiment wrappers.

## Controlled intervention

A7-P2 preserves the A7-P1 network topology, frozen SegFormer/Monodepth2 cues,
00-08 source-only split, MAE objective, learning rate, batch size, and 30-epoch
duration. It makes two paper-protocol corrections: explicit layer-scoped L2
and element-wise dropout semantics.

```text
0.02 * sum(rotation_head.conv.weight ** 2)
+ 0.02 * sum(translation_head.conv.weight ** 2)
+ 0.02 * sum(rotation_head.dense.weight ** 2)
+ 0.02 * sum(translation_head.dense.weight ** 2)
```

Biases, A-ResUNet, BatchNorm, SegFormer, and Monodepth2 are excluded. Global
Adam weight decay remains zero. The paper explicitly gives 0.02 for the
post-fusion Conv2D and says the dense layer is also L2-regularized without
stating a second coefficient; A7-P2 records the controlled assumption that the
dense coefficient is also 0.02. Both coefficients are independent environment
variables and checkpoint fields.

The probability remains 0.2, but `Dropout2d` is changed to ordinary
`nn.Dropout`. Keras `Dropout` is element-wise; PyTorch `Dropout2d` masks entire
channels. Because the post-fusion head has only one channel and training uses
batch size 1, `Dropout2d(p=0.2)` can erase the complete head representation for
roughly one fifth of samples.

## Updated existing files

- `deepdct/training/train_one_epoch.py`
- `deepdct/training/validate_one_epoch.py`
- `deepdct/models/pose_head.py`
- `scripts/train_deepdct_vo.py`
- `scripts/evaluate_deepdct_vo.py`
- `tests/integration/test_single_batch_training_step.py`
- `tests/models/test_deepdct_forward.py`

## New wrappers

- `experiments/track_a/paper_reproduction/a7_p2_paper_regularization/a7_p2_common.sh`
- `experiments/track_a/paper_reproduction/a7_p2_paper_regularization/preflight.sh`
- `experiments/track_a/paper_reproduction/a7_p2_paper_regularization/train.sh`
- `experiments/track_a/paper_reproduction/a7_p2_paper_regularization/evaluate.sh`
- `experiments/track_a/paper_reproduction/a7_p2_paper_regularization/evaluate_gt_rotation.sh`
- `experiments/track_a/paper_reproduction/a7_p2_paper_regularization/audit_rotation_attribution.sh`
- `experiments/track_a/paper_reproduction/a7_p2_paper_regularization/run_a7_p2.sh`

## Install and run

```bash
cd /media/sherifdeen/ext_hd250/projects/deepdct-vo
unzip -o a7_p2_regularization_dropin.zip

chmod +x \
  experiments/track_a/paper_reproduction/a7_p2_paper_regularization/*.sh

experiments/track_a/paper_reproduction/a7_p2_paper_regularization/preflight.sh
experiments/track_a/paper_reproduction/a7_p2_paper_regularization/run_a7_p2.sh
```

`train.sh` refuses to overwrite an existing A7-P2 `latest.pt`. This enforces a
fresh initialization and prevents accidental continuation from unregularized
A7-P1.

To run only training after preflight:

```bash
experiments/track_a/paper_reproduction/a7_p2_paper_regularization/train.sh
```

To override the two independently recorded coefficients:

```bash
A7_P2_HEAD_CONV_L2=0.02 \
A7_P2_HEAD_DENSE_L2=0.01 \
experiments/track_a/paper_reproduction/a7_p2_paper_regularization/train.sh
```

## Loss reporting

Training and validation report:

- `pose_data_loss`: weighted rotation + translation + optional geometry loss.
- `regularization_loss`: explicit selected-kernel L2 penalty.
- `total_loss`: the actual optimization objective, equal to their sum.

Evaluation never adds a regularization penalty. It prints the checkpoint's
coefficients and scope only to make P1/P2 result attribution unambiguous.
