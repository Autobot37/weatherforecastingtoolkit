#!/usr/bin/env bash
set -e
CONFIG=config.yaml
PYTHON_SCRIPT=train.py  # replace with actual script
PROJECT_NAME="ae_gan_night"

declare -a RUNS=(
  "project_name=$PROJECT_NAME experiment_name=ae_conv2_disc_1_lt_256 model.name=convautoencoder2 lpips.disc_start=1.0 lpips.disc_weight=0.0 ConvAutoencoder2.latent_dim=256"
  "project_name=$PROJECT_NAME experiment_name=ae_attn_disc_0_lt_512 model.name=attentionchargedautoencoder lpips.disc_start=0.0 lpips.disc_weight=1.0 AttentionChargedAutoencoder.latent_dim=512"
  "project_name=$PROJECT_NAME experiment_name=ae_attn_disc_1_lt_512 model.name=attentionchargedautoencoder lpips.disc_start=1.0 lpips.disc_weight=0.0 AttentionChargedAutoencoder.latent_dim=512"
)

for OVERRIDE in "${RUNS[@]}"; do
  echo "Running with: $OVERRIDE"
  python "$PYTHON_SCRIPT" $OVERRIDE
done