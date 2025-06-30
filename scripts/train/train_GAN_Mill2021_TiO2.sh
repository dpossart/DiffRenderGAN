#!/bin/bash

TAG="Mill2021TiO2"
MESH_PATH="utils/rendering/meshes/TiO2"
DATASET_PATH="datasets/DiffGAN_Mill2021_TiO2_dataset"

python train_gan.py \
    --tag="$TAG" \
    --mesh_position3d \
    --mesh_path="$MESH_PATH" \
    --mesh_scale=1.7 \
    --emission1=1. \
    --emission2=0.1 \
    --dataset_path="$DATASET_PATH" \
    --spp=16 \
    --gaussdev=3. \
    --mesh_unit=0.7 \
    --n_epochs=50 \
    --p_metallic=0 \
    --p_spec_tint=-1 \
    --p_sheen=-1 \
    --p_base_color=-1 \
    --p_roughness=-1 \
    --p_sheen_tint=-1 \
    --s_base_color=-1 \
    --s_roughness=0.5 \
    --s_specular=0. \
    --mesh_base_sample_limit=1 \
    --poisson_placement \
    --lognormal 0 0.1 \
    --particle_bsdf_limit=".01,5." \
    --stage_bsdf_limit="0.01, 0.3" \
    --noise_limit="0.001, 0.2"
