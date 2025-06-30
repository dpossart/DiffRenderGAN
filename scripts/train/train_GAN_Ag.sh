#!/bin/bash

TAG="Ag"
MESH_PATH="utils/rendering/meshes/Ag"
DATASET_PATH="datasets/DiffGAN_Ag_dataset"

python train_gan.py \
    --tag="$TAG" \
    --mesh_path="$MESH_PATH" \
    --mesh_scale=1.25 \
    --emission1=1. \
    --emission2=0.1 \
    --dataset_path="$DATASET_PATH" \
    --spp=121 \
    --gaussdev=0.5 \
    --mesh_unit=1 \
    --n_epochs=50 \
    --mesh_max_agglomerates=1 \
    --p_metallic=0 \
    --p_spec_tint=-1 \
    --p_sheen=-1 \
    --p_base_color=-1 \
    --p_roughness=-1 \
    --p_sheen_tint=-1 \
    --s_base_color=-1 \
    --s_roughness=0.5 \
    --s_specular=0. \
    --mesh_base_sample_limit=5.0 \
    --random_placement \
    --lognormal 0 0.1 \
    --particle_bsdf_limit=".01,5." \
    --stage_bsdf_limit="0.01, 5.0" \
    --noise_limit="0.001, 1"
