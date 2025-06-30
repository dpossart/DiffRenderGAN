#!/bin/bash

TAG="Ruehle2021TiO2"
MESH_PATH="utils/rendering/meshes/TiO2_2"
DATASET_PATH="datasets/DiffGAN_Ruehle2021_TiO2_dataset"

python train_gan.py \
    --tag="$TAG" \
    --mesh_path="$MESH_PATH" \
    --mesh_scale=1.0 \
    --emission1=1. \
    --emission2=0.1 \
    --dataset_path="$DATASET_PATH" \
    --spp=121 \
    --gaussdev=0.5 \
    --mesh_unit=0.25 \
    --n_epochs=50 \
    --mesh_max_agglomerates=5 \
    --p_sheen=-1 \
    --p_base_color=-1 \
    --p_roughness=-1 \
    --p_sheen_tint=-1 \
    --p_spec_tint=-1 \
    --s_base_color=-1 \
    --s_roughness=0.5 \
    --s_specular=0.0 \
    --mesh_base_sample_limit=2.5 \
    --poisson_placement \
    --lognormal 0.1 0.2 \
    --particle_bsdf_limit=".01,5." \
    --stage_bsdf_limit="0.01, 0.4" \
    --noise_limit="0.001, 0.1"
