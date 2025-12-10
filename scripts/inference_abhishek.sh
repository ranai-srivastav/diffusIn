# Running inference for Abhishek's experiments
python src/diffusion_inference.py \
    --file_dir_path /home/student/Documents/parth/planning_project/diffusIn/data_recorded/diffusion_policy_models_20251208_205814/ \
    --checkpoint_name diffusion_model_checkpoint_latest.pth \
    --device cuda:0 \
    --save_gif \
    --render_onscreen