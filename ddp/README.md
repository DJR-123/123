# Event-guided HDR Reconstruction with Diffusion Prior

Yixin Yang, Jiawei Zhang, Yang Zhang, Yunxuan Wei, Dongqing Zou, Jimmy Ren, and Boxin Shi.

## Requirements

- Python 3.10
- CUDA-compatible GPU

## Installation

### 1. Create conda environment

```bash
conda create -n hdrev-diffu python=3.10
conda activate hdrev-diffu
```

### 2. Install PyTorch

```bash
pip3 install torch torchvision torchaudio
```

### 3. Install other dependencies

```bash
pip install lpips wandb diffusers omegaconf transformers opencv-python h5py hdf5plugin matlablib
```

## Dataset

Please downloading our training dataset (https://www.modelscope.cn/datasets/Mcallor/HDR-EVS/) and specify the dataroot in configuration file.

## Usage

### Training

#### Single GPU Training

**Main Model Training**

```bash
python train.py \
    --name "experiment_name" \
    --config "config/Train_stage1+2.yaml"
```

**Upsampler Training**

First, run inference sampling results without refinement (change the dataroot in config file to match the training data):
```bash
python up_test.py \
    --name "test_experiment" \
    --config "config/up_Test.yaml" \
    --pretrained_unet "path/to/trained/model.ckpt" \
    --save
```

Then, start training based on the saved latents (change the dataroot in config file to match the saved latents):
```bash
python train_upsampler.py \
    --name "upsampler_experiment" \
    --config "config/Train_upsampler.yaml" \
    --pretrained_unet "path/to/trained/model.ckpt"
```

#### Multi-GPU Training (DDP)

This project supports Distributed Data Parallel (DDP) for multi-GPU training and testing. DDP maintains the same training results while significantly accelerating the process.

**Single Node Multi-GPU Training**

```bash
# Main model training with 2 GPUs
torchrun --nproc_per_node=2 train.py \
    --name "experiment_name" \
    --config "config/Train_stage1+2.yaml"

# Upsampler training with 4 GPUs
torchrun --nproc_per_node=4 train_upsampler.py \
    --name "upsampler_experiment" \
    --config "config/Train_upsampler.yaml" \
    --pretrained_unet "path/to/trained/model.ckpt"
```

**Multi-Node Multi-GPU Training**

```bash
# On node 0 (master node)
torchrun --nnodes=2 --nproc_per_node=4 \
    --node_rank=0 --master_addr="10.0.0.1" --master_port=29500 \
    train.py --name "experiment_name" --config "config/Train_stage1+2.yaml"

# On node 1
torchrun --nnodes=2 --nproc_per_node=4 \
    --node_rank=1 --master_addr="10.0.0.1" --master_port=29500 \
    train.py --name "experiment_name" --config "config/Train_stage1+2.yaml"
```

**DDP Notes**:
- Total batch size = `batch_size` × `num_gpus`, each GPU uses the same batch size as single GPU
- **Learning rate is linearly scaled**: effective lr = `config.learning_rate` × `num_gpus` (to maintain training consistency)
- **Training epochs aligned**: optimizer steps per GPU = `config.max_train_steps` / `num_gpus`, so epochs remain same as single GPU
- Model synchronization is handled automatically by DDP
- Only rank 0 (main process) handles logging, checkpoint saving, and validation
- Training results remain consistent with single GPU training

### Testing

#### Single GPU Testing

```bash
python test.py \
    --name "test_experiment" \
    --config "config/Test.yaml" \
    --pretrained_unet "path/to/trained/model.ckpt" \
    --pretrained_upsample "path/to/trained/upsampler.ckpt"
```

#### Multi-GPU Testing (DDP)

```bash
torchrun --nproc_per_node=2 test.py \
    --name "test_experiment" \
    --config "config/Test.yaml" \
    --pretrained_unet "path/to/trained/model.ckpt" \
    --pretrained_upsample "path/to/trained/upsampler.ckpt"
```

The test data will be distributed across GPUs, and only rank 0 will save the final results.

## Configuration

### Main Configuration Parameters

- `output_dir`: Output directory
- `pretrained_model_path`: Path to pre-trained Stable Diffusion model
- `pretrained_controlnet_model_path`: Path to pre-trained ControlNet model
- `conditioning_channels`: Conditioning channel configuration
- `learning_rate`: Learning rate
- `batch_size`: Batch size
- `max_train_steps`: Maximum training steps

### Dataset Configuration

- `dataset_type`: Dataset type (evs_image_h5single)
- `dataroot`: Data root directory
- `patch_size`: Image patch size
- `num_bins`: Number of event time bins
- `event_representation`: Event representation method

## Model Architecture

### HDRev_Encoder
Specially designed event data encoder for processing temporal information from event cameras.

### OursControlNetModel
ControlNet-based conditional control model supporting multiple input types:
- Event data (evs)
- Low dynamic range images (ldr)
- Event + low dynamic range images (evs+ldr)

### OursAutoencoderKL
Improved autoencoder for image upsampling and detail enhancement.

## Data Format

The project supports HDF5 format datasets containing:
- Event data (pixel_events)
- Low dynamic range images (pixel_images)
- High dynamic range images (target_images)

## Citation

```bibtex
@inproceedings{yang2025eventguided,
  author    = {Yang, Yixin and Zhang, Jiawei and Zhang, Yang and Wei, Yunxuan and Zou, Dongqing and Ren, Jimmy and Shi, Boxin},
  title     = {Event-guided HDR Reconstruction with Diffusion Priors},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year      = {2025},
}
```

## Contact

For questions or suggestions, please contact us via:
- Email: yangyixin93@pku.edu.cn
