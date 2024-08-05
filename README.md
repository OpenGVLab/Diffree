# Diffree
Official PyTorch implement of paper "Diffree: Text-Guided Shape Free Object Inpainting with Diffusion Model"

<p align="center">
  <a href="https://opengvlab.github.io/Diffree/"><u>[🌐 Project Page]</u></a>
  &nbsp;&nbsp;
  <a href="https://huggingface.co/datasets/LiruiZhao/OABench"><u>[🗞️ Dataset]</u></a>
  &nbsp;&nbsp;
  <a href="https://drive.google.com/file/d/1AdIPA5TK5LB1tnqqZuZ9GsJ6Zzqo2ua6/view"><u>[🎥 Video]</u></a>
  &nbsp;&nbsp;
  <a href="https://arxiv.org/pdf/2407.16982"><u>[📜 Arxiv]</u></a>
  &nbsp;&nbsp;
  <a href="https://huggingface.co/spaces/LiruiZhao/Diffree"><u>[🤗 Hugging Face Demo]</u></a>
</p>

## Abstract

<details><summary>CLICK for the full abstract</summary>

> This paper addresses an important problem of object addition for images with only text guidance. It is challenging because the new object must be integrated seamlessly into the image with consistent visual context, such as lighting, texture, and spatial location. While existing text-guided image inpainting methods can add objects, they either fail to preserve the background consistency or involve cumbersome human intervention in specifying bounding boxes or user-scribbled masks. To tackle this challenge, we introduce Diffree, a Text-to-Image (T2I) model that facilitates text-guided object addition with only text control. To this end, we curate OABench, an exquisite synthetic dataset by removing objects with advanced image inpainting techniques. OABench comprises 74K real-world tuples of an original image, an inpainted image with the object removed, an object mask, and object descriptions. Trained on OABench using the Stable Diffusion model with an additional mask prediction module, Diffree uniquely predicts the position of the new object and achieves object addition with guidance from only text. Extensive experiments demonstrate that Diffree excels in adding new objects with a high success rate while maintaining background consistency, spatial appropriateness, and object relevance and quality.
> </details>

We are open to any suggestions and discussions and feel free to contact us through [liruizhao@stu.xmu.edu.cn](mailto:liruizhao@stu.xmu.edu.cn).

## News
- [2024/07] Release inference code and <a href="https://huggingface.co/LiruiZhao/Diffree">checkpoint</a>
- [2024/07] Release <a href="https://huggingface.co/spaces/LiruiZhao/Diffree">🤗 Hugging Face Demo</a>
- [2024/08] Release ConfyUI demo. Thanks [smthemex](https://github.com/smthemex) ([ComfyUI_Diffree](https://github.com/smthemex/ComfyUI_Diffree)) for helping!
- [2024/08] Release [training dataset OABench](https://huggingface.co/datasets/LiruiZhao/OABench) in Hugging Face
- [2024/08] Release training code
- [2024/08] Update <a href="https://huggingface.co/spaces/LiruiZhao/Diffree">🤗 Demo</a>, now support iterative generation through a text list

## Contents
- [Install](#install)
- [Inference](#inference)
- [Data Download](#data-download)
- [Training](#training)
- [Citation](#citation)

## Install
1. Clone this repository and navigate to Diffree folder
```
git clone https://github.com/OpenGVLab/Diffree.git

cd Diffree
```

2. Install package
```
conda create -n diffree python=3.8.5

conda activate diffree

pip install -r requirements.txt
```

## Inference

1. Download the Diffree model from Huggingface.
```
pip install huggingface_hub

huggingface-cli download LiruiZhao/Diffree --local-dir ./checkpoints
```

2. You can inference with the script:

```
python app.py
```

Specifically, `--resolution` defines the maximum size for both the resized input image and output image. For our <a href="https://huggingface.co/spaces/LiruiZhao/Diffree">Hugging Face Demo</a>, we set the `--resolution` to `512` to enhance the user experience with higher-resolution results. While during the training process of Diffree, `--resolution` is set to `256`. Therefore, reducing `--resolution` might improve results (e.g., consider trying `320` as a potential value).

## Data Download

You can download the OABench here, which are used for training the Diffree.

1. Download the OABench dataset from Huggingface.

```
huggingface-cli download --repo-type dataset LiruiZhao/OABench --local-dir ./dataset --local-dir-use-symlinks False
```

2. Find and extract all compressed files in the dataset directory

```
cd dataset

ls *.tar.gz | xargs -n1 tar xvf
```

The data structure should be like:

```
|-- dataset
    |-- original_images
        |-- 58134.jpg
        |-- 235791.jpg
        |-- ...
    |-- inpainted_images
        |-- 58134
          |-- 634757.jpg
          |-- 634761.jpg
          |-- ...
        |-- 235791
        |-- ...
    |-- mask_images
        |-- 58134
          |-- 634757.png
          |-- 634761.png
          |-- ...
        |-- 235791
        |-- ...
    |-- annotations.json
```

In the `inpainted_images` and `mask_images` directories, the top-level folders correspond to the original images, and the contents of each folder are the inpainted images and masks for those images.

## Training
Diffree is trained by fine-tuning from an initial StableDiffusion checkpoint. 

1. Download a Stable Diffusion checkpoint and move it to the `checkpoints` directory. For our trained models, we used [the v1.5 checkpoint](https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.ckpt) as the starting point. You can also use the following command:

```
curl -L https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.ckpt -o checkpoints/v1-5-pruned-emaonly.ckpt
```


2. Next, you can start training.

```
python main.py --name diffree --base config/train.yaml --train --gpus 0,1,2,3
```

All configurations are stored in the YAML file. If you need to use custom configuration settings, you can modify the `--base` to point to your custom config file.


## Citation
If you found this work useful, please consider citing:

```
@article{zhao2024diffree,
  title={Diffree: Text-Guided Shape Free Object Inpainting with Diffusion Model},
  author={Zhao, Lirui and Yang, Tianshuo and Shao, Wenqi and Zhang, Yuxin and Qiao, Yu and Luo, Ping and Zhang, Kaipeng and Ji, Rongrong},
  journal={arXiv preprint arXiv:2407.16982},
  year={2024}
}
```
