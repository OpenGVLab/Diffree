# Diffree
Official PyTorch implement of paper "Diffree: Text-Guided Shape Free Object Inpainting with Diffusion Model"

<p align="center">
  <a href="https://arxiv.org/pdf/2407.16982"><u>[📜 Arxiv]</u></a>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
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

## Contents
- [Install](#install)
- [Inference](#inference)
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

## Training

1. Diffree is fine-tuned from an initial Stable Diffusion v1.5 checkpoint. The process begins by downloading this checkpoint.
```
curl -L https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.ckpt -o ./stable_diffusion/models/ldm/stable-diffusion-v1/v1-5-pruned-emaonly.ckpt
```

2. You can inference with the script:

```
python app.py
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
