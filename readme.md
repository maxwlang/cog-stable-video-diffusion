# cog-deep-danbooru
A Docker container generated using [Cog](https://github.com/replicate/cog). Cog is TOTALLY AWESOME and makes packaging ML models into API driven Docker containers insanely easy. Check it out!


[Deep Danbooru](https://github.com/RF5/danbooru-pretrained) is a model trained on images and tags from the anime image hosting website Danbooru. This model takes an input image and then returns a dictionary of descriptor tags relating to the image, similar to how Danbooru categorizes images. This model is extremely useful when it comes to generating images with the Stable Diffusion machine learning model, as it helps provide useful image descriptors for image generation.

# Usage
1. Obtain a token from hugging face for model download
2. Build with `cog build`
3. Run `docker run -d --gpus all -p "5000:5000" -e TOKEN=hf_token_here built_image_name_here:latest`
4. Access API documentations at `GET http://host:5000/`
5. POST an image and receive tags back

# Citations
```
@misc{danbooru2018resnet,
    author = {Matthew Baas},
    title = {Danbooru2018 pretrained resnet models for PyTorch},
    howpublished = {\url{https://rf5.github.io}},
    url = {https://rf5.github.io/2019/07/08/danbuuro-pretrained.html},
    type = {pretrained model},
    year = {2019},
    month = {July},
    timestamp = {2019-07-08},
    note = {Accessed: DATE}
}
```