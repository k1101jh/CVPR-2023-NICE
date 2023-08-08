FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime

RUN mkdir NICE_image_captioning
ADD codes NICE_image_captioning/codes
ADD configs NICE_image_captioning/configs
ADD datasets NICE_image_captioning/datasets

