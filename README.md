# Show_Attend_Tell

Show Attend Tell paper implementation. This repo is heavily based on [image-captioning-project](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning).
This repository is based on pytorch-lightning.

Differences from original repo

- [x] Different dataset setting (cap_per_img=5, min_word_freq=5) 
- [x] I did not develop inference module yet.
- [ ] Use EfficientNet V2 Small as Encoder (original repo resnet101)
- [ ] Different dropout rate (original repo 0.5)
- [ ] 1 stage learning (original repo use freezing stages)
- [ ] Different training setup (batch_size, init_lr, optimizer, learning_rate_scheduler)