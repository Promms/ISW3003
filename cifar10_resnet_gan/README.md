# CIFAR-10 ResNet GAN

Unconditional 32×32 image generation on CIFAR-10 with a ResNet-style GAN: ResBlock generator with InstanceNorm, ResBlock discriminator with SpectralNorm.

## Architecture

| | Generator | Discriminator |
|---|---|---|
| Style | Pre-activation ResBlocks | Pre-activation ResBlocks |
| Activation norm | InstanceNorm (affine) | none |
| Weight norm | none | SpectralNorm on every conv/linear |
| Up/down sampling | nearest-neighbor 2× upsample | 2×2 average pool |
| Channels | 256 → 256 → 256 → 128 → 3 | 3 → 128 → 128 → 256 → 256 |
| Output | Tanh, image in [-1, 1] | scalar logit |
| Params | ~3.5M | ~2.6M |

Loss is the non-saturating logistic loss:
- `L_D = softplus(D(fake)) + softplus(-D(real))`
- `L_G = softplus(-D(fake))`

## Notes
- `n_critic=3` gives a stable D under SpectralNorm. With more capacity or harder data you might bump it to 5; with an even stronger D regularizer you can drop to 1.
- Adam betas `(0.0, 0.9)` follow SNGAN/SAGAN/StyleGAN. `(0.5, 0.999)` (DCGAN convention) also works.
