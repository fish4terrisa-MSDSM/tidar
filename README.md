# TiDAR: Think in Diffusion, Talk in Autoregression (Unofficial Implemention)

This is an unofficial implemention of TiDAR, support most LLM models that have transformers support(in theory)
Support both SFT/DPO training and inference(tidar mode, block diffusion only mode and ar only mode are all supported)
It's kinda hacky so may break with minor changes.

(Custom DPO trainer isnt really tested and I'm not sure if it'll really work as expected)

## Installation
Install this package with
```bash
pip install . --no-build-isolation
```
Or if you want to install it as a system package
```bash
sudo pip install . --break-system-packages --no-build-isolation
```
`--no-build-isolation` is here so you wont need to install another torch and other dependencies in the isolated environment just to build and install this package.

## Paper (not mine nor related)
[TiDAR: Think in Diffusion, Talk in Autoregression](https://arxiv.org/pdf/2511.08923)
