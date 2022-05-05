# arbitraty_style_transfer_modification
Neural style transfer pipeline for arbitrary stylization with strength control and color transfer control.

Paper on Russian (diploma work): ["Modification of automatic stylization method via transformer network"](ElistratovS_diploma_work.pdf)

[Presentation](presentation.pdf)

Code for style transfer module based on [this repository](https://github.com/philipjackson/style-augmentation).

Model weights (checkpoint): [link](https://drive.google.com/drive/folders/1tnS9f4O-9j78h_3fGZFW87pUGcJnYrOP?usp=sharing)

Checkpoint has the following keys: "model_state_dict", "optimizer_state_dict", "epoch", "loss". So, to get model's weights, you should use only "model_state_dict".
