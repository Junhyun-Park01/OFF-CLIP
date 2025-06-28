from . import text_model
from . import bert_model
from . import vision_model
from . import dqn_wo_self_atten_mlp
from . import cnn_backbones
from . import fusion_module
from . import offclip_model

IMAGE_MODELS = {
    "offclip": vision_model.ImageEncoder
}
