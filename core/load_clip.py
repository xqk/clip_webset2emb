"""load clip"""
from functools import lru_cache
from torch import autocast, nn
import torch
import clip
from PIL import Image
import time

LOADED_MODEL_DICT = {}


def get_loaded_mode(clip_model):
    """"""
    model_preprocess = LOADED_MODEL_DICT.get(clip_model)
    if not model_preprocess:
        return None, None
    model, preprocess = model_preprocess
    return model, preprocess


def set_loaded_mode(clip_model, model, preprocess):
    """"""
    LOADED_MODEL_DICT[clip_model] = (model, preprocess)


class OpenClipWrapper(nn.Module):
    """
    Wrap OpenClip for managing input types
    """

    def __init__(self, inner_model, device):
        super().__init__()
        self.inner_model = inner_model
        self.device = torch.device(device=device)
        if self.device.type == "cpu":
            self.dtype = torch.float32
        else:
            self.dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    def encode_image(self, image):
        if self.device.type == "cpu":
            return self.inner_model.encode_image(image)
        with autocast(device_type=self.device.type, dtype=self.dtype):
            return self.inner_model.encode_image(image)

    def encode_text(self, text):
        if self.device.type == "cpu":
            return self.inner_model.encode_text(text)
        with autocast(device_type=self.device.type, dtype=self.dtype):
            return self.inner_model.encode_text(text)

    def forward(self, *args, **kwargs):
        return self.inner_model(*args, **kwargs)


def load_open_clip(clip_model, use_jit=True, device="cuda", clip_cache_path=None):
    """load open clip"""
    o_clip_model = f"{clip_model}"
    model, preprocess = get_loaded_mode(clip_model)
    if model:
        return model, preprocess

    import open_clip  # pylint: disable=import-outside-toplevel

    torch.backends.cuda.matmul.allow_tf32 = True

    pretrained = dict(open_clip.list_pretrained())
    checkpoint = pretrained[clip_model]
    model, _, preprocess = open_clip.create_model_and_transforms(
        clip_model, pretrained=checkpoint, device=device, jit=use_jit, cache_dir=clip_cache_path
    )
    model = OpenClipWrapper(inner_model=model, device=device)
    model.to(device=device)

    set_loaded_mode(o_clip_model, model, preprocess)
    return model, preprocess


def load_cn_clip(clip_model, device="cuda", download_root="./"):
    """"""
    o_clip_model = f"{clip_model}"
    model, preprocess = get_loaded_mode(clip_model)
    if model:
        return model, preprocess

    from cn_clip.clip import load_from_name
    # Available models: ['ViT-B-16', 'ViT-L-14', 'ViT-L-14-336', 'ViT-H-14', 'RN50']

    clip_model = clip_model[len("cn_clip:"):]
    model, preprocess = load_from_name(clip_model, device=device, download_root=download_root)
    model.eval()

    set_loaded_mode(o_clip_model, model, preprocess)
    return model, preprocess


@lru_cache(maxsize=None)
def get_tokenizer(clip_model):
    """Load clip"""
    if clip_model.startswith("open_clip:"):
        import open_clip  # pylint: disable=import-outside-toplevel

        clip_model = clip_model[len("open_clip:"):]
        return open_clip.get_tokenizer(clip_model)

    elif clip_model.startswith("cn_clip:"):
        import cn_clip.clip

        return lambda text: cn_clip.clip.tokenize(text)
    else:
        return lambda t: clip.tokenize(t, truncate=True)


@lru_cache(maxsize=None)
def load_clip_without_warmup(clip_model, use_jit, device, clip_cache_path):
    """Load clip"""
    o_clip_model = f"{clip_model}"
    model, preprocess = get_loaded_mode(clip_model)
    if model:
        return model, preprocess

    if clip_model.startswith("open_clip:"):
        clip_model = clip_model[len("open_clip:"):]
        model, preprocess = load_open_clip(clip_model, use_jit, device, clip_cache_path)
    elif clip_model.startswith("cn_clip:"):
        from cn_clip.clip import load_from_name
        clip_model = clip_model[len("cn_clip:"):]
        model, preprocess = load_from_name(clip_model, device=device, download_root=clip_cache_path)
        model.eval()
    else:
        model, preprocess = clip.load(clip_model, device=device, jit=use_jit, download_root=clip_cache_path)

    set_loaded_mode(o_clip_model, model, preprocess)
    return model, preprocess


@lru_cache(maxsize=None)
def load_clip(clip_model="ViT-B/32", use_jit=True, warmup_batch_size=1, clip_cache_path=None, device=None):
    """Load clip then warmup"""
    o_clip_model = f"{clip_model}"
    model, preprocess = get_loaded_mode(clip_model)
    if model:
        return model, preprocess

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = load_clip_without_warmup(clip_model, use_jit, device, clip_cache_path)

    start = time.time()
    print(f"warming up with batch size {warmup_batch_size} on {device}", flush=True)
    warmup(warmup_batch_size, device, preprocess, model)
    duration = time.time() - start
    print(f"done warming up in {duration}s", flush=True)

    set_loaded_mode(o_clip_model, model, preprocess)
    return model, preprocess


def warmup(batch_size, device, preprocess, model):
    fake_img = Image.new("RGB", (224, 224), color="red")
    fake_text = ["fake"] * batch_size
    image_tensor = torch.cat([torch.unsqueeze(preprocess(fake_img), 0)] * batch_size).to(device)
    text_tokens = clip.tokenize(fake_text).to(device)
    for _ in range(2):
        with torch.no_grad():
            model.encode_image(image_tensor)
            model.encode_text(text_tokens)
