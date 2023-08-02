import io
import os

import requests
import torch
from PIL import Image
from torchvision import transforms

from lib.convnext import ConvNeXt


class WatermarksPredictor:
    """水印预测"""

    def __init__(self, clip_cache_path="./"):
        """"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.wm_model, self.process = self.watermark_model_init(clip_cache_path=clip_cache_path)
        if self.wm_model is not None:
            self.wm_model.eval()

    def watermark_model_init(self, clip_cache_path="./"):
        """初始化水印模型"""
        weights_path = os.path.join(clip_cache_path, "convnext_tiny.pt")
        if not os.path.exists(weights_path):
            return None, None

        model_ft = ConvNeXt(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768])
        model_ft.head = torch.nn.Sequential(
            torch.nn.Linear(in_features=768, out_features=512),
            torch.nn.GELU(),
            torch.nn.Linear(in_features=512, out_features=256),
            torch.nn.GELU(),
            torch.nn.Linear(in_features=256, out_features=2),
        )

        detector_transforms = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        weights = torch.load(weights_path, self.device)
        model_ft.load_state_dict(weights)
        model_ft.eval()
        model_ft = model_ft.to(self.device)

        del weights, weights_path

        return model_ft, detector_transforms

    def get_image(self, img):
        """ 获取图片 """

        pil_img = None

        if isinstance(img, str):
            # url图
            if img.startswith('http://') or img.startswith('https://'):
                try:
                    img_byte = requests.get(img).content
                    pil_img = Image.open(io.BytesIO(img_byte)).convert('RGB')
                    del img_byte
                except Exception as error:
                    print("error: ", error)

            # 本地图
            else:
                if os.path.isfile(img) and os.path.exists(img):
                    pil_img = Image.open(img).convert('RGB')
                    del img

        # PIL 图
        elif isinstance(img, Image.Image):
            pil_img = img
            del img

        # 字节流
        elif isinstance(img, bytes):
            pil_img = Image.open(io.BytesIO(img)).convert('RGB')

        return pil_img

    def predict(self, img_input):
        pil_image = self.get_image(img_input)
        input_img = self.process(pil_image).float().unsqueeze(0)
        with torch.no_grad():
            outputs = self.wm_model(input_img.to(self.device))
            result = torch.max(outputs, 1)[1].cpu().reshape(-1).tolist()[0]
            # print("result: ", result)
            del input_img, pil_image, img_input, outputs

        return 'watermark' if result else 'clear'

    def infer(self, img_input):
        """"""
        if self.wm_model is None:
            return None
        img = self.get_image(img_input)

        input_img = self.process(img).float().unsqueeze(0)
        with torch.no_grad():
            outputs = self.wm_model(input_img.to(self.device))
            result = torch.max(outputs, 1)[1].cpu().reshape(-1).tolist()[0]
            # print("result: ", result)
            res_wm = True if result else False
            del input_img, img, img_input, outputs

        return res_wm


if __name__ == "__main__":
    """"""
    from settings.config import CLIP_CACHE_PATH
    wm = WatermarksPredictor(clip_cache_path=CLIP_CACHE_PATH)

    # url = "https://www.feimosheji.com/uploads/ueditor/image/20221217/1671257538315857.jpg"
    url = "https://media.zhuke.com/FvS-g7iKIJKUDG-mtlBOIyCa2RJq~736x.jpg"
    res = wm.infer(url)
    print("res: ", res)
