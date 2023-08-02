import os

import pandas as pd
from img2dataset import download
from core.inference.inference import main_to_inference


class Dataset:
    """数据集"""

    def __init__(self,
                 dataset_name,
                 dataset_dir,
                 clip_model,
                 clip_cache_path,
                 image_size: int = 336,
                 output_format: str = "webdataset",
                 ):
        """"""
        self.clip_model = clip_model
        self.dataset_name = dataset_name
        self.dataset_dir = dataset_dir

        self.urls_folder = f"{dataset_dir}/urls"
        self.urls_parquet_path = f"{self.urls_folder}/{dataset_name}.parquet"
        self.images_folder = f"{dataset_dir}/images/{dataset_name}"
        self.embeddings_folder = f"{dataset_dir}/embeddings/{dataset_name}"

        self.clip_cache_path = clip_cache_path

        self.image_size = image_size
        self.output_format = output_format

        for d in [self.dataset_dir, self.images_folder, self.embeddings_folder, self.urls_folder]:
            if not os.path.exists(d):
                os.makedirs(d)

    def urls2parquet(self, urls):
        """"""
        pd.DataFrame(urls).to_parquet(self.urls_parquet_path)

    def parquet2webdataset(self):
        """parquet转webdataset"""
        download(
            self.urls_parquet_path,
            image_size=self.image_size,
            output_folder=self.images_folder,
            thread_count=256,
            input_format="parquet",
            output_format=self.output_format,
            url_col="url",
            caption_col="caption",
        )

    def webdataset2inference(self):
        """webdataset转向量"""
        import fsspec

        fs, _ = fsspec.core.url_to_fs(self.dataset_dir)
        input_files = [self.images_folder + "/" + p for p in next(fs.walk(self.images_folder))[2] if p.endswith(".tar")]
        main_to_inference(
            input_dataset=input_files,
            output_folder=self.embeddings_folder,
            input_format="webdataset",
            enable_metadata=True,
            write_batch_size=100000,
            batch_size=256,
            cache_path=None,
            num_prepro_workers=1,
            clip_model=self.clip_model,
            clip_cache_path=self.clip_cache_path,
            enable_text=True,
        )
        
    def webdataset2inference_by_files(self, input_files, embeddings_folder="embeddings", batch_size=256):
        """webdataset转向量"""
        main_to_inference(
            input_dataset=input_files,
            output_folder=embeddings_folder,
            input_format="webdataset",
            enable_metadata=True,
            write_batch_size=100000,
            batch_size=batch_size,
            cache_path=None,
            num_prepro_workers=1,
            clip_model=self.clip_model,
            clip_cache_path=self.clip_cache_path,
            enable_text=True,
        )