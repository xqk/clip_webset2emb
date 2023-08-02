"""
Inference Worker:

A completely independent process that will be started once for each GPU node.
Distributors will call this either through the CLI or directly.

The worker sequentially process the tasks passed to it.
Tasks are lists of partition_id's that this worker will be responsible for.
"""

import torch
from braceexpand import braceexpand

from core.inference.runner import Runner
from core.inference.mapper import ClipMapper
from core.inference.writer import NumpyWriter
from core.inference.logger import LoggerWriter
from core.inference.reader import FilesReader, WebdatasetReader
from core.load_clip import load_clip, load_cn_clip, load_open_clip


def worker(
        tasks,
        input_dataset,
        output_folder,
        output_partition_count,
        input_format="files",
        cache_path=None,
        batch_size=256,
        num_prepro_workers=8,
        enable_text=True,
        enable_image=True,
        enable_metadata=False,
        wds_image_key="jpg",
        wds_caption_key="txt",
        clip_model="ViT-B/32",
        mclip_model="sentence-transformers/clip-ViT-B-32-multilingual-v1",
        use_mclip=False,
        use_jit=True,
        clip_cache_path=None,
):
    """Start a worker"""
    print("Starting the worker", flush=True)

    # check for brace expansion
    if input_format == "webdataset" and not isinstance(input_dataset, list):
        input_dataset = list(braceexpand(input_dataset))

    print(f"dataset is {len(input_dataset)}", flush=True)

    def reader_builder(sampler):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if clip_model.startswith("open_clip:"):
            _, preprocess = load_open_clip(
                clip_model, use_jit=use_jit, device=device, clip_cache_path=clip_cache_path
            )

        elif clip_model.startswith("cn_clip:"):
            _, preprocess = load_cn_clip(
                clip_model=clip_model, device=device, download_root=clip_cache_path
            )
        else:
            _, preprocess = load_clip(
                clip_model=clip_model, use_jit=use_jit, warmup_batch_size=batch_size, clip_cache_path=clip_cache_path
            )

        if input_format == "files":
            return FilesReader(
                sampler,
                preprocess,
                input_dataset,
                batch_size,
                num_prepro_workers,
                enable_text=enable_text,
                enable_image=enable_image,
                enable_metadata=enable_metadata,
            )
        elif input_format == "webdataset":
            return WebdatasetReader(
                sampler,
                preprocess,
                input_dataset,
                batch_size,
                num_prepro_workers,
                enable_text=enable_text,
                enable_image=enable_image,
                enable_metadata=enable_metadata,
                wds_image_key=wds_image_key,
                wds_caption_key=wds_caption_key,
                cache_path=cache_path,
                clip_model=clip_model,
            )
        else:
            raise ValueError(f"Unknown input_format: {input_format}")

    def mapper_builder():
        return ClipMapper(
            enable_image=enable_image,
            enable_text=enable_text,
            enable_metadata=enable_metadata,
            use_mclip=use_mclip,
            clip_model=clip_model,
            use_jit=use_jit,
            mclip_model=mclip_model,
            clip_cache_path=clip_cache_path,
            warmup_batch_size=batch_size,
        )

    def writer_builder(i):
        return NumpyWriter(
            partition_id=i,
            output_folder=output_folder,
            enable_text=enable_text,
            enable_image=enable_image,
            enable_metadata=enable_metadata,
            output_partition_count=output_partition_count,
        )

    def logger_builder(i):
        return LoggerWriter(
            partition_id=i,
            stats_folder=output_folder + "/stats",
        )

    runner = Runner(
        reader_builder=reader_builder,
        mapper_builder=mapper_builder,
        writer_builder=writer_builder,
        logger_builder=logger_builder,
        output_partition_count=output_partition_count,
    )

    for task in tasks:
        print(f"Starting work on task {task}", flush=True)
        runner(task)
