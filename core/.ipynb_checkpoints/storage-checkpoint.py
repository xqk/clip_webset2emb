import json
import shutil

import fsspec
import os.path
from pprint import pprint

import pandas as pd
import numpy as np
import torch

from pymilvus import Collection, CollectionSchema, FieldSchema, DataType, utility
from torch import Tensor
from tqdm import tqdm

from core.dataset import Dataset
from core.watermark import WatermarksPredictor
from lib.file import get_file_url_by_file_json
from lib.h14_nsfw_model import H14_NSFW_Detector
from lib.txt import num_get, sha1
from lib.utils import normalized
from model import dh_project, dh_user, dh_vector


class Storage:
    """存储数据到milvus"""

    def __init__(self, clip_cache_path="./"):
        """"""
        self.connections = {}

        self.project_dataset_name = "zhuke_pro_project_photo"
        self.fav_dataset_name = "zhuke_pro_fav"
        min_edge = 300  # 图片过滤：太扁/太宽
        max_aspect_ratio = 3  # 图片过滤：异常长宽比
        self.min_photo_edge = min_edge
        self.max_photo_aspect_ratio = max_aspect_ratio
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.safety_model = H14_NSFW_Detector(cache_folder=clip_cache_path, device=self.device)
        self.wm = WatermarksPredictor(clip_cache_path=clip_cache_path)

    def get_connection(self, collection_name):
        """"""
        collection = self.connections.get(collection_name)
        if not collection:
            collection = Collection(name=collection_name)
            self.connections[collection_name] = collection

        return collection

    def create_project_schema(self):
        """"""
        # utility.drop_collection(self.project_dataset_name)
        if not utility.has_collection(self.project_dataset_name):
            schema = CollectionSchema(
                fields=[
                    FieldSchema(
                        name="id",
                        dtype=DataType.VARCHAR,
                        is_primary=True,
                        max_length=40,
                    ),
                    FieldSchema(
                        name="media_id",
                        dtype=DataType.INT64,
                    ),
                    FieldSchema(
                        name="parent_table_id",
                        dtype=DataType.VARCHAR,
                        max_length=30,
                    ),
                    FieldSchema(
                        name="vector",
                        dtype=DataType.FLOAT_VECTOR,
                        dim=1024
                    )
                ],
                description="案例图片"
            )
            collection = Collection(
                name=self.project_dataset_name,
                schema=schema,
                using='default',
                shards_num=2,
            )
            index = {
                "index_type": "IVF_FLAT",
                "metric_type": "IP",
                "params": {"nlist": 1024},
            }

            status = collection.create_index(field_name="vector", index_params=index)
            print("create_index:", True if status.code == 0 else False)
            collection.load()

    def create_fav_schema(self):
        """"""
        # utility.drop_collection(self.fav_dataset_name)
        if not utility.has_collection(self.fav_dataset_name):
            schema = CollectionSchema(
                fields=[
                    FieldSchema(
                        name="id",
                        dtype=DataType.VARCHAR,
                        is_primary=True,
                        max_length=40,
                    ),
                    FieldSchema(
                        name="table_id",
                        dtype=DataType.VARCHAR,
                        max_length=104,
                    ),
                    FieldSchema(
                        name="parent_table_id",
                        dtype=DataType.VARCHAR,
                        max_length=104,
                    ),
                    FieldSchema(
                        name="user_id",
                        dtype=DataType.INT64,
                    ),
                    FieldSchema(
                        name="vector",
                        dtype=DataType.FLOAT_VECTOR,
                        dim=1024
                    )
                ],
                description="收藏图片"
            )
            collection = Collection(
                name=self.fav_dataset_name,
                schema=schema,
                using='default',
                shards_num=2,
            )
            index = {
                "index_type": "IVF_FLAT",
                "metric_type": "IP",
                "params": {"nlist": 1024},
            }

            status = collection.create_index(field_name="vector", index_params=index)
            print("create_index:", True if status.code == 0 else False)
            collection.load()

    async def project2milvus(self, project_ids=None, dataset_dir=None, clip_model="cn_clip:ViT-H-14", clip_cache_path="./"):
        """案例转milvus
        过滤：异常长宽比、太扁/太高
        结构
            {
                "id": "",  # sha1(key)
                "media_id": 1,
                "parent_table_id": "project:1",
                "vector": [0] * 1024
            }
        """
        dataset_name = self.project_dataset_name

        if not project_ids:
            project_ids = []
        if not dataset_dir:
            dataset_dir = "dataset"

        embeddings_folder = os.path.join(dataset_dir, f"embeddings/{dataset_name}")
        images_folder = os.path.join(dataset_dir, f"images/{dataset_name}")
        urls_parquet_path = os.path.join(dataset_dir, f"urls/{dataset_name}.parquet")
        await self._remove_dataset_file(embeddings_folder, images_folder, urls_parquet_path)

        # 创建数据集
        dataset = Dataset(dataset_name, dataset_dir, clip_model, clip_cache_path)

        collection = self.get_connection(dataset_name)

        # 案例编号转图片列表
        media_ids_list = await dh_project.ProjectMedias.filter(id__in=project_ids).values_list("media_ids", flat=True)
        media_ids = []
        for m in media_ids_list:
            m = m.strip("|")
            if not m:
                continue

            media_ids.extend(list(map(int, list(filter(bool, m.split('|'))))))
        if not media_ids:
            return

        photos = await dh_project.Media.filter(id__in=media_ids, kind="pic")
        
        # 过滤已存在的图片sha1(key)
        sha1keys = []
        for photo in photos:
            f = photo.file
            if not f:
                continue

            try:
                file = json.loads(f)
            except Exception as e:
                continue
            if not file:
                continue

            file_key = file.get("key", "")
            if not file_key:
                continue
            sha1keys.append(sha1(file_key))
        if len(sha1keys) == 0:
            return

        exists_res = collection.query(
            expr=f'id in {json.dumps(sha1keys)}',
            output_fields=[],
            offset=0,
            limit=10000,
            consistency_level="Strong"
        )
        exists_ids = set([x.get("id", "") for x in exists_res])

        # 图片转webdataset
        url_caption_list, extracted_list = await self._project_media2urls(exists_ids, photos)
        if len(url_caption_list) == 0 and len(extracted_list) == 0:
            return

        ids = []
        media_ids = []
        parent_table_ids = []
        vectors = []
        if len(url_caption_list) > 0:
            dataset.urls2parquet(url_caption_list)
            dataset.parquet2webdataset()

            # webdataset转向量
            dataset.webdataset2inference()

            # 向量写入milvus
            fs, _ = fsspec.core.url_to_fs(embeddings_folder)
            emb_folder = os.path.join(embeddings_folder, "img_emb")
            metadata_folder = os.path.join(embeddings_folder, "metadata")
            if os.path.exists(emb_folder) and os.path.exists(metadata_folder):
                metadata_files = [metadata_folder + "/" + p for p in next(fs.walk(metadata_folder))[2] if p.endswith(".parquet")]
                for i in range(len(metadata_files)):
                    metadata_file = os.path.join(metadata_folder, f"metadata_{i}.parquet")
                    emb_file = os.path.join(emb_folder, f"img_emb_{i}.npy")
                    df = pd.read_parquet(metadata_file)
                    embs = np.load(emb_file)
                    df["emb"] = embs.tolist()

                    for d in df.itertuples():
                        if not d.caption:
                            continue

                        caption_info = json.loads(d.caption)
                        ids.append(caption_info.get("id", ""))
                        media_ids.append(caption_info.get("media_id", 0))
                        parent_table_ids.append(caption_info.get("parent_table_id", ""))
                        vectors.append(d.emb)
                        file_key = caption_info.get("key", "")
                        if file_key and len(d.emb) > 0:
                            await dh_vector.Media1024.get_or_create(key=f"dhome-media:{file_key}", defaults={
                                "vector": d.emb,
                            })

        if len(extracted_list) > 0:
            for m in extracted_list:
                ids.append(m.get("id", ""))
                media_ids.append(m.get("media_id", 0))
                parent_table_ids.append(m.get("parent_table_id", ""))
                vectors.append(m.get("vector", []))

        # 过滤不安全、重复的数据
        remove_index_sets = await self.filter_embeddings_remove_index_sets(
            self.project_dataset_name, vectors,
            safety_model=self.safety_model,
            violence_detector=None, deduplicate_expr=None,
            deduplicate_anns_field="vector",
        )
        # print("remove_index_sets", remove_index_sets)
        # print("ids", ids)
        need_ids = []
        need_media_ids = []
        need_parent_table_ids = []
        need_vectors = []
        for i, _ in enumerate(ids):
            if i in remove_index_sets:
                continue
            need_ids.append(ids[i])
            need_media_ids.append(media_ids[i])
            need_parent_table_ids.append(parent_table_ids[i])
            need_vectors.append(vectors[i])
        ids = need_ids
        media_ids = need_media_ids
        parent_table_ids = need_parent_table_ids
        vectors = need_vectors

        # print("ids", ids)

        if len(ids) == 0:
            return

        # 插入数据
        batch_size = 1000
        for i in range(0, len(ids), batch_size):
            batch_data = [
                ids[i:i + batch_size],
                media_ids[i:i + batch_size],
                parent_table_ids[i:i + batch_size],
                vectors[i:i + batch_size],
            ]
            collection.insert(batch_data)
        collection.flush()

        # print("collection.num_entities:", collection.num_entities)

        # 删除webdataset、向量
        await self._remove_dataset_file(embeddings_folder, images_folder, urls_parquet_path)

    async def remove_project_photo_by_project_ids(self, project_ids=None):
        """"""
        if not project_ids:
            return

        collection = self.get_connection(self.project_dataset_name)

        for project_id in tqdm(project_ids):
            files = await dh_project.Media.filter(table_id=f"project:{project_id}", kind__in=("pic", "cover",)).values_list("file", flat=True)
            if len(files) == 0:
                continue
            sha1keys = []
            for f in files:
                try:
                    file = json.loads(f)
                except Exception as e:
                    continue
                if not file:
                    continue

                file_key = file.get("key", "")
                if not file_key:
                    continue
                sha1keys.append(sha1(file_key))
            if len(sha1keys) == 0:
                continue
            expr = f"id in {json.dumps(sha1keys)}"
            collection.delete(expr)

        collection.flush()

    async def remove_project_watermark_photo_by_project_ids(self, project_ids=None):
        """根据一组案例编号删除其中的水印图片"""
        if not project_ids:
            return

        collection = self.get_connection(self.project_dataset_name)

        for project_id in tqdm(project_ids):
            media_ids_list = await dh_project.ProjectMedias.filter(id=project_id).values_list("media_ids", flat=True)
            media_ids = []
            for m in media_ids_list:
                m = m.strip("|")
                if not m:
                    continue

                media_ids.extend(list(map(int, list(filter(bool, m.split('|'))))))
            if not media_ids:
                return

            files = await dh_project.Media.filter(id__in=media_ids, kind="pic").values_list("file", flat=True)
            if len(files) == 0:
                continue

            delete_ids = []
            for f in files:
                try:
                    file = json.loads(f)
                except Exception as e:
                    continue
                if not file:
                    continue

                file_key = file.get("key", "")
                if not file_key:
                    continue

                bucket = file.get("bucket", "dhome-media") or "dhome-media"
                o = await dh_project.FilePortrait.get_or_none(key=f"{bucket}:{file_key}")
                if o and o.with_watermark != 0:
                    if o.with_watermark == 1:
                        delete_ids.append(sha1(file_key))
                    continue

                # 判断是否有水印
                url = get_file_url_by_file_json(file)
                with_watermark = self.wm.infer(url)
                with_watermark_v = 0
                if with_watermark is not None:
                    with_watermark_v = 1 if with_watermark else 2
                    o, is_new = await dh_project.FilePortrait.get_or_create(key=f"{bucket}:{file_key}", defaults={
                        "with_watermark": with_watermark_v
                    })
                    if not is_new:
                        if o.with_watermark == 0 and with_watermark_v != 0:
                            o.with_watermark = with_watermark_v
                            await o.save()

                    if with_watermark_v == 1:
                        delete_ids.append(sha1(file_key))

            delete_ids = list(set(delete_ids))
            if len(delete_ids) == 0:
                continue
            expr = f"id in {json.dumps(delete_ids)}"
            collection.delete(expr)

        collection.flush()

    async def _project_media2urls(self, exists_ids, photos):
        extracted_list = []
        url_caption_list = []
        for photo in photos:
            if photo.id in exists_ids:
                continue

            f = photo.file
            if not f:
                continue

            try:
                file = json.loads(f)
            except Exception as e:
                continue
            if not file:
                continue

            file_key = file.get("key", "")
            if not file_key:
                continue
            
            # 图像大小过滤
            file_width = num_get(file.get("width", 0))
            file_height = num_get(file.get("height", 0))
            if min([file_width, file_height]) < self.min_photo_edge or \
                max([file_width, file_height])/min([file_width, file_height]) > self.max_photo_aspect_ratio:
                # print("_project_media2urls 图像大小过滤: ", [file_width, file_height])
                continue

            # 检查是否已经提取到数据库中了
            bucket = file.get("bucket", "dhome-media") or "dhome-media"
            o = await dh_vector.Media1024.get_or_none(key=f"{bucket}:{file_key}")
            if o:
                extracted_list.append({
                    "id": sha1(file_key),
                    "media_id": photo.id,
                    "parent_table_id": photo.table_id,
                    "key": file_key,
                    "vector": o.vector,
                })
                continue
            
            url = get_file_url_by_file_json(file)
            caption = json.dumps({
                "id": sha1(file_key),
                "media_id": photo.id,
                "parent_table_id": photo.table_id,
                "key": file_key,
            })
            url_caption_list.append({
                "url": url,
                "caption": caption
            })
        return url_caption_list, extracted_list

    async def _fav_media2urls(self, exists_ids, user_id, photos, is_project=False):
        extracted_list = []
        url_caption_list = []
        for photo in photos:
            if is_project:
                eid = sha1(f"{user_id}-{photo.table_id}-media:{photo.id}")
            else:
                eid = sha1(f"{user_id}-media:{photo.id}-media:{photo.id}")
            if eid in exists_ids:
                continue

            f = photo.file
            if not f:
                continue

            try:
                file = json.loads(f)
            except Exception as e:
                continue
            if not file:
                continue

            file_key = file.get("key", "")
            if not file_key:
                continue
            
            # 图像大小过滤
            if is_project:
                file_width = num_get(file.get("width", 0))
                file_height = num_get(file.get("height", 0))
                if min([file_width, file_height]) < self.min_photo_edge or \
                    max([file_width, file_height])/min([file_width, file_height]) > self.max_photo_aspect_ratio:
                    # print("_fav_media2urls 图像大小过滤: ", [file_width, file_height])
                    continue

            # 检查是否已经提取到数据库中了
            bucket = file.get("bucket", "dhome-media") or "dhome-media"
            o = await dh_vector.Media1024.get_or_none(key=f"{bucket}:{file_key}")
            if o:
                extracted_list.append({
                    "id": eid,
                    "table_id": f'media:{photo.id}',
                    "parent_table_id": photo.table_id if is_project else f'media:{photo.id}',
                    "user_id": user_id,
                    "key": file_key,
                    "vector": o.vector,
                })
                continue

            url = get_file_url_by_file_json(file)
            caption = json.dumps({
                "id": eid,
                "table_id": f'media:{photo.id}',
                "parent_table_id": photo.table_id if is_project else f'media:{photo.id}',
                "user_id": user_id,
                "key": file_key,
            })
            url_caption_list.append({
                "url": url,
                "caption": caption
            })
            
        return url_caption_list, extracted_list

    async def _fav_pin2urls(self, exists_ids, user_id, pins):
        """"""
        extracted_list = []
        url_caption_list = []
        for pin in pins:
            eid = sha1(f"{user_id}-pin:{pin.id}-pin:{pin.id}")
            if eid in exists_ids:
                continue

            f = pin.file
            if not f:
                continue

            try:
                file = json.loads(f)
            except Exception as e:
                continue
            if not file:
                continue

            file_key = file.get("key", "")
            if not file_key:
                continue

            # 检查是否已经提取到数据库中了
            bucket = file.get("bucket", "dhome-media") or "dhome-media"
            o = await dh_vector.Media1024.get_or_none(key=f"{bucket}:{file_key}")
            if o:
                extracted_list.append({
                    "id": eid,
                    "table_id": f'pin:{pin.id}',
                    "parent_table_id": f'pin:{pin.id}',
                    "user_id": user_id,
                    "key": file_key,
                    "vector": o.vector,
                })
                continue

            url = get_file_url_by_file_json(file)
            caption = json.dumps({
                "id": eid,
                "table_id": f'pin:{pin.id}',
                "parent_table_id": f'pin:{pin.id}',
                "user_id": user_id
            })
            url_caption_list.append({
                "url": url,
                "caption": caption
            })

        return url_caption_list, extracted_list

    async def _user_fav2milvus(self, user_id=0, fav_item_ids=None, dataset_dir=None, clip_model="cn_clip:ViT-H-14", clip_cache_path="./"):
        """收藏转milvus
        注意：fav_item_ids必须是user_id名下的
        结构
            {
                "id": "xxxxxx",
                "table_id": "media:1",
                "parent_table_id": "project:1",
                "user_id": 1,
                "vector": [0] * 1024
            }
        """
        dataset_name = self.fav_dataset_name

        if not fav_item_ids:
            fav_item_ids = []
        if not dataset_dir:
            dataset_dir = "dataset"

        if not fav_item_ids or not user_id:
            return

        embeddings_folder = os.path.join(dataset_dir, f"embeddings/{dataset_name}")
        images_folder = os.path.join(dataset_dir, f"images/{dataset_name}")
        urls_parquet_path = os.path.join(dataset_dir, f"urls/{dataset_name}.parquet")
        await self._remove_dataset_file(embeddings_folder, images_folder, urls_parquet_path)

        # 创建数据集
        dataset = Dataset(dataset_name, dataset_dir, clip_model, clip_cache_path)

        collection = self.get_connection(dataset_name)

        # 收藏项编号转图片列表
        fav_items = await dh_user.FavItem.filter(id__in=fav_item_ids)
        pin_ids = set()
        media_ids = set()
        project_ids = set()
        ids = []
        for fav_item in fav_items:
            if fav_item.table_id.startswith("pin:"):
                _, pid = fav_item.table_id.split(":", 1)
                if pid.isdigit():
                    pin_ids.add(pid)
                    ids.append(sha1(f"{user_id}-{fav_item.table_id}-{fav_item.table_id}"))
            elif fav_item.table_id.startswith("media:"):
                _, pid = fav_item.table_id.split(":", 1)
                media_ids.add(num_get(pid))
                ids.append(sha1(f"{user_id}-{fav_item.table_id}-{fav_item.table_id}"))
            elif fav_item.table_id.startswith("project:"):
                _, pid = fav_item.table_id.split(":", 1)
                project_ids.add(num_get(pid))

        # 案例图片列表
        project_photos = []
        if len(project_ids) > 0:
            media_ids_list = await dh_project.ProjectMedias.filter(id__in=list(project_ids)).values_list("media_ids", flat=True)
            project_media_ids = []
            for m in media_ids_list:
                m = m.strip("|")
                if not m:
                    continue

                project_media_ids.extend(list(map(int, list(filter(bool, m.split('|'))))))

            # 案例下的图片
            project_photos = await dh_project.Media.filter(id__in=project_media_ids, kind="pic")
            for p in project_photos:
                ids.append(sha1(f"{user_id}-{p.table_id}-media:{p.id}"))

        # 单图图片列表
        media_photos = []
        if len(media_ids) > 0:
            media_photos = await dh_project.Media.filter(id__in=media_ids, kind="pic")

        # 拼趣图片列表
        pins = []
        if len(pin_ids) > 0:
            pins = await dh_project.Pin.filter(id__in=list(pin_ids))

        if not ids:
            return

        # 已存在的编号
        exists_res = collection.query(
            expr=f'id in {json.dumps(ids[:10000])}',
            output_fields=[],
            offset=0,
            limit=10000,
            consistency_level="Strong"
        )
        exists_ids = set([x.get("id", 0) for x in exists_res])

        url_caption_list = []
        extracted_list = []
        url_caption_list_1, extracted_list_1 = await self._fav_media2urls(exists_ids, user_id, project_photos, True)
        url_caption_list_2, extracted_list_2 = await self._fav_media2urls(exists_ids, user_id, media_photos, False)
        url_caption_list_3, extracted_list_3 = await self._fav_pin2urls(exists_ids, user_id, pins)
        url_caption_list.extend(url_caption_list_1)
        extracted_list.extend(extracted_list_1)
        url_caption_list.extend(url_caption_list_2)
        extracted_list.extend(extracted_list_2)
        url_caption_list.extend(url_caption_list_3)
        extracted_list.extend(extracted_list_3)
        if len(url_caption_list) == 0 and len(extracted_list) == 0:
            return

        ids = []
        table_ids = []
        parent_table_ids = []
        user_ids = []
        vectors = []
        if len(url_caption_list) > 0:
            # 图片转webdataset
            dataset.urls2parquet(url_caption_list)
            dataset.parquet2webdataset()

            # webdataset转向量
            dataset.webdataset2inference()

            # 向量写入milvus
            fs, _ = fsspec.core.url_to_fs(embeddings_folder)
            emb_folder = os.path.join(embeddings_folder, "img_emb")
            metadata_folder = os.path.join(embeddings_folder, "metadata")
            if not os.path.exists(emb_folder) or not os.path.exists(metadata_folder):
                return
            metadata_files = [metadata_folder + "/" + p for p in next(fs.walk(metadata_folder))[2] if
                              p.endswith(".parquet")]

            for i in range(len(metadata_files)):
                metadata_file = os.path.join(metadata_folder, f"metadata_{i}.parquet")
                emb_file = os.path.join(emb_folder, f"img_emb_{i}.npy")
                df = pd.read_parquet(metadata_file)
                embs = np.load(emb_file)
                df["emb"] = embs.tolist()

                for d in df.itertuples():
                    if not d.caption:
                        continue

                    caption_info = json.loads(d.caption)
                    ids.append(caption_info.get("id", ""))
                    table_ids.append(caption_info.get("table_id", ""))
                    parent_table_ids.append(caption_info.get("parent_table_id", ""))
                    user_ids.append(caption_info.get("user_id", 0))
                    vectors.append(d.emb)
                    file_key = caption_info.get("key", "")
                    if file_key and len(d.emb) > 0:
                        await dh_vector.Media1024.get_or_create(key=file_key, defaults={
                            "vector": d.emb,
                        })

        if len(extracted_list) > 0:
            for m in extracted_list:
                ids.append(m.get("id", ""))
                table_ids.append(m.get("table_id", ""))
                parent_table_ids.append(m.get("parent_table_id", ""))
                user_ids.append(m.get("user_id", 0))
                vectors.append(m.get("vector", []))

        if len(ids) == 0:
            return

        # 插入数据
        batch_size = 1000
        for i in range(0, len(ids), batch_size):
            batch_data = [
                ids[i:i + batch_size],
                table_ids[i:i + batch_size],
                parent_table_ids[i:i + batch_size],
                user_ids[i:i + batch_size],
                vectors[i:i + batch_size],
            ]
            collection.insert(batch_data)
        collection.flush()

        # 删除webdataset、向量
        await self._remove_dataset_file(embeddings_folder, images_folder, urls_parquet_path)

    async def fav2milvus(self, fav_item_ids=None, dataset_dir=None, clip_model="cn_clip:ViT-H-14", clip_cache_path="./"):
        """"""
        if not fav_item_ids:
            return
        fav_items = await dh_user.FavItem.filter(id__in=fav_item_ids)
        user_fav_item_ids_dict = {}
        for fav_item in fav_items:
            user_id = fav_item.account_id
            user_fav_item_ids = user_fav_item_ids_dict.get(user_id, [])
            user_fav_item_ids.append(fav_item.id)
            user_fav_item_ids_dict[user_id] = user_fav_item_ids

        for user_id, user_fav_item_ids in user_fav_item_ids_dict.items():
            await self._user_fav2milvus(
                user_id=user_id,
                fav_item_ids=user_fav_item_ids,
                dataset_dir=dataset_dir,
                clip_model=clip_model,
                clip_cache_path=clip_cache_path
            )

    async def _remove_dataset_file(self, embeddings_folder, images_folder, urls_parquet_path):
        if os.path.exists(embeddings_folder):
            shutil.rmtree(embeddings_folder)
        if os.path.exists(images_folder):
            shutil.rmtree(images_folder)
        if os.path.exists(urls_parquet_path):
            os.remove(urls_parquet_path)

    async def filter_embeddings_remove_index_sets(self, dataset_name, embeddings=None, safety_model=None,
                                                  violence_detector=None, deduplicate_expr=None,
                                                  deduplicate_anns_field="vector"):
        """
        过滤掉不安全、暴力、重复的图片
        :param dataset_name:
        :param embeddings:
        :param safety_model: 安全模型（检测黄图）
        :param violence_detector: 暴力探测器
        :param deduplicate_expr: 消除重复正则
        :param deduplicate_anns_field: 消除重复查询字段
        :return:
        """
        remove_index_set = set()
        if not embeddings:
            return remove_index_set
        collection = self.get_connection(dataset_name)

        # # 测试 905391 media:1566993
        # test_results = collection.query('id in [1155256,4624825]', offset=0, limit=1000, output_fields=["vector"])
        # for v in test_results:
        #     vector = v.get("vector", [])
        #     if not vector:
        #         continue
        #     embeddings.append(vector)

        embeddings = normalized(embeddings)
        # print(len(embeddings))

        # 暴力
        if violence_detector is not None:
            remove_index_set |= set(self._get_violent_items(violence_detector, embeddings))

        # 安全
        if safety_model is not None:
            nsfw_to_remove = set(self._get_unsafe_items(safety_model, embeddings, threshold=0.5))
            remove_index_set.update(nsfw_to_remove)

        # 重复
        search_params = {"metric_type": "IP", "params": {"nprobe": 10}}
        for i, vector in enumerate(embeddings):
            results = collection.search([vector], deduplicate_anns_field, search_params, limit=1, expr=deduplicate_expr)
            if not results:
                continue
            if len(results[0].ids) == 0:
                continue
            # print(results[0].distances)
            if results[0].distances[0] > 0.988:
                remove_index_set.add(i)

        return remove_index_set

    def _get_violent_items(self, safety_prompts, embeddings):
        """获取暴利的特征"""
        safety_predictions = np.einsum("ij,kj->ik", embeddings, safety_prompts)
        safety_results = np.argmax(safety_predictions, axis=1)
        return np.where(safety_results == 1)[0]

    def _get_unsafe_items(self, safety_model, embeddings, threshold=0.5):
        """获取不安全的特征"""
        if isinstance(embeddings, Tensor):
            embeddings = embeddings.cpu().to(torch.float32).detach().numpy()
        if isinstance(embeddings, np.ndarray):
            if embeddings.dtype != 'float32':
                embeddings = embeddings.astype(np.float32)
        nsfw_values = safety_model.predict(embeddings, batch_size=embeddings.shape[0])
        x = np.array([e[0] for e in nsfw_values])
        return np.where(x > threshold)[0]


if __name__ == "__main__":
    """"""
    import math
    from tortoise import run_async
    from settings.config import MILVUS_HOST, MILVUS_PORT, CLIP_CACHE_PATH, DATASET_DIR, CLIP_MODEL
    from core.db import new_mysql_connection, close_mysql_connections, new_milvus_connection


    async def dev():
        """"""
        dataset_dir = DATASET_DIR
        clip_model = CLIP_MODEL
        clip_cache_path = CLIP_CACHE_PATH
        await new_mysql_connection()
        new_milvus_connection(MILVUS_HOST, MILVUS_PORT)

        # 业务代码
        storage = Storage(clip_cache_path=clip_cache_path)
        storage.create_project_schema()
        storage.create_fav_schema()

        # # 测试数据：知末案例
        # project_ids = await dh_project.Project.filter(source="zhimo", enabled=1, cate_id=2).values_list("id", flat=True)
        # batch_size = 50
        # page_size = int(math.ceil(len(project_ids) / batch_size))
        # for i in tqdm(range(page_size)):
        #     batch_project_ids = project_ids[i * batch_size: (i+1) * batch_size]
        #     await storage.project2milvus(
        #         project_ids=batch_project_ids,
        #         dataset_dir=dataset_dir,
        #         clip_model=clip_model,
        #         clip_cache_path=clip_cache_path
        #     )

        # # 测试数据：收藏夹
        # await storage.fav2milvus(
        #     fav_item_ids=[646923, 646919, 646951, 1089254],
        #     dataset_dir=dataset_dir,
        #     clip_model=clip_model,
        #     clip_cache_path=clip_cache_path
        # )

        # # # 过滤
        # device = "cuda" if torch.cuda.is_available() else "cpu"
        # safety_model = H14_NSFW_Detector(cache_folder=clip_cache_path, device=device)
        # embeddings = [[-0.019363403, 0.008735657, 0.009971619, 0.00093603134, 0.007347107, 0.023422241, 0.017425537, 0.014694214, -0.004650116, 0.010498047, -0.015823364, -0.01878357, -0.01789856, -0.0155181885, 0.053466797, 0.0059280396, 0.015289307, 0.0035915375, -0.0059509277, 0.010383606, -0.01637268, -0.060760498, -0.057739258, -0.04208374, 0.0206604, -0.014137268, -0.02029419, -0.0047302246, 0.0016784668, 0.003211975, 0.037902832, -0.019638062, 0.022720337, -0.0012311935, -0.028793335, -0.009338379, 0.035125732, 0.018310547, -0.0022525787, -0.026535034, 0.0026245117, -0.004760742, 0.041809082, 0.018081665, -0.010566711, 0.020080566, -0.006641388, -0.047607422, -0.039367676, -0.011169434, -0.009155273, 0.028320312, 0.012123108, -0.019195557, -0.03149414, -0.0084991455, 0.016204834, 0.019989014, -0.009002686, -0.020202637, 0.016326904, 0.034454346, -0.006576538, 0.025375366, 0.025436401, -0.013618469, 0.03744507, 0.018325806, 0.0036525726, -0.0041618347, 0.021408081, 0.0071754456, -0.02973938, -0.004299164, -0.0048942566, -0.019821167, -0.037902832, -0.009246826, -0.021957397, -0.026153564, 0.0060768127, -0.011299133, -0.0032749176, -0.019165039, -0.014457703, 0.002861023, -0.004016876, 0.028793335, 0.007007599, 0.015640259, -0.0075416565, 0.030929565, -0.014045715, -0.0013399124, -0.0027332306, -0.025100708, 0.045837402, -0.06726074, -0.0046691895, -0.022232056, 9.000301e-05, -0.0030097961, 0.0065612793, 0.026916504, 0.01625061, 0.023529053, -0.020721436, -0.012123108, -0.013069153, -0.0008163452, -0.0005836487, -0.0016202927, 0.02772522, 0.030914307, -0.010795593, -0.033691406, -0.0075302124, 0.03527832, -0.022277832, -0.00054740906, 0.032318115, -0.023101807, 0.013320923, 0.057617188, 0.04019165, -0.009384155, -0.018493652, -0.0085372925, 0.0056495667, -0.043121338, 0.020965576, -0.008102417, -0.0017366409, 0.027786255, -0.019195557, 0.0099105835, -0.029190063, 0.001950264, -0.016082764, 0.004108429, 0.009925842, -0.005104065, -0.02494812, -0.01713562, 0.008018494, -0.010482788, 0.0031433105, 0.04586792, -0.0051078796, 0.04248047, -0.014312744, 0.023086548, 0.0076675415, -0.0052337646, -0.0036029816, -0.018753052, -0.015487671, 0.008293152, 0.0038585663, -0.013038635, -0.016052246, 0.019714355, 0.013946533, -0.013092041, 0.024551392, 0.014381409, -0.0030956268, -0.010238647, 0.027267456, -0.0110321045, 0.031555176, -0.0020427704, 0.012878418, 0.017059326, 0.0027103424, 0.0033626556, 0.018295288, -0.029129028, -9.179115e-06, 0.031463623, 0.012413025, 0.023361206, 0.017700195, 0.008033752, -0.0018339157, -0.0024490356, 0.024459839, -0.01776123, 0.007843018, 0.016647339, 0.03164673, 0.017913818, -0.0009899139, 0.017593384, -0.007949829, 0.00071287155, 0.000248909, 0.015777588, 0.02482605, 0.035614014, 0.017623901, 0.0075416565, 0.022338867, 0.0022296906, -0.025527954, -0.025253296, 0.0317688, 0.047851562, -0.00015425682, 0.00040507317, 0.014976501, -0.023422241, 0.012359619, -0.061767578, 0.01133728, 0.01751709, 0.024734497, 0.05731201, 0.006416321, -0.07458496, -0.039001465, 0.0072021484, -0.022064209, -0.016448975, 0.024734497, -0.008636475, -0.00970459, 0.016189575, -0.07885742, -0.017974854, 0.013671875, -0.01626587, -0.0043792725, -0.017700195, 0.011222839, 0.043884277, -0.013069153, -0.031707764, 0.010978699, 0.025421143, -0.033813477, 0.0090408325, -0.047210693, 0.014968872, -0.005405426, -0.003660202, -0.007965088, 0.025268555, -0.028915405, 0.004512787, 0.039245605, -0.02027893, -0.018829346, 0.013511658, -0.001124382, 0.0022125244, 0.017105103, 0.0015525818, -0.024276733, 0.017410278, -0.032958984, -0.012435913, -0.0045814514, 0.019012451, -0.0072784424, -0.012435913, -0.032409668, -0.0026245117, 0.070739746, -0.0005865097, 0.0029354095, -0.0004901886, 0.014884949, 0.00010418892, -0.00019621849, -0.0053138733, 0.016479492, 0.05770874, -0.010971069, -0.0007739067, 0.020202637, -0.022216797, 0.036865234, 0.029663086, 0.018951416, 0.032104492, 0.033294678, 0.008331299, -0.0024700165, 0.014778137, -0.003894806, 0.0070343018, 0.01058197, 0.0035114288, -0.024414062, 0.008811951, 0.0051841736, 0.00995636, -0.026519775, -0.0063323975, 0.009178162, -0.0059280396, 0.033203125, -0.0121154785, 0.013832092, -0.026779175, 0.00033330917, 0.03366089, 0.0012483597, 0.006084442, 0.011009216, -0.01285553, -0.013999939, 0.0049476624, -0.00315094, 0.007537842, -0.020980835, 0.022872925, 0.016143799, -0.011703491, -0.0015087128, -0.028442383, -0.011550903, -0.007797241, 0.0026416779, -0.04296875, -0.003616333, -0.031234741, -0.035339355, -0.021347046, 0.023544312, 0.0005350113, -0.026535034, -0.013198853, 0.025482178, 0.015594482, -0.02532959, -0.025146484, -0.0340271, 0.016204834, 0.012641907, -0.021865845, 0.0619812, -0.0067634583, 0.040618896, 0.04840088, 0.0021381378, 0.008331299, -0.012199402, -0.03970337, -0.043670654, 0.06536865, -0.01209259, 0.010101318, 0.0057868958, -0.01939392, 0.0061798096, 0.0062332153, -0.014572144, -0.015975952, -0.017684937, -0.005493164, 0.03189087, -0.014808655, -0.015319824, 0.007835388, -0.012229919, 0.0155181885, -0.0022506714, -0.010482788, 0.004623413, 0.006843567, 0.008277893, -0.0033493042, -0.022720337, 0.011054993, 0.019302368, -0.009162903, 0.0068206787, -0.02835083, 0.0206604, -0.005493164, -0.033050537, -0.03149414, -0.02330017, -0.057495117, -0.021011353, -0.014541626, 0.023101807, -0.019699097, -0.0423584, -0.006729126, -0.014328003, 0.018051147, -0.041809082, -0.025100708, 0.014862061, -0.021102905, 0.059051514, 0.021957397, 0.0075416565, 0.0070228577, 0.0016202927, 0.016281128, -0.009269714, 0.023101807, 0.00025439262, 0.019836426, -0.0015039444, 0.041992188, 0.042877197, -0.0027580261, -0.011657715, 0.025115967, -0.0013542175, -0.0158844, 0.033447266, -0.0078125, -0.0039596558, -0.0019893646, 0.0037784576, -0.03152466, -0.03579712, -0.0058784485, -0.04660034, -0.01776123, 0.0016069412, -0.015106201, 0.011947632, 0.028900146, -0.010848999, 0.008392334, 0.0005965233, 0.0011606216, 0.011123657, -0.09289551, -0.04244995, -0.018066406, 0.0014238358, -0.018249512, -0.0038337708, 0.019119263, -0.038269043, -0.022079468, 0.019256592, -0.008232117, -0.051513672, -0.011985779, -0.0211792, -0.008232117, 0.087890625, -0.03677368, 0.014205933, -0.016021729, -0.016830444, 0.001947403, 0.007499695, 0.4272461, 0.012069702, 0.0077438354, -0.028366089, -0.023757935, -0.0118255615, 0.01789856, 0.027404785, 0.0050697327, -0.010299683, -0.015335083, -0.047424316, 0.00687027, 0.030044556, 0.010360718, 0.006702423, 0.0064811707, 0.022354126, -0.039215088, -0.002122879, -0.0748291, 0.018463135, 0.029846191, -0.025054932, -0.009918213, 0.014076233, 0.002111435, 0.007835388, -0.051635742, 0.016296387, 0.021514893, 0.052612305, 0.027832031, -0.004398346, 0.020019531, -0.011711121, 0.0074310303, -0.022445679, -0.026245117, -0.0181427, -0.03842163, 0.008605957, 0.01939392, -0.030914307, -0.011634827, 0.0066337585, 0.0066070557, 0.0131073, 0.0418396, -0.041992188, -0.031234741, 0.060333252, -0.006793976, 0.0592041, 0.004722595, -0.04324341, 0.0032081604, 0.0047035217, 0.012420654, -0.0033569336, -0.017822266, 0.009056091, -0.018814087, -0.030288696, 0.015625, 0.028259277, -0.01058197, 0.025817871, -0.01727295, 0.0041618347, 0.0234375, 0.01663208, 0.076049805, -0.039611816, 0.01285553, 0.028793335, -0.0066947937, -0.015899658, -0.059417725, 0.0044021606, 0.059417725, 0.028045654, 0.017684937, -0.037200928, 0.010055542, -0.082336426, 0.025222778, -0.016983032, 0.01222229, 0.04034424, -0.02722168, 0.027130127, 0.025985718, 0.034820557, 0.024719238, 0.005252838, -0.020584106, -0.020950317, 0.006626129, -0.029937744, 0.0014028549, 0.003112793, 0.002368927, -0.0074310303, 0.0023441315, 0.058776855, -0.0061035156, 0.037994385, -0.026779175, 0.040985107, 0.0158844, 0.019744873, -0.0057640076, -0.00982666, 0.010414124, -0.01838684, -0.024505615, -0.042541504, -0.017669678, -0.00091838837, 0.021331787, 0.02017212, 0.01789856, -0.019302368, -0.02407837, 0.007663727, 0.06781006, -0.03866577, 0.05670166, -0.012069702, 0.014640808, 0.035949707, -0.021835327, -0.05227661, 0.005584717, -0.032226562, -0.012207031, -0.005130768, -0.014122009, -0.010368347, 0.014732361, 0.0012998581, 0.012687683, -0.026245117, 0.045562744, 0.010749817, 0.041107178, 0.0335083, -0.020339966, -0.022003174, 0.0022411346, -0.019638062, -0.0072021484, 0.014312744, -0.05947876, -0.0044021606, 0.05960083, 0.022232056, -0.027877808, 0.024246216, -0.021484375, -0.020126343, -0.0024738312, -0.0067214966, -0.011222839, -0.006374359, 0.027938843, 0.023544312, -0.011932373, -0.04925537, 0.04067993, 0.028366089, -0.01902771, -0.00093460083, -0.047576904, -0.015655518, -0.030807495, -0.02722168, 0.03314209, 0.030807495, 0.032043457, -0.040252686, 0.010353088, 0.008666992, -0.019882202, -0.0031871796, 0.1496582, 0.031677246, -0.07940674, 0.039367676, -0.009269714, -0.12060547, 0.024398804, -0.006729126, -0.003917694, 0.017318726, 0.008110046, 0.015602112, -0.011245728, -0.03930664, 0.0073127747, 0.035888672, -0.0012578964, -0.03353882, 0.030059814, 0.04360962, 0.031707764, 0.01776123, 0.021957397, -0.03692627, 0.025985718, 0.06341553, 0.0128479, 0.012771606, 0.041748047, -0.023269653, -0.0105896, -0.032684326, 0.0135650635, 0.02949524, 0.017593384, 0.014022827, 0.0030345917, 0.010612488, 0.02583313, 0.0104599, -0.018417358, 0.003484726, 0.011001587, 0.0036830902, 0.0013036728, 0.023910522, 0.066833496, -0.012939453, 0.012176514, 0.009857178, 0.05731201, -0.052368164, -0.030059814, -0.026306152, -0.00141716, -0.010070801, 0.023803711, -0.011383057, 0.045837402, -0.008132935, -0.0039367676, -0.0013713837, 0.022537231, -0.007095337, -0.030273438, 0.011192322, -0.030960083, 0.0014324188, -0.039367676, 0.0011205673, 0.0041122437, 0.002960205, 0.011077881, -0.012123108, 0.045074463, -0.01727295, -0.016204834, 0.05041504, -0.010009766, 0.007911682, 0.001912117, -0.0340271, -0.013771057, -0.0041656494, 0.021255493, 0.018676758, 0.0104522705, 0.09454346, 0.013183594, -0.056427002, -0.0107421875, 0.0019359589, -0.023040771, -0.013771057, -2.6345253e-05, 0.008453369, 0.018325806, -0.097839355, -0.00818634, -0.00049972534, 0.013893127, -0.009025574, -0.0011129379, 0.03149414, 0.01789856, 0.00907135, 0.027069092, 0.022949219, 0.020553589, 0.0059547424, -0.010375977, 0.023925781, -0.39208984, 0.030715942, 0.02243042, 0.0032253265, 0.027450562, 0.016082764, 0.0035476685, 0.018951416, 0.033203125, -0.020751953, -0.007598877, -0.024719238, 0.06732178, 0.0043029785, 0.013916016, 0.0020332336, -0.016357422, 0.020339966, -0.029785156, -0.037261963, -0.014190674, -0.032409668, -0.05050659, -0.021865845, 0.016693115, -0.0023708344, 0.010360718, -0.0020217896, 0.01725769, 0.0077705383, 0.014099121, 0.008102417, -0.018051147, -0.004016876, -0.011161804, 0.068481445, 0.0043792725, -0.0054244995, -0.015296936, 0.011947632, 0.02810669, 0.017562866, -0.027023315, -0.02204895, -0.00856781, 0.014755249, 0.027862549, 0.010978699, -0.002565384, -0.0043754578, 0.012916565, -0.0025939941, -0.051086426, -0.018997192, -0.003479004, -0.039916992, -0.0046691895, 0.004146576, -0.024963379, 0.039642334, 0.007499695, 0.009941101, -0.039611816, 0.0057296753, -0.014122009, -0.00061655045, -0.010543823, -0.020599365, 0.0027656555, 0.03149414, -0.003921509, -0.02571106, -0.021652222, -0.022842407, 0.020629883, -0.028915405, 0.05328369, -0.0413208, 0.01776123, 0.027236938, 0.010154724, -0.0020980835, 0.007820129, 0.01737976, 0.014976501, -0.020339966, 0.003282547, 0.025283813, 0.002960205, 0.0067977905, 0.023529053, -0.013946533, -0.0033130646, -0.0032138824, -0.0038433075, -0.01802063, -0.011009216, 0.019897461, 0.0028820038, 0.03302002, 0.009788513, 0.012176514, 0.0317688, 0.003435135, 0.014373779, 0.00774765, -0.01247406, -0.013893127, 0.0071029663, -0.036010742, -0.031280518, -0.027389526, -0.03149414, -0.02609253, -0.111328125, -0.025527954, -0.006843567, 0.005466461, 0.006210327, -0.007873535, -0.0031318665, 0.014152527, -0.02532959, -0.007896423, 0.025604248, -0.040740967, -0.0006842613, 0.029037476, 0.004917145, 0.03314209, 0.0032863617, 0.0016031265, 0.001657486, -0.0034122467, 0.017868042, -0.019058228, 0.030822754, 0.027770996, 0.019378662, -0.042541504, 0.006958008, -0.039611816, 0.00013279915, 0.0052337646, 0.012329102, 0.013282776, 0.026885986, 0.008460999, -0.028869629, 0.0075263977, -0.0058250427, 0.021530151, 0.015899658, 0.020019531, 0.015975952, 0.043945312, -0.010231018, -0.014320374, -0.045318604, 0.011360168, -0.017105103, 0.018966675, -0.024520874, 0.004764557, 0.0034046173, -0.020904541, 0.03894043, 0.0013628006, -0.01234436, 0.024932861, -0.022460938, 0.030090332, 0.028259277, -0.028930664, -0.03213501, -0.020767212, -0.003200531, 0.009155273, 0.013999939, -0.014778137, -0.034484863, -0.0047569275, 0.020721436, -0.0119018555, -0.0119018555, -0.024353027, -0.0058937073, 0.011329651, 0.0115737915, 0.038360596, 0.026321411, -0.0105896, 0.007095337, -0.021347046, 0.014801025, 0.0070228577, -0.0028953552, 0.009628296, -0.01386261, 0.0064430237, -0.00749588, -0.03744507, 0.072753906, -0.00868988, -0.021728516, -0.035949707, -0.0061912537, 0.0015487671, -0.011039734, -0.0055274963, -0.0024719238, 0.034698486, -0.011512756, 0.031982422, 0.011856079, -0.0006966591, 0.019058228, -0.0031719208, -0.0012178421, -0.019592285, -0.009666443, -0.07519531, 0.006111145, -0.017333984, 0.010551453, 0.009544373, -0.0079193115, -0.007598877, -0.025634766, -0.0033836365, 0.026733398, 0.006576538, 0.01171875, -0.015594482, -0.005004883, -0.023101807, 0.018463135, 0.0055122375, -0.023040771, -0.029144287, 0.0027980804, 0.039001465, -0.033233643, 0.023147583, -0.0017299652, 0.029342651, -0.008293152, 0.009651184, 0.02583313, 0.007980347, 0.01612854, -0.010948181, 0.015716553, -0.03869629, 0.007537842, 0.012313843, 0.015541077, -0.004814148, -0.0124435425, -0.011161804, -0.01737976, 0.0395813, -0.010719299, 0.068847656, 0.032409668, 0.038238525, -0.02267456, -0.027664185, 0.012290955]]
        # remove_index_sets = await storage.filter_embeddings_remove_index_sets(storage.project_dataset_name, embeddings, safety_model, None)
        # print(remove_index_sets)

        await close_mysql_connections()

    run_async(dev())
