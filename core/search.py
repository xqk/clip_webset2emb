import json
from collections import defaultdict

import torch
import numpy as np
import cn_clip.clip as clip

from pymilvus import Collection
from torch import Tensor

from core.inference.mapper import normalized
from core.load_clip import load_cn_clip
from lib.utils import paraphrase_mining_embeddings
from lib.h14_nsfw_model import H14_NSFW_Detector


class Search:
    """"""
    def __init__(self, clip_model="cn_clip:ViT-H-14", clip_cache_path="./", deduplicate=False, use_safety_model=False,
                 use_violence_detector=False):
        """"""
        self.connections = {}

        self.project_dataset_name = "zhuke_pro_project_photo"
        self.fav_dataset_name = "zhuke_pro_fav"

        self.clip_cache_path = clip_cache_path
        self.clip_model = clip_model

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.safety_model = H14_NSFW_Detector(cache_folder=clip_cache_path, device=self.device)
        self.deduplicate = deduplicate
        self.use_safety_model = use_safety_model
        self.use_violence_detector = use_violence_detector

    def get_connection(self, collection_name):
        """"""
        collection = self.connections.get(collection_name)
        if not collection:
            collection = Collection(name=collection_name)
            self.connections[collection_name] = collection

        return collection

    def projects(self, keyword, top_k=3000, threshold:float=0.1, user_filter=False):
        """搜索案例"""
        collection = self.get_connection(self.project_dataset_name)

        # 提取搜索词向量

        model, preprocess = load_cn_clip(self.clip_model, device=self.device, download_root=self.clip_cache_path)
        text = clip.tokenize([keyword]).to(self.device)
        # print(text)
        with torch.no_grad():
            text_features = model.encode_text(text)
            text_features /= text_features.norm(dim=-1, keepdim=True)
        q_records = text_features.tolist()

        # 搜索
        search_params = {"metric_type": "IP", "params": {"nprobe": 10}}
        output_fields = ["parent_table_id", "media_id"]
        if user_filter:
            output_fields = []
        results = collection.search(q_records, "vector", search_params, limit=top_k, output_fields=output_fields)

        del q_records
        if not results:
            return []

        if len(results[0].ids) == 0:
            return []

        if user_filter:
            id_distance_map = dict(zip(results[0].ids, results[0].distances))
            items = collection.query(
                expr=f"id in {json.dumps(list(results[0].ids))}",
                offset=0,
                limit=top_k,
                output_fields=["id", "parent_table_id", "media_id", "vector"],
                consistency_level="Strong"
            )
            for item in items:
                item["distance"] = id_distance_map.get(item.get('id', ""))

            items = sorted(items, key=lambda x: x["distance"], reverse=True)

            # 过滤
            distances, ids, embeddings, parent_table_ids = [], [], [], []
            for item in items:
                distances.append(item.get('distance', 0))
                ids.append(item.get("media_id", 0))
                embeddings.append(item.get('vector', []))
                parent_table_ids.append(item.get('parent_table_id', []))

            results = ids
            nb_results = len(results)
            result_ids = results[:nb_results]
            result_distances = distances[:nb_results]
            result_embeddings = embeddings[:nb_results]
            result_embeddings = normalized(result_embeddings)
            local_indices_to_remove = self.post_filter(
                embeddings=result_embeddings,
                deduplicate=self.deduplicate,
                use_safety_model=self.use_safety_model,
                use_violence_detector=self.use_violence_detector,
                violence_detector=self.use_violence_detector
            )

            milvus_ids_to_remove = set()
            for local_index in local_indices_to_remove:
                milvus_ids_to_remove.add(result_ids[local_index])

            # 删除大变量
            del items, distances, ids, embeddings

            # 过滤掉需要删除的数据
            knn_items = []
            for mid, distance, parent_table_id in zip(result_ids, result_distances, parent_table_ids):
                if mid not in milvus_ids_to_remove:
                    if distance < threshold:
                        continue
                    
                    milvus_ids_to_remove.add(mid)
                    knn_items.append({
                        "id": mid,
                        "parent_table_id": parent_table_id,
                        "distance": distance
                    })
        else:
            items = []
            for hit in results[0]:
                entity = hit.entity
                distance = hit.distance
                if distance < threshold:
                    continue
                    
                items.append({
                    "id": entity.get("media_id"),
                    "parent_table_id": entity.get("parent_table_id"),
                    "distance": distance
                })
            knn_items = items

        # 按parent_table_id聚类，保留最相似的一条
        parent_table_id_set = set()
        need_items = []
        for item in knn_items:
            parent_table_id = item.get("parent_table_id")
            if parent_table_id in parent_table_id_set:
                continue
            parent_table_id_set.add(parent_table_id)
            need_items.append(item)

        del parent_table_id_set, knn_items

        return need_items

    def favs(self, user_id, keyword, top_k=3000, user_filter=False):
        """搜索收藏夹"""
        collection = self.get_connection(self.fav_dataset_name)

        # 提取搜索词向量
        model, preprocess = load_cn_clip(self.clip_model, device=self.device, download_root=self.clip_cache_path)
        text = clip.tokenize([keyword]).to(self.device)
        # print(text)
        with torch.no_grad():
            text_features = model.encode_text(text)
            text_features /= text_features.norm(dim=-1, keepdim=True)
        q_records = text_features.tolist()

        # 搜索
        search_params = {"metric_type": "IP", "params": {"nprobe": 10}}
        output_fields = ["table_id", "parent_table_id"]
        if user_filter:
            output_fields = []
        results = collection.search(q_records, "vector", search_params, limit=top_k, output_fields=output_fields,
                                    expr=f"user_id=={user_id}")

        del q_records
        if not results:
            return []

        if len(results[0].ids) == 0:
            return []

        if user_filter:
            id_distance_map = dict(zip(results[0].ids, results[0].distances))
            items = collection.query(
                expr=f"id in {json.dumps(list(results[0].ids))}",
                offset=0,
                limit=top_k,
                output_fields=["id", "table_id", "parent_table_id", "vector"],
                consistency_level="Strong"
            )
            for item in items:
                item["distance"] = id_distance_map.get(item.get('id', 0))

            items = sorted(items, key=lambda x: x["distance"])

            # 过滤
            distances, ids, embeddings, table_ids, parent_table_ids = [], [], [], [], []
            for item in items:
                distances.append(item.get('distance', 0))
                ids.append(item.get("id", 0))
                embeddings.append(item.get('vector', []))
                table_ids.append(item.get('table_id', []))
                parent_table_ids.append(item.get('parent_table_id', []))

            results = ids
            nb_results = len(results)
            result_ids = results[:nb_results]
            result_distances = distances[:nb_results]
            result_embeddings = embeddings[:nb_results]
            result_embeddings = normalized(result_embeddings)
            local_indices_to_remove = self.post_filter(
                embeddings=result_embeddings,
                deduplicate=self.deduplicate,
                use_safety_model=self.use_safety_model,
                use_violence_detector=self.use_violence_detector,
                violence_detector=self.use_violence_detector
            )

            milvus_ids_to_remove = set()
            for local_index in local_indices_to_remove:
                milvus_ids_to_remove.add(result_ids[local_index])

            # 删除大变量
            del items, distances, ids, embeddings

            # 过滤掉需要删除的数据
            knn_items = []
            for mid, distance, table_id, parent_table_id in zip(result_ids, result_distances, table_ids, parent_table_ids):
                if mid not in milvus_ids_to_remove:
                    milvus_ids_to_remove.add(mid)
                    knn_items.append({
                        "table_id": table_id,
                        "parent_table_id": parent_table_id,
                        "distance": distance
                    })
        else:
            items = []
            for hit in results[0]:
                entity = hit.entity
                items.append({
                    "table_id": entity.get("table_id"),
                    "parent_table_id": entity.get("parent_table_id"),
                    "distance": hit.distance
                })
            knn_items = items

        # 按parent_table_id聚类，保留最相似的一条
        parent_table_id_set = set()
        need_items = []
        for item in knn_items:
            parent_table_id = item.get("parent_table_id")
            if parent_table_id in parent_table_id_set:
                continue
            parent_table_id_set.add(parent_table_id)
            need_items.append(item)

        del parent_table_id_set, knn_items

        return need_items

    def connected_components(self, neighbors):
        """相似-重复过滤"""
        seen = set()

        def component(node):
            r = []
            nodes = set([node])
            while nodes:
                node = nodes.pop()
                seen.add(node)
                nodes |= set(neighbors[node]) - seen
                r.append(node)
            return r

        u = []
        for node in list(neighbors.keys()):
            if node not in seen:
                u.append(component(node))
        return u

    def get_non_uniques(self, embeddings, top_k=100, threshold=0.90):
        """ 向量去重 """
        if not isinstance(embeddings, Tensor):
            embeddings = torch.tensor(embeddings).to(self.device)

        res = paraphrase_mining_embeddings(embeddings, top_k=top_k)
        same_mapping = defaultdict(list)
        for i, item in enumerate(res):
            score = item[0]
            if score >= threshold:
                same_mapping[item[1]].extend(item[1:])

        # 向量簇筛选
        groups = self.connected_components(same_mapping)
        non_uniques = set()
        for g in groups:
            for e in g[1:]:
                non_uniques.add(e)

        del embeddings, res, groups, same_mapping

        return list(non_uniques)

    def connected_components_dedup(self, embeddings):
        non_uniques = self.get_non_uniques(embeddings)
        del embeddings
        return non_uniques

    def get_unsafe_items(self, safety_model, embeddings, threshold=0.5):
        """获取不安全的特征"""
        if isinstance(embeddings, Tensor):
            embeddings = embeddings.cpu().to(torch.float32).detach().numpy()
        if isinstance(embeddings, np.ndarray):
            if embeddings.dtype != 'float32':
                embeddings = embeddings.astype(np.float32)
        nsfw_values = safety_model.predict(embeddings, batch_size=embeddings.shape[0])
        x = np.array([e[0] for e in nsfw_values])
        return np.where(x > threshold)[0]

    def get_violent_items(self, safety_prompts, embeddings):
        """获取暴利的特征"""
        safety_predictions = np.einsum("ij,kj->ik", embeddings, safety_prompts)
        safety_results = np.argmax(safety_predictions, axis=1)
        return np.where(safety_results == 1)[0]

    def post_filter(
            self, embeddings, deduplicate, use_safety_model, use_violence_detector, violence_detector
    ):
        " 通过向量过滤 "
        """post filter results : dedup, safety, violence"""
        to_remove = set()
        if deduplicate:
            # 去重
            dedup_to_remove = set(self.connected_components_dedup(embeddings))
            to_remove.update(dedup_to_remove)

        if use_violence_detector and violence_detector is not None:
            # 暴力
            to_remove |= set(self.get_violent_items(violence_detector, embeddings))

        if use_safety_model and self.safety_model is not None:
            # nswf
            nsfw_to_remove = set(self.get_unsafe_items(self.safety_model, embeddings, threshold=0.5))
            to_remove.update(nsfw_to_remove)

        return to_remove


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

        search = Search(clip_model=clip_model, clip_cache_path=clip_cache_path)

        # # 测试案例
        need_items = search.projects("客厅", top_k=30)
        for i in need_items:
            print(i)

        # # 测试收藏
        # need_items = search.favs(220, "客厅", top_k=30)
        # for i in need_items:
        #     print(i)

        await close_mysql_connections()


    run_async(dev())
