from fastapi import FastAPI
from pymilvus import connections
from tortoise import Tortoise
from tortoise.contrib.fastapi import register_tortoise
from settings.config import MYSQL_USER, MYSQL_HOST, MYSQL_PASSWORD, MYSQL_PORT, MYSQL_CACHE_HOST, MYSQL_CACHE_USER, \
    MYSQL_CACHE_PORT, MYSQL_CACHE_PASSWORD


async def new_mysql_connection():
    """初始化MySQL链接"""
    config = mysql_config()
    await Tortoise.init(config)


async def close_mysql_connections():
    await Tortoise.close_connections()


def new_milvus_connection(host, port):
    """链接milvus"""
    connections.connect(
        host=host,
        port=port
    )


def new_app_mysql_connection(app: FastAPI) -> None:
    """ 初始化app数据库连接
    """
    config = mysql_config()
    return register_tortoise(app, config, generate_schemas=False, add_exception_handlers=True)


def mysql_config():
    config = {
        "connections": {
            "dh_project": f"mysql://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOST}:{MYSQL_PORT}/dh_project",
            "dh_user": f"mysql://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOST}:{MYSQL_PORT}/dh_user",
            "dh_vector": f"mysql://{MYSQL_CACHE_USER}:{MYSQL_CACHE_PASSWORD}@{MYSQL_CACHE_HOST}:{MYSQL_CACHE_PORT}/dh_vector",
        },
        "apps": {
            "dh_project": {
                "models": ["model.dh_project"],
                "default_connection": "dh_project",
            },
            "dh_user": {
                "models": ["model.dh_user"],
                "default_connection": "dh_user",
            },
            "dh_vector": {
                "models": ["model.dh_vector"],
                "default_connection": "dh_vector",
            },
        },
    }
    return config
