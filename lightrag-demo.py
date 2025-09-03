import os
import inspect
import logging
import asyncio
import nest_asyncio
from typing import Optional, List
import traceback

# 应用 nest_asyncio 以支持嵌套事件循环
nest_asyncio.apply()

try:
    from lightrag import LightRAG, QueryParam
    from lightrag.llm.ollama import ollama_model_complete, ollama_embed
    from lightrag.utils import EmbeddingFunc
    from lightrag.kg.shared_storage import initialize_pipeline_status
except ImportError as e:
    logging.error(f"导入 LightRAG 模块失败: {e}")
    logging.error("请确保已正确安装 lightrag-hku: pip install lightrag-hku")
    exit(1)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 配置常量
WORKING_DIR = "./dickens2"
DOCS_DIR = './docs'
SUPPORTED_FILE_EXTENSIONS = {'.txt', '.md', '.doc', '.docx', '.pdf'}


def ensure_directories():
    """确保必要的目录存在"""
    for directory in [WORKING_DIR, DOCS_DIR]:
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            logger.info(f"创建目录: {directory}")


async def initialize_rag() -> Optional[LightRAG]:
    """初始化 RAG 系统，包含错误处理和参数验证"""
    try:
        # 创建 embedding 函数
        embedding_func = EmbeddingFunc(
            embedding_dim=1024,
            max_token_size=8192,
            func=lambda texts: ollama_embed(
                texts,
                embed_model="bge-m3",
                host="http://localhost:11434"
            ),
        )

        # 尝试使用完整参数初始化
        rag = LightRAG(
            working_dir=WORKING_DIR,
            llm_model_func=ollama_model_complete,
            llm_model_name="qwen2.5",
            llm_model_max_async=4,
            llm_model_max_token_size=32768,
            llm_model_kwargs={
                "host": "http://localhost:11434",
                "options": {"num_ctx": 32768},
            },
            embedding_func=embedding_func,
        )

        logger.info("LightRAG 实例创建成功")

        # 初始化存储
        await rag.initialize_storages()
        logger.info("存储初始化完成")

        # 初始化管道状态
        await initialize_pipeline_status()
        logger.info("管道状态初始化完成")

        return rag

    except TypeError as e:
        logger.warning(f"使用完整参数初始化失败: {e}")
        logger.info("尝试使用基础参数初始化...")

        # 回退到基础参数
        try:
            rag = LightRAG(
                working_dir=WORKING_DIR,
                llm_model_func=ollama_model_complete,
                llm_model_name="qwen2.5",
                llm_model_kwargs={
                    "host": "http://localhost:11434",
                    "options": {"num_ctx": 32768},
                },
                embedding_func=embedding_func,
            )

            await rag.initialize_storages()
            logger.info("使用基础参数初始化成功")
            return rag

        except Exception as e2:
            logger.error(f"基础参数初始化也失败: {e2}")
            return None

    except Exception as e:
        logger.error(f"初始化 RAG 系统时发生未知错误: {e}")
        traceback.print_exc()
        return None


async def print_stream(stream):
    """异步打印流式响应"""
    try:
        async for chunk in stream:
            print(chunk, end="", flush=True)
        print()  # 添加换行
    except Exception as e:
        logger.error(f"打印流式响应时出错: {e}")


def is_supported_file(filename: str) -> bool:
    """检查文件是否为支持的格式"""
    _, ext = os.path.splitext(filename.lower())
    return ext in SUPPORTED_FILE_EXTENSIONS


def load_docs_from_folder(folder_path: str, rag: LightRAG) -> int:
    """从文件夹加载文档，返回加载的文件数量"""
    if not os.path.exists(folder_path):
        logger.warning(f"文档文件夹不存在: {folder_path}")
        logger.info(f"请在 {folder_path} 目录下放置您的文档文件")
        return 0

    loaded_count = 0
    failed_count = 0

    logger.info(f"开始从 {folder_path} 加载文档...")

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        # 跳过目录和不支持的文件
        if not os.path.isfile(file_path) or not is_supported_file(filename):
            continue

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # 检查内容是否为空
            if not content.strip():
                logger.warning(f"文件 {filename} 为空，跳过")
                continue

            # 插入文档
            rag.insert(content)
            loaded_count += 1
            logger.info(f"✅ 成功加载: {filename} ({len(content)} 字符)")

        except UnicodeDecodeError:
            logger.warning(f"❌ 编码错误，跳过文件: {filename}")
            failed_count += 1
        except Exception as e:
            logger.error(f"❌ 加载文件 {filename} 时出错: {e}")
            failed_count += 1

    logger.info(f"文档加载完成: 成功 {loaded_count} 个，失败 {failed_count} 个")
    return loaded_count


async def query_rag_async(query: str, rag: LightRAG):
    """异步查询 RAG 系统"""
    try:
        # 创建查询参数
        param = QueryParam(mode="global", stream=True)

        # 执行查询
        resp = rag.query(query, param=param)

        # 处理响应
        if inspect.isasyncgen(resp):
            await print_stream(resp)
        else:
            print(resp)

    except Exception as e:
        logger.error(f"查询时出错: {e}")

        # 尝试非流式查询作为回退
        try:
            logger.info("尝试非流式查询...")
            param = QueryParam(mode="global")
            resp = rag.query(query, param=param)
            print(resp)
        except Exception as e2:
            logger.error(f"非流式查询也失败: {e2}")
            print("抱歉，查询失败。请检查您的查询或系统状态。")


def query_rag(query: str, rag: LightRAG):
    """同步查询接口"""
    asyncio.run(query_rag_async(query, rag))


def print_welcome_message():
    """打印欢迎信息"""
    print("=" * 60)
    print("🚀 LightRAG 系统启动成功!")
    print("=" * 60)
    print("使用说明:")
    print("- 输入您的问题进行查询")
    print("- 输入 'exit' 或 'quit' 退出系统")
    print("- 输入 'help' 查看更多命令")
    print("=" * 60)


def print_help():
    """打印帮助信息"""
    print("\n📚 可用命令:")
    print("  exit/quit - 退出系统")
    print("  help      - 显示此帮助信息")
    print("  clear     - 清屏")
    print("  status    - 显示系统状态")


async def main():
    """主函数"""
    # 确保目录存在
    ensure_directories()

    # 初始化 RAG 系统
    logger.info("正在初始化 LightRAG 系统...")
    rag = await initialize_rag()

    if rag is None:
        logger.error("RAG 系统初始化失败，程序退出")
        return

    # 加载文档
    doc_count = load_docs_from_folder(DOCS_DIR, rag)

    if doc_count == 0:
        logger.warning("没有加载任何文档，系统将正常运行但可能无法提供有用的答案")

    # 打印欢迎信息
    print_welcome_message()

    # 主循环
    try:
        while True:
            query = input("\n💬 请输入您的问题: ").strip()

            if not query:
                continue

            if query.lower() in ["exit", "quit", "退出"]:
                print("👋 再见!")
                break
            elif query.lower() in ["help", "帮助"]:
                print_help()
                continue
            elif query.lower() == "clear":
                os.system('cls' if os.name == 'nt' else 'clear')
                continue
            elif query.lower() == "status":
                print(f"📊 系统状态: 运行中 | 工作目录: {WORKING_DIR} | 文档数: {doc_count}")
                continue

            print("\n🔍 正在查询...")
            print("-" * 50)
            query_rag(query, rag)
            print("-" * 50)

    except KeyboardInterrupt:
        print("\n\n👋 检测到 Ctrl+C，正在退出...")
    except Exception as e:
        logger.error(f"主循环中发生错误: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        logger.error(f"程序启动失败: {e}")
        traceback.print_exc()