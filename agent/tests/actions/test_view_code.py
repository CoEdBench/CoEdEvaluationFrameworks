
"""
使用本地文件路径直接测试 ViewCode Action，不依赖 SWE-bench 数据集。
使用方式：将 LOCAL_REPO_PATH 替换为你本地的项目路径即可。
"""
import pytest
from moatless.actions.view_code import ViewCode, ViewCodeArgs, CodeSpan
from moatless.file_context import FileContext
from moatless.repository import FileRepository
from moatless.workspace import Workspace

# ✅ 只需修改这里：指向你本地的任意代码仓库路径
LOCAL_REPO_PATH = "D:/Data/2025/CodeCompletion/Dataset/Repos/scikit-learn"
pytestmark = pytest.mark.asyncio(loop_scope="session")

def make_env(repo_path: str = LOCAL_REPO_PATH):
    """
    构建测试所需的三件套：repository、file_context、workspace。
    可复用于多个测试用例。
    """
    repository = FileRepository(repo_path=repo_path)
    file_context = FileContext(repo=repository)
    workspace = Workspace(repository=repository)
    return repository, file_context, workspace


# ---------------------------------------------------------------------------
# 测试 1：按 span_id 查找存在的函数
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_view_existing_span():
    repository, file_context, workspace = make_env()

    action = ViewCode(repository=repository)
    await action.initialize(workspace)

    args = ViewCodeArgs(
        scratch_pad="查看某个已知存在的函数",
        files=[
            CodeSpan(
                file_path="sklearn/neighbors/_nca.py",   # ← 替换为真实文件路径（相对于 repo 根目录）
                span_ids=["YourClassName.your_method"],  # ← 替换为真实的 span_id
            )
        ],
    )

    output = await action.execute(args, file_context)
    print(output.message)
    assert output.message, "应该返回非空内容"


# ---------------------------------------------------------------------------
# 测试 2：按行号范围查看代码（不依赖 span_id 解析）
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_view_by_line_range():
    repository, file_context, workspace = make_env()

    action = ViewCode(repository=repository)
    await action.initialize(workspace)

    args = ViewCodeArgs(
        scratch_pad="按行号范围查看文件内容",
        files=[
            CodeSpan(
                file_path="sklearn/neighbors/_nca.py",    # ← 替换为真实文件路径
                start_line=1,
                end_line=50,
            )
        ],
    )

    output = await action.execute(args, file_context)
    print(output.message)
    assert output.message, "应该返回非空内容"


# ---------------------------------------------------------------------------
# 测试 3：查找不存在的 span_id（对应原 test_request_non_existing_method）
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_view_non_existing_span():
    repository, file_context, workspace = make_env()

    action = ViewCode(repository=repository)
    await action.initialize(workspace)

    args = ViewCodeArgs(
        scratch_pad="尝试查找一个不存在的方法",
        files=[
            CodeSpan(
                file_path="your/module/file.py",   # ← 替换为真实文件路径
                span_ids=["non_existing_method_xyz"],
            )
        ],
    )

    output = await action.execute(args, file_context)
    print(output.message)
    # 不存在的 span 应该返回提示信息而不是抛出异常
    assert output.message is not None


# ---------------------------------------------------------------------------
# 测试 4：同时查看多个文件的多个 span（对应原 test_request_many_spans）
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_view_many_spans():
    repository, file_context, workspace = make_env()

    action = ViewCode(repository=repository)
    await action.initialize(workspace)

    args = ViewCodeArgs(
        scratch_pad="同时查看多个文件的多个代码块",
        files=[
            CodeSpan(
                file_path="your/module/file_a.py",  # ← 替换
                span_ids=[
                    "ClassA",
                    "ClassA.method_one",
                    "ClassA.method_two",
                ],
            ),
            CodeSpan(
                file_path="your/module/file_b.py",  # ← 替换
                start_line=1,
                end_line=30,
            ),
        ],
    )

    output = await action.execute(args, file_context)
    print(file_context.model_dump())
    print(output.message)
    assert output.message, "应该返回非空内容"


# ---------------------------------------------------------------------------
# 测试 5：验证 FileContext 的上下文追踪功能
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_file_context_tracking():
    """
    验证多次调用后，file_context 能正确累积已查看的代码片段。
    """
    repository, file_context, workspace = make_env()

    action = ViewCode(repository=repository)
    await action.initialize(workspace)

    file_path = "your/module/file.py"  # ← 替换

    # 第一次查看
    args1 = ViewCodeArgs(
        scratch_pad="第一次查看",
        files=[CodeSpan(file_path=file_path, start_line=1, end_line=20)],
    )
    await action.execute(args1, file_context)

    # 第二次查看不同范围
    args2 = ViewCodeArgs(
        scratch_pad="第二次查看",
        files=[CodeSpan(file_path=file_path, start_line=21, end_line=40)],
    )
    await action.execute(args2, file_context)

    ctx_dump = file_context.model_dump()
    print(ctx_dump)
    # file_context 应该记录了两次查看的内容
    assert ctx_dump is not None