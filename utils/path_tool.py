import os


def get_path() -> str:
    """获取工程所在根目录"""
    current_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_path)
    project_root = os.path.dirname(current_dir)
    return project_root


def get_abs_path(relative_path: str = None) -> str:
    """传入相对路径获取绝对路径"""
    project_root = get_path()
    return os.path.join(project_root, relative_path)


if __name__ == '__main__':
    print(get_abs_path("utils/path_tool.py"))