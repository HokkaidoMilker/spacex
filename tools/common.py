def clean_text(text: str) -> str:
    """清理文本中的非 ASCII 特殊字符（如 non-breaking space \\xa0）。

    Args:
        text: 原始文本

    Returns:
        清理后的文本
    """
    if not text:
        return text
    # 替换常见的不可见 Unicode 字符
    return (
        text.replace("\xa0", " ")       # non-breaking space → 普通空格
            .replace("​", "")       # zero-width space
            .replace("﻿", "")       # BOM
            .replace("‎", "")       # left-to-right mark
            .replace("‏", "")       # right-to-left mark
    )
