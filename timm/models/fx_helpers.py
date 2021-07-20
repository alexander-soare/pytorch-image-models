def fx_float_to_int(x: float) -> int:
    """
    Symbolic tracing helper as inbuilt int can't be called directly with a Proxy
    """
    return int(x)


def fx_and(a: bool, b: bool) -> bool:
    """
    Symbolic tracing helper for to replace normal `* and *`. Use it in `torch._assert`.
    """
    return (a and b)