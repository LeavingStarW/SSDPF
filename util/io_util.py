import argparse
import pydicom as dicom

from datetime import date, datetime
from sys import stderr



#w 设置多进程
def set_spawn_enabled():
    import torch.multiprocessing as mp
    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        pass



#w 参数类中的布尔值会被认定为字符串类型
def str_to_bool(v):
    if v.lower() in ('true'):
        return True
    elif v.lower() in ('false'):
        return False
    else:
        raise ValueError('error, util/io_util/str_to_bool')




#w 把逗号分隔的数字转换为列表
def args_to_list(arg):
    arg_list = [int(d) for d in str(arg).split(',')]
    return arg_list





def read_dicom(dicom_path):
    """Read a DICOM object from path to a DICOM.

    Args:
        dicom_path: Path to DICOM file to read.

    Raises:
        IOError: If we can't find a file at the path given.
    """
    dcm = None
    try:
        with open(dicom_path, 'rb') as dicom_file:
            dcm = dicom.dcmread(dicom_file)
    except IOError:
        print('Warning: Failed to open {}'.format(dicom_path))

    return dcm





def json_encoder(obj):
    """JSON encoders for objects not normally supported by the JSON library."""
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    raise TypeError("Type %s not serializable" % type(obj))





def try_parse(s, type_fn=int):
    """Try parsing a string into type given by `type_fn`, and return None on ValueError."""
    i = None
    try:
        i = type_fn(s)
    except ValueError:
        pass
    return i
