import os

__all__ = ["what"]

def what(file, h=None):
    if h is None:
        if isinstance(file, (str, os.PathLike)):
            try:
                with open(file, 'rb') as f:
                    h = f.read(32)
            except OSError:
                return None
        else:
            location = file.tell()
            h = file.read(32)
            file.seek(location)
            
    if not h:
        return None
        
    if h.startswith(b'\xff\xd8\xff'):
        return 'jpeg'
    if h.startswith(b'\x89PNG\r\n\x1a\n'):
        return 'png'
    if h.startswith(b'GIF8'):
        return 'gif'
    if h.startswith(b'RIFF') and h[8:12] == b'WEBP':
        return 'webp'
    if h.startswith(b'BM'):
        return 'bmp'
    
    return None

