import os
import ssl
import urllib.request
import bz2
import logging

logger = logging.getLogger(__name__)

def ensure_openh264_dll(base_dir=None):
    """
    Ensure OpenH264 Cisco DLL is present in the working directory so OpenCV can 
    write H.264 (avc1) web-compatible MP4 files playable in Chrome, Edge, and Firefox.
    """
    target_dir = base_dir if base_dir else os.getcwd()
    dll_filename = 'openh264-2.5.0-win64.dll'
    target_dll_path = os.path.join(target_dir, dll_filename)

    if os.path.exists(target_dll_path) or os.path.exists(dll_filename):
        return True

    url = 'http://ciscobinary.openh264.org/openh264-2.5.0-win64.dll.bz2'
    bz2_path = os.path.join(target_dir, 'openh264.dll.bz2')

    try:
        logger.info(f"Downloading OpenH264 Cisco DLL from {url} for Web H.264 playback...")
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE

        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, context=ctx) as response, open(bz2_path, 'wb') as out_file:
            out_file.write(response.read())

        with bz2.open(bz2_path, 'rb') as f_in, open(target_dll_path, 'wb') as f_out:
            f_out.write(f_in.read())

        if os.path.exists(bz2_path):
            os.remove(bz2_path)

        logger.info("OpenH264 Cisco DLL installed successfully.")
        return True
    except Exception as e:
        logger.warning(f"Could not download OpenH264 DLL: {e}")
        if os.path.exists(bz2_path):
            os.remove(bz2_path)
        return False
