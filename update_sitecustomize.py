import os
os.environ['SSL_CERT_FILE'] = 'D:/Anaconda/envs/agentworld/Lib/site-packages/certifi/cacert.pem'

# 修补ssl模块,避免加载Windows证书存储
import ssl
_original_load_default_certs = ssl.SSLContext.load_default_certs

def _patched_load_default_certs(self, purpose=ssl.Purpose.SERVER_AUTH):
    try:
        import certifi
        self.load_verify_locations(cafile=certifi.where())
    except:
        # 如果certifi不可用,回退到原始方法
        _original_load_default_certs(self, purpose)

ssl.SSLContext.load_default_certs = _patched_load_default_certs
