"""
Run this once before starting the app to generate the RSA key pair used
for signing anomaly alerts:

    python keys/generate_keys.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from security.digital_signature import generate_key_pair  # noqa: E402

if __name__ == "__main__":
    generate_key_pair()
    print("RSA key pair generated in keys/")
