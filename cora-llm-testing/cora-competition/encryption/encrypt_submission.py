import os
import sys
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend

def encrypt_file(input_file, output_file, public_key_file):
    # Load RSA public key
    with open(public_key_file, "rb") as f:
        public_key = serialization.load_pem_public_key(
            f.read(),
            backend=default_backend()
        )

    # Generate random AES key (32 bytes = 256-bit)
    aes_key = os.urandom(32)
    iv = os.urandom(16)

    # Read file
    with open(input_file, "rb") as f:
        data = f.read()

    # Pad data manually (simple padding)
    padding_length = 16 - (len(data) % 16)
    data += bytes([padding_length]) * padding_length

    # Encrypt file with AES
    cipher = Cipher(algorithms.AES(aes_key), modes.CBC(iv), backend=default_backend())
    encryptor = cipher.encryptor()
    encrypted_data = encryptor.update(data) + encryptor.finalize()

    # Encrypt AES key with RSA
    encrypted_key = public_key.encrypt(
        aes_key,
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None
        )
    )

    # Save encrypted file (IV + encrypted_data)
    with open(output_file, "wb") as f:
        f.write(iv + encrypted_data)

    # Save encrypted AES key
    with open(output_file + ".key", "wb") as f:
        f.write(encrypted_key)

    print("Hybrid encryption successful!")
    print("Generated files:")
    print(f" - {output_file}")
    print(f" - {output_file}.key")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python encrypt_submission.py input.csv output.enc public_key.pem")
        sys.exit(1)

    encrypt_file(sys.argv[1], sys.argv[2], sys.argv[3])
