import sys
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend

def decrypt_file(encrypted_file, encrypted_key_file, output_file, private_key_file):
    # Load RSA private key
    with open(private_key_file, "rb") as f:
        private_key = serialization.load_pem_private_key(
            f.read(),
            password=None,
            backend=default_backend()
        )

    # Load encrypted AES key
    with open(encrypted_key_file, "rb") as f:
        encrypted_key = f.read()

    # Decrypt AES key
    aes_key = private_key.decrypt(
        encrypted_key,
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None
        )
    )

    # Load encrypted file
    with open(encrypted_file, "rb") as f:
        file_data = f.read()

    iv = file_data[:16]
    encrypted_data = file_data[16:]

    # Decrypt AES
    cipher = Cipher(algorithms.AES(aes_key), modes.CBC(iv), backend=default_backend())
    decryptor = cipher.decryptor()
    padded_data = decryptor.update(encrypted_data) + decryptor.finalize()

    # Remove padding
    padding_length = padded_data[-1]
    data = padded_data[:-padding_length]

    # Save decrypted file
    with open(output_file, "wb") as f:
        f.write(data)

    print("Decryption successful!")

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python decrypt.py input.enc input.enc.key output.csv private_key.pem")
        sys.exit(1)

    decrypt_file(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
