
import hashlib
import hmac
import secrets

ITERATIONS = 310000


def hash_password(password):
    """Store a salted PBKDF2 password hash, never plain text."""
    salt = secrets.token_hex(16)

    password_hash = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        ITERATIONS
    ).hex()

    return f"pbkdf2_sha256${ITERATIONS}${salt}${password_hash}"


def verify_password(password, stored_hash):
    """Verify a password against its stored hash."""
    try:
        algorithm, iterations, salt, expected_hash = stored_hash.split("$")

        if algorithm != "pbkdf2_sha256":
            return False

        actual_hash = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            salt.encode("utf-8"),
            int(iterations)
        ).hex()

        return hmac.compare_digest(actual_hash, expected_hash)

    except (ValueError, TypeError):
        return False
