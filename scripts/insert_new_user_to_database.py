import getpass
import bcrypt
from database.database import db_connector

def hash_password(password):
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def create_user(username, email, password_hash):
    query = "INSERT INTO users.users (username, email, password_hash) VALUES (%s, %s, %s)"
    db_connector.run_query(query, (username, email, password_hash), return_df=False)

def main():
    username = input("Enter username: ")
    email = input("Enter email: ")
    password = getpass.getpass("Enter password: ")  # Securely read password
    password_hash = hash_password(password)

    create_user(username, email, password_hash)
    print(f"User {username} created successfully.")

if __name__ == "__main__":
    main()
