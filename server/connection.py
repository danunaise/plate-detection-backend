import psycopg2

def get_db_connection():
    try:
        conn = psycopg2.connect(
            dbname="license-plate",
            user="postgres",
            password="danunai",
            host="localhost",
            port="5432"
        )
        return conn
    except psycopg2.Error as e:
        print(f"Database connection error: {e}")
        return None