import sqlite3

# Lokasi database
DATABASE = 'database.db'

def print_users():
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()

    print("\n📋 Data dari tabel 'users':")
    for row in cursor.execute("SELECT * FROM users"):
        print(row)

    conn.close()

def print_reports():
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()

    print("\n📋 Data dari tabel 'reports':")
    for row in cursor.execute("SELECT * FROM reports"):
        print(row)

    conn.close()

if __name__ == "__main__":
    print_users()
    print_reports()
