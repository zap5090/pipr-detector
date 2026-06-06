import sqlite3

def init_db():
    conn = sqlite3.connect("inventario.db")
    cursor = conn.cursor()
    # Tabla específica para tuberías
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS productos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            nombre TEXT NOT NULL,
            diametro REAL,
            longitud REAL,
            cantidad INTEGER DEFAULT 0,
            precio REAL
        )
    ''')
    conn.commit()
    conn.close()
    print("Base de datos lista.")
if __name__ == "__main__":
    init_db()