from flask import Flask, render_template, redirect, url_for, request, send_file
from werkzeug.utils import secure_filename
import os
import sqlite3
from IA_logica import contar_tuberias, procesar_imagen
import pandas as pd

app = Flask(__name__)
os.makedirs(os.path.join(app.root_path, 'static'), exist_ok=True)
os.makedirs(os.path.join(app.root_path, 'static', 'uploads'), exist_ok=True)
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    conn = sqlite3.connect("inventario.db")
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM productos")
    data = cursor.fetchall()
    conn.close()
    image_path = os.path.join(app.root_path, 'static', 'detection.png')
    has_image = os.path.exists(image_path)
    image_url = url_for('static', filename='detection.png') if has_image else None
    return render_template('index.html', productos=data, image_ready=has_image, image_url=image_url)



@app.route('/agregar', methods=['POST'])
def agregar():
    nombre = request.form['nombre']
    diametro = request.form['diametro']
    longitud = request.form['longitud']
    cantidad = request.form['cantidad']
    precio = request.form['precio']

    conn = sqlite3.connect("inventario.db")
    cursor = conn.cursor()
    sql = (
        "INSERT INTO productos (nombre, diametro, longitud, cantidad, precio) "
        "VALUES (?,?,?,?,?)"
    )
    cursor.execute(sql, (nombre, diametro, longitud, cantidad, precio))
    conn.commit()
    conn.close()
    return redirect(url_for('index'))
    
@app.route('/escanear/<int:id_producto>')
def escanear(id_producto):
    image_path = os.path.join(app.root_path, 'static', 'detection.png')
    cantidad_detectada = contar_tuberias(output_path=image_path)
    
    conn = sqlite3.connect("inventario.db")
    cursor = conn.cursor()
    cursor.execute("UPDATE productos SET cantidad = ? WHERE id = ?", (cantidad_detectada, id_producto))
    conn.commit()
    conn.close()
    
    return redirect(url_for('index'))


@app.route('/subir-imagen', methods=['POST'])
def subir_imagen():
    if 'imagen' not in request.files:
        return redirect(url_for('index'))

    imagen = request.files['imagen']
    if imagen.filename == '' or not allowed_file(imagen.filename):
        return redirect(url_for('index'))

    filename = secure_filename(imagen.filename)
    upload_path = os.path.join(app.root_path, 'static', 'uploads', filename)
    imagen.save(upload_path)

    output_path = os.path.join(app.root_path, 'static', 'detection.png')
    procesar_imagen(input_path=upload_path, output_path=output_path)
    return redirect(url_for('index'))


def exportar_a_excel():
    conn = sqlite3.connect("inventario.db")

    df = pd.read_sql_query("SELECT * FROM productos", conn)
    conn.close()

    df.to_excel("Inventario_Final.xlsx", index=False)
    print("¡Excel actualizado!")
    
    
@app.route('/eliminar/<int:id_producto>')
def eliminar(id_producto):
    conn = sqlite3.connect("inventario.db")
    cursor = conn.cursor()
    cursor.execute("DELETE FROM productos WHERE id = ?", (id_producto,))
    conn.commit()
    conn.close()
    exportar_a_excel() # Actualizamos el Excel al borrar
    return redirect(url_for('index'))

@app.route('/download-image')
def download_image():
    image_path = os.path.join(app.root_path, 'static', 'detection.png')
    if os.path.exists(image_path):
        return send_file(image_path, as_attachment=True, download_name='detection.png')
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
    
#10.0.1.115
