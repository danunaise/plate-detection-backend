import os
from datetime import datetime
import psycopg2
from flask import Flask, jsonify, send_from_directory, request
from flask_cors import CORS
from flask_socketio import SocketIO

from connection import get_db_connection

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})
socketio = SocketIO(app, cors_allowed_origins="*")

# Directory to serve images from
IMAGE_DIR = '../images'

def insert_plate(f_image, p_image, p_text, province, date=None):
    conn = get_db_connection()
    if conn is None:
        return False

    cursor = conn.cursor()
    try:
        cursor.execute(
            "INSERT INTO plates (f_image, p_image, p_text, province, date) VALUES (%s, %s, %s, %s, NOW());",
            (f_image, p_image, p_text, province)
        )
        conn.commit()
        print('Emitting new_plate event with data:', {
            "f_image": f_image,
            "p_image": p_image,
            "p_text": p_text,
            "province": province,
            "date": date
        })
        socketio.emit('new_plate', {
            "id": cursor.lastrowid,  # Ensure you have the ID of the newly inserted plate
            "f_image": f_image,
            "p_image": p_image,
            "p_text": p_text,
            "province": province,
            "date": date
        })
        cursor.close()
        conn.close()
        return True
    except psycopg2.Error as e:
        print(f"Error inserting data: {e}")
        cursor.close()
        conn.close()
        return False

@app.route('/test_emit', methods=['POST'])
def test_emit():
    data = request.json
    print('Emitting test data:', data)  # Log the data to see what's being emitted
    socketio.emit('test_event', data)
    return jsonify({'status': 'Event emitted'})

@app.route('/api/plates', methods=['GET'])
def get_plates():
    conn = get_db_connection()
    if conn is None:
        return jsonify({"error": "Database connection error"}), 500

    cursor = conn.cursor()
    try:
        cursor.execute('SELECT id, f_image, p_image, p_text, province, date FROM "plateDetection";')
        rows = cursor.fetchall()
        plates = []
        for row in rows:
            plates.append({
                "id": row[0],
                "f_image": row[1],
                "p_image": row[2],
                "p_text": row[3],
                "province": row[4],
                # Convert the 'date' field to a string before returning
                "date": row[5].isoformat() if isinstance(row[5], datetime) else str(row[5])
            })
        cursor.close()
        conn.close()
        return jsonify(plates)
    except psycopg2.Error as e:
        print(f"Error fetching data: {e}")
        cursor.close()
        conn.close()
        return jsonify({"error": "Error fetching data"}), 500

@app.route('/images/<path:filename>')
def serve_image(filename):
    # Remove the "images/" prefix from the filename
    return send_from_directory(IMAGE_DIR, filename)

# search
@app.route('/api/search', methods=['GET'])
def search_plates():
    search_term = request.args.get('q', '').strip()  # รับพารามิเตอร์การค้นหา 'q'
    if not search_term:
        return jsonify({"error": "Search term is required"}), 400

    conn = get_db_connection()
    if conn is None:
        return jsonify({"error": "Database connection error"}), 500

    cursor = conn.cursor()
    try:
        # ค้นหาข้อมูลที่มีค่า `p_text` หรือ `province` ที่ตรงกับคำค้นหา
        search_query = """
            SELECT id, f_image, p_image, p_text, province, date
            FROM plates
            WHERE p_text ILIKE %s OR province ILIKE %s;
        """
        cursor.execute(search_query, (f"%{search_term}%", f"%{search_term}%"))
        rows = cursor.fetchall()

        plates = []
        for row in rows:
            plates.append({
                "id": row[0],
                "f_image": f"{row[1]}",
                "p_image": f"{row[2]}",
                "p_text": row[3],
                "province": row[4],
                "date": row[5]
            })
        cursor.close()
        conn.close()
        return jsonify(plates)
    except psycopg2.Error as e:
        print(f"Error fetching data: {e}")
        cursor.close()
        conn.close()
        return jsonify({"error": "Error fetching data"}), 500

@socketio.on('connect')
def handle_connect():
    print("Client connected")

@socketio.on('disconnect')
def handle_disconnect():
    print("Client disconnected")

def emit_new_plate(data):
    socketio.emit('new_plate', data)

if __name__ == '__main__':
    socketio.run(app, debug=True, allow_unsafe_werkzeug=True)
