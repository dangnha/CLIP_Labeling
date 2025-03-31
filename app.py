from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for, flash
from models.blip_captioner import BLIPCaptioner
from models.clip_manager import CLIPManager
from utils.database import ImageDatabase
from utils.file_handler import save_uploaded_file, delete_file
from utils.visualization import create_classification_chart
import os
import pandas as pd
from io import StringIO
from config import UPLOAD_FOLDER, ALLOWED_EXTENSIONS, MAX_FILE_SIZE, DEFAULT_CATEGORIES
from flask import g
import json
from werkzeug.utils import secure_filename
from flask_toastr import Toastr
from datetime import datetime

app = Flask(__name__)
app.config.from_pyfile('config.py')

# Initialize components
captioner = BLIPCaptioner()
clip_manager = CLIPManager()

# Initialize Toastr
toastr = Toastr()
toastr.init_app(app)

def get_db():
    if 'db' not in g:
        g.db = ImageDatabase()
    return g.db

@app.teardown_appcontext
def close_db(exception):
    db = g.pop('db', None)
    if db is not None:
        # Assuming the ImageDatabase class does not have a close method, we remove the call to db.close()
        pass

@app.route('/')
def index():
    db = get_db()
    stats = db.get_system_stats()
    recent = db.get_recent_images(5)
    
    # Convert labels from JSON string back to dict if they are not already a dict
    for image in recent:
        if isinstance(image['labels'], str):
            image['labels'] = json.loads(image['labels'])
    
    return render_template('index.html', stats=stats, recent_images=recent)

def process_and_save_image(file):
    """Process image through caption and classification pipeline"""
    filepath = None
    try:
        # Save the uploaded file
        filepath, filename = save_uploaded_file(file, UPLOAD_FOLDER)
        
        # Convert Windows path to forward slashes and make it relative to static
        relative_path = f'/static/uploads/{filename}'
        
        print(f"File saved as: {relative_path}")  # Debug print
        
        # Generate caption
        caption = captioner.generate_caption(filepath)
        print(f"Generated caption: {caption}")
        
        # Classify image with default categories
        classification_results = clip_manager.classify_image(filepath, DEFAULT_CATEGORIES)
        print(f"Classification results: {classification_results}")
        
        # Filter labels with confidence above threshold
        threshold = 0.2
        labels = {
            category: score 
            for category, score in classification_results.items() 
            if score > threshold
        }
        
        if not labels:
            labels = {"unclassified": 1.0}
        
        # Store in database with relative path
        db = get_db()
        image_id = db.add_image(relative_path, caption=caption, labels=labels)
        
        if not image_id:
            raise Exception("Failed to save to database")
            
        return {
            'success': True,
            'image_id': image_id,
            'filename': filename,
            'caption': caption,
            'labels': labels,
            'path': relative_path
        }
        
    except Exception as e:
        print(f"Error in process_and_save_image: {str(e)}")
        if filepath and os.path.exists(filepath):
            os.remove(filepath)
        raise

@app.route('/caption', methods=['GET', 'POST'])
def caption():
    if request.method == 'POST':
        if 'image' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No image uploaded'
            }), 400
            
        file = request.files['image']
        if file.filename == '':
            return jsonify({
                'success': False,
                'error': 'No selected file'
            }), 400
            
        try:
            result = process_and_save_image(file)
            return jsonify(result)
            
        except Exception as e:
            print(f"Error in caption endpoint: {str(e)}")  # Debug print
            return jsonify({
                'success': False,
                'error': str(e)
            }), 400
            
    return render_template('caption.html')

# Classification Endpoints
@app.route('/classify', methods=['GET', 'POST'])
def classify():
    db = get_db()
    if request.method == 'POST':
        file = request.files['image']
        categories = [c.strip() for c in request.form.get('categories', '').split(',') if c.strip()]
        
        try:
            filepath, _ = save_uploaded_file(file, UPLOAD_FOLDER)
            results = clip_manager.classify_image(filepath, categories)
            chart_html = create_classification_chart(results)
            return jsonify({
                'success': True,
                'results': results,
                'chart': chart_html,
                'image_url': f'/static/uploads/{os.path.basename(filepath)}'
            })
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 400
    return render_template('classify.html')

# Search Endpoints
@app.route('/search/image', methods=['POST'])
def search_by_image():
    db = get_db()
    if 'image' not in request.files:
        return jsonify({'success': False, 'error': 'No image uploaded'}), 400
    
    file = request.files['image']
    try:
        filepath, _ = save_uploaded_file(file, UPLOAD_FOLDER)
        results = clip_manager.search_by_image(filepath, db.get_all_images())
        
        # Convert labels from string to dict for each result if they are not already a dict
        for result in results:
            if isinstance(result.get('labels'), str):
                result['labels'] = json.loads(result['labels'])
        
        return jsonify({'success': True, 'results': results})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/search/text', methods=['POST'])
def search_by_text():
    db = get_db()
    query = request.form.get('query', '').strip()
    if not query:
        return jsonify({'success': False, 'error': 'No search query provided'}), 400
    
    try:
        print(f"Received query: {query}")  # Debug logging
        all_images = db.get_all_images()
        print(f"Number of images to search: {len(all_images)}")  # Debug logging
        
        results = clip_manager.search_by_text(query, all_images)
        print(f"Found {len(results)} results")  # Debug logging
        
        for result in results:
            print(f"Processing result: {result.get('id')}")  # Debug logging
            if isinstance(result.get('labels'), str):
                try:
                    result['labels'] = json.loads(result['labels'])
                except json.JSONDecodeError as e:
                    print(f"Error parsing labels for image {result.get('id')}: {e}")
                    result['labels'] = {}  # Provide default empty dict
        
        return jsonify({'success': True, 'results': results})
    except Exception as e:
        print(f"Error in search_by_text: {str(e)}")  # Debug logging
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/search', methods=['GET'])
def search():
    return render_template('search.html')

# Database Management Endpoints
@app.route('/manage', methods=['GET'])
def manage():
    db = get_db()
    images = db.get_all_images()
    return render_template('manage.html', images=images)

@app.route('/export/csv')
def export_csv():
    db = get_db()
    images = db.get_all_images()
    df = pd.DataFrame(images)
    output = StringIO()
    df.to_csv(output, index=False)
    output.seek(0)
    return send_file(
        output,
        mimetype='text/csv',
        as_attachment=True,
        download_name='image_database.csv'
    )

@app.route('/delete/<int:image_id>', methods=['DELETE'])
def delete_image(image_id):
    db = get_db()
    try:
        image = db.get_image(image_id)
        if not image:
            return jsonify({'success': False, 'error': 'Image not found'}), 404
        
        # Get the file path and remove the static prefix for proper file deletion
        file_path = image['path']
        if file_path.startswith('/static/'):
            file_path = file_path[7:]  # Remove '/static/' prefix
        full_path = os.path.join(os.getcwd(), 'static', file_path)
        
        # Delete the file first
        try:
            os.remove(full_path)
        except OSError as e:
            print(f"Error deleting file: {e}")
            # Continue even if file doesn't exist
            
        # Delete from database
        db.delete_image(image_id)
        return jsonify({'success': True})
    except Exception as e:
        print(f"Error in delete_image: {str(e)}")  # Add logging
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/save-caption', methods=['POST'])
def save_caption():
    db = get_db()
    try:
        # Check if the request has JSON content
        if not request.is_json:
            print("Request Content-Type:", request.headers.get('Content-Type'))
            print("Request data:", request.get_data())
            return jsonify({
                'success': False,
                'error': 'Content-Type must be application/json'
            }), 415

        data = request.get_json(force=True)  # Try to force JSON parsing
        print("Received data:", data)  # Debug print
        
        image_id = data.get('image_id')
        new_caption = data.get('caption')

        if image_id is None or new_caption is None:
            return jsonify({
                'success': False,
                'error': 'Missing image_id or caption'
            }), 400

        # Convert image_id to int if it's a string
        try:
            image_id = int(image_id)
        except (TypeError, ValueError):
            return jsonify({
                'success': False,
                'error': f'Invalid image_id format: {image_id}'
            }), 400

        # Update the caption
        if db.update_caption(image_id, new_caption):
            return jsonify({
                'success': True,
                'message': 'Caption updated successfully',
                'caption': new_caption
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Failed to update caption in database'
            }), 400

    except Exception as e:
        print(f"Error in save_caption: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@app.route('/debug/database')
def debug_database():
    db = get_db()
    images = db.get_all_images()
    return jsonify({
        'image_count': len(images),
        'images': images
    })

if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True, port=5001)
