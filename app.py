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
        
        # Take top k = 1 and check if it exceeds the threshold
        threshold = 0.5
        top_category, top_score = max(classification_results.items(), key=lambda item: item[1])
        
        if top_score > threshold:
            labels = {top_category: top_score}
        else:
            labels = {"others": 1.0}
        
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
            # Save the uploaded file temporarily
            filepath, filename = save_uploaded_file(file, UPLOAD_FOLDER)
            relative_path = f'/static/uploads/{filename}'
            
            # Generate caption only
            caption = captioner.generate_caption(filepath)
            
            # Get CLIP similarity score for the generated caption
            similarity_score = clip_manager.get_similarity(filepath, caption)
            
            return jsonify({
                'success': True,
                'filename': filename,
                'caption': caption,
                'path': relative_path,
                'similarity_score': float(similarity_score),
                'temp_filepath': filepath
            })
            
        except Exception as e:
            print(f"Error in caption endpoint: {str(e)}")
            if 'filepath' in locals() and os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({
                'success': False,
                'error': str(e)
            }), 400
            
    return render_template('caption.html')

@app.route('/save-to-database', methods=['POST'])
def save_to_database():
    try:
        data = request.get_json()
        filepath = data.get('filepath')
        filename = data.get('filename')
        caption = data.get('caption')
        
        if not all([filepath, filename, caption]):
            return jsonify({
                'success': False,
                'error': 'Missing required fields'
            }), 400
        
        # Classify image with default categories
        classification_results = clip_manager.classify_image(filepath, DEFAULT_CATEGORIES)
        print(f"Classification results: {classification_results}")
        
        # Sort results by score and take top results above threshold
        threshold = 0.5
        sorted_results = {k: v for k, v in classification_results.items() if v > threshold}
        sorted_results = dict(sorted(sorted_results.items(), key=lambda x: x[1], reverse=True))
        
        # If no categories meet the threshold, take the highest scoring one
        if not sorted_results:
            top_category, top_score = max(classification_results.items(), key=lambda x: x[1])
            sorted_results = {top_category: top_score}
        
        # Store in database
        relative_path = f'/static/uploads/{filename}'
        db = get_db()
        image_id = db.add_image(relative_path, caption=caption, labels=sorted_results)
        
        if not image_id:
            raise Exception("Failed to save to database")
            
        return jsonify({
            'success': True,
            'image_id': image_id,
            'labels': sorted_results,
            'path': relative_path
        })
        
    except Exception as e:
        print(f"Error in save_to_database: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

# Classification Endpoints
@app.route('/classify', methods=['GET', 'POST'])
def classify():
    db = get_db()
    if request.method == 'POST':
        if 'image' not in request.files:
            return jsonify({'success': False, 'error': 'No image uploaded'}), 400
            
        file = request.files['image']
        categories = [c.strip() for c in request.form.get('categories', '').split(',') if c.strip()]
        
        try:
            filepath, _ = save_uploaded_file(file, UPLOAD_FOLDER)
            
            # Use provided categories or default ones
            categories_to_use = categories if categories else DEFAULT_CATEGORIES
            
            # Get classification results
            results = clip_manager.classify_image(filepath, categories_to_use)
            
            # Sort all results by score (no threshold)
            sorted_results = dict(sorted(results.items(), key=lambda x: x[1], reverse=True))
            
            # Create visualization for all results
            chart_html = create_classification_chart(sorted_results)
            
            # Generate explanation text
            top_category, top_score = next(iter(sorted_results.items()))
            explanation = (
                f"The bar chart above shows the classification scores for each category. "
                f"The top prediction is '<b>{top_category}</b>' with a confidence of <b>{top_score*100:.1f}%</b>. "
                f"Other categories are shown in descending order of confidence."
            )
            
            return jsonify({
                'success': True,
                'results': sorted_results,
                'chart': chart_html,
                'image_url': f'/static/uploads/{os.path.basename(filepath)}',
                'explanation': explanation
            })
            
        except Exception as e:
            if 'filepath' in locals() and os.path.exists(filepath):
                os.remove(filepath)
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
        print(f"Processing uploaded file: {file.filename}")
        
        filepath, _ = save_uploaded_file(file, UPLOAD_FOLDER)
        print(f"File saved at: {filepath}")
        
        all_images = db.get_all_images()
        print(f"Retrieved {len(all_images)} images from database")
        
        # Convert database image paths to full filesystem paths
        for image in all_images:
            if image['path'].startswith('/static/'):
                # Remove '/static/' and convert to full path
                relative_path = image['path'].replace('/static/', '')
                image['full_path'] = os.path.join(os.getcwd(), 'static', relative_path)
                print(f"Converting path: {image['path']} -> {image['full_path']}")
        
        # Use full_path for CLIP processing but keep original path for frontend
        results = clip_manager.search_by_image(filepath, [
            {**img, 'path': img['full_path']} for img in all_images
        ])
        print(f"Search completed, found {len(results)} results")
        
        # Format results for frontend
        formatted_results = []
        for result in results:
            try:
                # Keep the web-friendly path for frontend
                original_image = next(img for img in all_images if img['id'] == result['id'])
                formatted_result = {
                    'id': result['id'],
                    'path': original_image['path'],  # Use original web path
                    'caption': result['caption'],
                    'similarity_score': float(result['similarity']),
                    'labels': result['labels'] if isinstance(result['labels'], dict) 
                             else json.loads(result['labels'])
                }
                formatted_results.append(formatted_result)
            except Exception as e:
                print(f"Error formatting result: {e}")
                continue
        
        # Sort by similarity score
        formatted_results.sort(key=lambda x: x['similarity_score'], reverse=True)
        
        # Clean up uploaded file
        try:
            os.remove(filepath)
        except OSError:
            pass
            
        return jsonify({
            'success': True,
            'results': formatted_results,
            'total_results': len(formatted_results)
        })
        
    except Exception as e:
        print(f"Error in search_by_image: {str(e)}")
        if 'filepath' in locals():
            try:
                os.remove(filepath)
            except OSError:
                pass
        return jsonify({'success': False, 'error': f'Error performing search: {str(e)}'}), 400

@app.route('/search/text', methods=['POST'])
def search_by_text():
    db = get_db()
    query = request.form.get('query', '').strip()
    if not query:
        return jsonify({'success': False, 'error': 'No search query provided'}), 400
    
    try:
        print(f"Received query: {query}")
        all_images = db.get_all_images()
        print(f"Number of images to search: {len(all_images)}")
        
        # Convert paths for CLIP processing
        for image in all_images:
            if image['path'].startswith('/static/'):
                # Remove '/static/' and convert to full path
                relative_path = image['path'].replace('/static/', '')
                image['full_path'] = os.path.join(os.getcwd(), 'static', relative_path)
                print(f"Converting path: {image['path']} -> {image['full_path']}")
        
        # Use full_path for CLIP processing but keep original path for frontend
        results = clip_manager.search_by_text(query, [
            {**img, 'path': img['full_path']} for img in all_images
        ])
        print(f"Found {len(results)} results")
        
        # Format results for frontend
        formatted_results = []
        for result in results:
            try:
                # Keep the web-friendly path for frontend
                original_image = next(img for img in all_images if img['id'] == result['id'])
                formatted_result = {
                    'id': result['id'],
                    'path': original_image['path'],  # Use original web path
                    'caption': result['caption'],
                    'similarity_score': float(result['similarity']),
                    'labels': result['labels'] if isinstance(result['labels'], dict) 
                             else json.loads(result['labels'])
                }
                formatted_results.append(formatted_result)
            except Exception as e:
                print(f"Error formatting result: {e}")
                continue
        
        # Sort by similarity score
        formatted_results.sort(key=lambda x: x['similarity_score'], reverse=True)
        
        return jsonify({
            'success': True,
            'results': formatted_results,
            'query': query,
            'total_results': len(formatted_results)
        })
    except Exception as e:
        print(f"Error in search_by_text: {str(e)}")
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

import os
import torch
import clip
from PIL import Image
from flask import request, jsonify
from models.clip_manager import CLIPManager  # Assuming this wraps CLIP functionality
from models.blip_captioner import BLIPCaptioner

@app.route('/verify-caption', methods=['POST'])
def verify_caption():
    try:
        # Load CLIP model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        clip_model, preprocess = clip.load("ViT-B/32", device=device)
        
        # Parse incoming JSON data
        data = request.get_json()
        image_path = data.get('image_path')
        original_caption = data.get('original_caption')
        new_caption = data.get('new_caption')

        if image_path.startswith('/static/'):
            image_path = os.path.join(os.getcwd(), image_path.lstrip('/'))

        print(f"Received data: {data}")
        print(f"Image Path: {image_path}")
        print(f"Original caption: {original_caption}")
        print(f"New caption: {new_caption}")

        # Validate inputs
        if not image_path or not os.path.exists(image_path):
            return jsonify({'success': False, 'error': 'Invalid or missing image_path'}), 400
        if not original_caption:
            return jsonify({'success': False, 'error': 'Missing original_caption'}), 400
        if not new_caption:
            return jsonify({'success': False, 'error': 'Missing new_caption'}), 400

        # Preprocess image
        image = Image.open(image_path).convert("RGB")
        image_input = preprocess(image).unsqueeze(0).to(device)

        # Preprocess captions
        text_inputs = clip.tokenize([original_caption, new_caption]).to(device)

        # Compute embeddings
        with torch.no_grad():
            image_features = clip_model.encode_image(image_input)
            text_features = clip_model.encode_text(text_inputs)

            # Normalize embeddings
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # Calculate cosine similarities
        similarities = (image_features @ text_features.T).squeeze(0)
        old_similarity = similarities[0].item()
        new_similarity = similarities[1].item()

        print(f"Original caption: {original_caption}, similarity: {old_similarity}")
        print(f"New caption: {new_caption}, similarity: {new_similarity}")

        # Optionally use BLIP to validate captions (if BLIPCaptioner is intended for this)
        blip_captioner = BLIPCaptioner()  # Initialize BLIP model
        blip_caption = blip_captioner.generate_caption(image_path)
        print(f"BLIP-generated caption: {blip_caption}")

        # Determine if new caption is better
        is_better = new_similarity > old_similarity

        return jsonify({
            'success': True,
            'is_better': is_better,
            'old_similarity': old_similarity,
            'new_similarity': new_similarity,
            'old_caption': original_caption,
            'new_caption': new_caption,
            'blip_caption': blip_caption,  # Optional: for reference
            'allow_update': is_better
        })

    except Exception as e:
        print(f"Error in verify_caption: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

# @app.route('/verify-caption', methods=['POST'])
# def verify_caption():
#     try:
#         data = request.get_json()
#         image_path = data.get('image_path')
#         original_caption = data.get('original_caption')
#         new_caption = data.get('new_caption')

#         print(f"Received data: {data}")
#         print(f"Image Path: {image_path}")
#         print(f"Original caption: {original_caption}")
#         print(f"New caption: {new_caption}")
        
#         # Ensure all required fields are provided
#         if not image_path:
#             return jsonify({
#                 'success': False,
#                 'error': 'Missing image_path'
#             }), 400
#         if original_caption is None:
#             return jsonify({
#                 'success': False,
#                 'error': 'Missing original_caption'
#             }), 400
#         if new_caption is None:
#             return jsonify({
#                 'success': False,
#                 'error': 'Missing new_caption'
#             }), 400

#         # Correct the image path to ensure it is an absolute path
#         if image_path.startswith('/static/'):
#             image_path = os.path.join(os.getcwd(), image_path.lstrip('/'))

#         # Calculate similarity scores using embeddings
#         image_embedding = clip_manager.get_image_embedding(image_path)
#         original_caption_embedding = clip_manager.get_text_embedding(original_caption)
#         new_caption_embedding = clip_manager.get_text_embedding(new_caption)

#         old_similarity = clip_manager.calculate_similarity(image_embedding, original_caption_embedding)
#         new_similarity = clip_manager.calculate_similarity(image_embedding, new_caption_embedding)
        
#         print(f"Original caption: {original_caption}, similarity: {old_similarity}")
#         print(f"New caption: {new_caption}, similarity: {new_similarity}")

#         is_better = new_similarity > old_similarity
        
#         return jsonify({
#             'success': True,
#             'is_better': is_better,
#             'old_similarity': float(old_similarity),
#             'new_similarity': float(new_similarity),
#             'old_caption': original_caption,
#             'allow_update': is_better
#         })

#     except Exception as e:
#         print(f"Error in verify_caption: {str(e)}")
#         return jsonify({
#             'success': False,
#             'error': str(e)
#         }), 400

# @app.route('/save-caption', methods=['POST'])
# def save_caption():
#     db = get_db()
#     try:
#         data = request.get_json()
#         image_id = data.get('image_id')
#         new_caption = data.get('caption')
#         force_update = data.get('force_update', False)  # Add force update option
        
#         if not image_id or new_caption is None:
#             return jsonify({
#                 'success': False,
#                 'error': 'Missing image_id or caption'
#             }), 400

#         # If not forcing update, verify the caption
#         if not force_update:
#             image = db.get_image(image_id)
#             image_path = os.path.join(os.getcwd(), image['path'].lstrip('/'))
#             old_caption = image['caption']
            
#             old_similarity = clip_manager.get_similarity(image_path, old_caption) if old_caption else 0
#             new_similarity = clip_manager.get_similarity(image_path, new_caption)
            
#             if new_similarity <= old_similarity:
#                 return jsonify({
#                     'success': False,
#                     'error': 'New caption is not better than the current one',
#                     'old_similarity': float(old_similarity),
#                     'new_similarity': float(new_similarity),
#                     'old_caption': old_caption
#                 }), 400

#         # Update the caption
#         if db.update_caption(image_id, new_caption):
#             return jsonify({
#                 'success': True,
#                 'message': 'Caption updated successfully',
#                 'caption': new_caption
#             })
#         else:
#             return jsonify({
#                 'success': False,
#                 'error': 'Failed to update caption in database'
#             }), 400

#     except Exception as e:
#         print(f"Error in save_caption: {str(e)}")
#         return jsonify({
#             'success': False,
#             'error': str(e)
#         }), 400

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
