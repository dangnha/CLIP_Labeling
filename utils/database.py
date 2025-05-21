import sqlite3
from sqlite3 import Error
from datetime import datetime
from config import DATABASE_PATH
import json
import os

class ImageDatabase:
    def __init__(self):
        self.conn = sqlite3.connect('images.db', detect_types=sqlite3.PARSE_DECLTYPES)
        self.conn.row_factory = sqlite3.Row
        self.migrate_database()  # Replace recreate_tables() with migrate_database()

    def migrate_database(self):
        cursor = self.conn.cursor()
        try:
            # Check if table exists
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='images'")
            table_exists = cursor.fetchone() is not None

            if not table_exists:
                # Create new table if it doesn't exist
                cursor.execute('''
                    CREATE TABLE images (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        path TEXT NOT NULL,
                        display_name TEXT,
                        caption TEXT,
                        labels TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                self.conn.commit()
                print("Created new images table")
                return

            # Check if display_name column exists
            cursor.execute("PRAGMA table_info(images)")
            columns = [column[1] for column in cursor.fetchall()]
            
            if 'display_name' not in columns:
                print("Adding display_name column...")
                
                # Create temporary table with new schema
                cursor.execute('''
                    CREATE TABLE images_new (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        path TEXT NOT NULL,
                        display_name TEXT,
                        caption TEXT,
                        labels TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Copy existing data and set display_name from path
                cursor.execute('''
                    INSERT INTO images_new (id, path, display_name, caption, labels, created_at)
                    SELECT id, path, 
                           SUBSTR(path, INSTR(path, '/')+1) as display_name,
                           caption, labels, created_at 
                    FROM images
                ''')
                
                # Drop old table and rename new one
                cursor.execute('DROP TABLE images')
                cursor.execute('ALTER TABLE images_new RENAME TO images')
                
                self.conn.commit()
                print("Successfully migrated database schema")
            
        except Exception as e:
            print(f"Error in database migration: {str(e)}")
            self.conn.rollback()
        finally:
            cursor.close()

    def add_image(self, path, caption=None, labels=None):
        try:
            # Clean the path and get display name
            clean_path = path.replace('\\', '/')
            # Store path relative to static folder
            if '/static/' in clean_path:
                relative_path = clean_path[clean_path.index('/static/'):]
            else:
                relative_path = f'/static/uploads/{os.path.basename(clean_path)}'
            
            display_name = os.path.basename(clean_path)
            
            cursor = self.conn.cursor()
            cursor.execute('''
                INSERT INTO images (path, display_name, caption, labels) 
                VALUES (?, ?, ?, ?)
            ''', (
                relative_path,  # Store relative path
                display_name,
                caption,
                json.dumps(labels) if labels else None
            ))
            self.conn.commit()
            image_id = cursor.lastrowid
            cursor.close()
            print(f"Successfully added image with ID: {image_id}, path: {relative_path}")
            return image_id
        except Exception as e:
            print(f"Error adding image: {str(e)}")
            self.conn.rollback()
            return None

    def get_image(self, image_id):
        cursor = self.conn.cursor()
        cursor.execute('SELECT * FROM images WHERE id = ?', (image_id,))
        image = cursor.fetchone()
        cursor.close()
        
        if image:
            image_dict = dict(image)
            if image_dict['labels']:
                try:
                    image_dict['labels'] = json.loads(image_dict['labels'])
                except:
                    image_dict['labels'] = {}
            return image_dict
        return None

    def get_all_images(self):
        cursor = self.conn.cursor()
        try:
            # static\uploads\pink-flowers-float-clear-waters-hawaii-soft-white-sand-below-347952870_20250331130141.jpg
            cursor.execute('SELECT * FROM images ORDER BY created_at DESC')
            images = []
            for row in cursor.fetchall():
                image_dict = dict(row)
                if image_dict['labels']:
                    try:
                        image_dict['labels'] = json.loads(image_dict['labels'])
                    except:
                        image_dict['labels'] = {}
                # Ensure path is using forward slashes
                image_dict['path'] = image_dict['path'].replace('\\', '/')
                images.append(image_dict)
            return images
        except Exception as e:
            print(f"Error getting images: {str(e)}")
            return []
        finally:
            cursor.close()

    def get_recent_images(self, limit=5):
        cursor = self.conn.cursor()
        try:
            cursor.execute('SELECT * FROM images ORDER BY created_at DESC LIMIT ?', (limit,))
            images = []
            for row in cursor.fetchall():
                image_dict = dict(row)
                if image_dict['labels']:
                    try:
                        image_dict['labels'] = json.loads(image_dict['labels'])
                    except:
                        image_dict['labels'] = {}
                # Ensure path is using forward slashes
                image_dict['path'] = image_dict['path'].replace('\\', '/')
                images.append(image_dict)
            return images
        except Exception as e:
            print(f"Error getting recent images: {str(e)}")
            return []
        finally:
            cursor.close()

    def get_system_stats(self):
        cursor = self.conn.cursor()
        
        # Get total images
        cursor.execute("SELECT COUNT(*) FROM images")
        total_images = cursor.fetchone()[0]
        
        # Get labeled images (images with captions)
        cursor.execute("SELECT COUNT(*) FROM images WHERE caption IS NOT NULL")
        labeled_images = cursor.fetchone()[0]
        
        # Get last upload time
        cursor.execute("SELECT created_at FROM images ORDER BY created_at DESC LIMIT 1")
        last_upload = cursor.fetchone()
        last_upload = last_upload[0] if last_upload else None
        
        cursor.close()
        return {
            'total_images': total_images,
            'labeled_images': labeled_images,
            'last_upload': last_upload
        }

    def delete_image(self, image_id):
        try:
            cursor = self.conn.cursor()
            cursor.execute('DELETE FROM images WHERE id = ?', (image_id,))
            self.conn.commit()
            cursor.close()
            return True
        except Exception as e:
            print(f"Error deleting image: {str(e)}")
            self.conn.rollback()
            return False

    def update_caption(self, image_id, new_caption):
        try:
            cursor = self.conn.cursor()
            cursor.execute(
                'UPDATE images SET caption = ? WHERE id = ?',
                (new_caption, image_id)
            )
            self.conn.commit()
            cursor.close()
            return True
        except Exception as e:
            print(f"Error updating caption: {str(e)}")
            self.conn.rollback()
            return False

    def close(self):
        if self.conn:
            self.conn.close()

    def __del__(self):
        self.close()

    def delete_all_images(self):
        """Delete all images from the database"""
        try:
            with self.conn as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM images")
                conn.commit()
        except Exception as e:
            print(f"Error deleting all images: {str(e)}")
            raise