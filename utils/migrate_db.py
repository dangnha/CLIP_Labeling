import sqlite3
import os

def migrate_database():
    # Connect to the database
    conn = sqlite3.connect('images.db')
    cursor = conn.cursor()
    
    try:
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
            
            # Copy data from old table to new table
            cursor.execute('''
                INSERT INTO images_new (id, path, display_name, caption, labels, created_at)
                SELECT id, path, path, caption, labels, created_at FROM images
            ''')
            
            # Drop old table
            cursor.execute('DROP TABLE images')
            
            # Rename new table to original name
            cursor.execute('ALTER TABLE images_new RENAME TO images')
            
            # Update display_names to just the filename
            cursor.execute('SELECT id, path FROM images')
            for row in cursor.fetchall():
                id, path = row
                display_name = os.path.basename(path)
                cursor.execute('UPDATE images SET display_name = ? WHERE id = ?', (display_name, id))
            
            conn.commit()
            print("Migration completed successfully!")
        else:
            print("Database schema is already up to date!")
            
    except Exception as e:
        print(f"Error during migration: {str(e)}")
        conn.rollback()
    finally:
        conn.close()

if __name__ == "__main__":
    migrate_database() 