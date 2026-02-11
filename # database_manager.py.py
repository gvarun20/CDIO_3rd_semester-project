# database_manager.py
import sqlite3
from datetime import datetime
import json
from typing import List, Dict, Any

class DatabaseManager:
    def __init__(self, db_path="music_transcription.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database with enhanced schema"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Audio files table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS audio_files (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                file_path TEXT NOT NULL,
                file_type TEXT NOT NULL,
                duration REAL,
                sample_rate INTEGER,
                max_simultaneous_notes INTEGER DEFAULT 1,
                upload_time DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Transcription sessions
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS transcription_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                audio_file_id INTEGER,
                detection_method TEXT,
                confidence_threshold REAL,
                total_events INTEGER,
                total_notes INTEGER,
                processing_time REAL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (audio_file_id) REFERENCES audio_files (id)
            )
        ''')
        
        # Enhanced note events with confidence
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS note_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER,
                onset_time REAL,
                offset_time REAL,
                duration REAL,
                note_count INTEGER,
                confidence_mean REAL,
                FOREIGN KEY (session_id) REFERENCES transcription_sessions (id)
            )
        ''')
        
        # Individual notes
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS individual_notes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id INTEGER,
                pitch REAL,
                note_name TEXT,
                confidence REAL,
                detection_method TEXT,
                note_order INTEGER,
                FOREIGN KEY (event_id) REFERENCES note_events (id)
            )
        ''')
        
        # Confidence statistics
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS confidence_stats (
                session_id INTEGER,
                mean_confidence REAL,
                median_confidence REAL,
                std_confidence REAL,
                high_confidence_count INTEGER,
                low_confidence_count INTEGER,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES transcription_sessions (id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def save_transcription_session(self, audio_file_id: int, 
                                 transcription_results: List[Dict],
                                 settings: Dict) -> int:
        """Save complete transcription session"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Calculate statistics
        total_notes = sum(event['note_count'] for event in transcription_results)
        all_confidences = []
        for event in transcription_results:
            for note in event['notes']:
                all_confidences.append(note.get('confidence', 0))
        
        # Save session
        cursor.execute('''
            INSERT INTO transcription_sessions 
            (audio_file_id, detection_method, confidence_threshold,
             total_events, total_notes, processing_time)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            audio_file_id,
            settings.get('detection_method', 'mixed'),
            settings.get('confidence_threshold', 0.3),
            len(transcription_results),
            total_notes,
            settings.get('processing_time', 0)
        ))
        
        session_id = cursor.lastrowid
        
        # Save note events and individual notes
        for event in transcription_results:
            # Calculate event confidence mean
            event_confidences = [n.get('confidence', 0) for n in event['notes']]
            confidence_mean = np.mean(event_confidences) if event_confidences else 0
            
            cursor.execute('''
                INSERT INTO note_events 
                (session_id, onset_time, offset_time, duration, 
                 note_count, confidence_mean)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                session_id,
                event['onset'],
                event['offset'],
                event['duration'],
                event['note_count'],
                confidence_mean
            ))
            
            event_id = cursor.lastrowid
            
            # Save individual notes
            for i, note in enumerate(event['notes']):
                cursor.execute('''
                    INSERT INTO individual_notes 
                    (event_id, pitch, note_name, confidence, 
                     detection_method, note_order)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    event_id,
                    note['pitch'],
                    note['note'],
                    note.get('confidence', 0),
                    note.get('detection_method', 'unknown'),
                    i
                ))
        
        # Save confidence statistics
        if all_confidences:
            cursor.execute('''
                INSERT INTO confidence_stats 
                (session_id, mean_confidence, median_confidence,
                 std_confidence, high_confidence_count, low_confidence_count)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                session_id,
                np.mean(all_confidences),
                np.median(all_confidences),
                np.std(all_confidences),
                sum(1 for c in all_confidences if c > 0.7),
                sum(1 for c in all_confidences if c < 0.3)
            ))
        
        conn.commit()
        conn.close()
        
        return session_id
    
    def get_session_history(self, limit: int = 10) -> List[Dict]:
        """Get transcription session history"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT 
                s.id, a.filename, s.timestamp, 
                s.total_events, s.total_notes,
                s.confidence_threshold, s.detection_method
            FROM transcription_sessions s
            JOIN audio_files a ON s.audio_file_id = a.id
            ORDER BY s.timestamp DESC
            LIMIT ?
        ''', (limit,))
        
        sessions = []
        for row in cursor.fetchall():
            sessions.append({
                'id': row[0],
                'filename': row[1],
                'timestamp': row[2],
                'total_events': row[3],
                'total_notes': row[4],
                'confidence_threshold': row[5],
                'detection_method': row[6]
            })
        
        conn.close()
        return sessions
    
    def get_session_details(self, session_id: int) -> Dict:
        """Get detailed session information"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get session info
        cursor.execute('''
            SELECT * FROM transcription_sessions 
            WHERE id = ?
        ''', (session_id,))
        
        session_row = cursor.fetchone()
        if not session_row:
            return {}
        
        # Get note events
        cursor.execute('''
            SELECT * FROM note_events 
            WHERE session_id = ?
            ORDER BY onset_time
        ''', (session_id,))
        
        events = []
        for event_row in cursor.fetchall():
            event_id = event_row[0]
            
            # Get individual notes for this event
            cursor.execute('''
                SELECT * FROM individual_notes 
                WHERE event_id = ?
                ORDER BY note_order
            ''', (event_id,))
            
            notes = []
            for note_row in cursor.fetchall():
                notes.append({
                    'pitch': note_row[2],
                    'note_name': note_row[3],
                    'confidence': note_row[4],
                    'detection_method': note_row[5]
                })
            
            events.append({
                'onset': event_row[2],
                'offset': event_row[3],
                'duration': event_row[4],
                'note_count': event_row[5],
                'confidence_mean': event_row[6],
                'notes': notes
            })
        
        conn.close()
        
        return {
            'session_id': session_row[0],
            'audio_file_id': session_row[1],
            'detection_method': session_row[2],
            'confidence_threshold': session_row[3],
            'total_events': session_row[4],
            'total_notes': session_row[5],
            'processing_time': session_row[6],
            'timestamp': session_row[7],
            'events': events
        }