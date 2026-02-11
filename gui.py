# gui.py
from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QLabel, QFileDialog, QMessageBox, 
                             QGroupBox, QComboBox, QTextEdit, QTabWidget,
                             QProgressBar, QSlider, QSpinBox, QFrame, QSplitter)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QFont, QPalette, QColor
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import numpy as np
import librosa
import librosa.display
import os
from datetime import datetime
import tempfile
import webbrowser
from typing import Optional, List, Dict, Tuple
import mido

from audio_processor import (
    AudioProcessor, TranscriptionEngine, RecordingManager,
    VisualizationEngine, load_audio_file, transcribe_midi_file,
    calculate_confidence_stats
)



class AudioPlayerWidget(QWidget):
    """Custom audio player widget with play/pause/stop controls"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.initUI()
        
    def initUI(self):
        layout = QHBoxLayout(self)
        
        # Play button
        self.btn_play = QPushButton("▶ Play")
        self.btn_play.setFixedWidth(80)
        
        # Pause button
        self.btn_pause = QPushButton("⏸ Pause")
        self.btn_pause.setFixedWidth(80)
        
        # Stop button
        self.btn_stop = QPushButton("⏹ Stop")
        self.btn_stop.setFixedWidth(80)
        
        # Progress slider
        self.progress_slider = QSlider(Qt.Horizontal)
        self.progress_slider.setRange(0, 1000)
        
        # Time labels
        self.time_current = QLabel("00:00")
        self.time_total = QLabel("00:00")
        
        layout.addWidget(self.btn_play)
        layout.addWidget(self.btn_pause)
        layout.addWidget(self.btn_stop)
        layout.addWidget(self.time_current)
        layout.addWidget(self.progress_slider)
        layout.addWidget(self.time_total)
        
        self.setLayout(layout)


class AudioProcessorGUI(QMainWindow):
    """Enhanced GUI with all requested features"""
    
    # Signals for cross-thread communication
    processing_complete = pyqtSignal(list)
    
    def __init__(self):
        super().__init__()
        
        # Initialize components
        self.audio_processor = AudioProcessor(sample_rate=44100)
        self.transcription_engine = TranscriptionEngine(sample_rate=44100)
        self.recording_manager = RecordingManager()
        self.visualization_engine = VisualizationEngine()
        
        # State variables
        self.current_file_path = None
        self.audio_data = None
        self.sample_rate = 44100
        self.transcription_results = []
        self.is_playing = False
        self.playback_position = 0
        
        # Timer for real-time updates
        self.recording_timer = QTimer()
        self.recording_timer.timeout.connect(self.update_recording_display)
        
        # Initialize UI
        self.initUI()
        self.setup_connections()
        
    def initUI(self):
        """Initialize the user interface"""
        self.setWindowTitle("🎵 Advanced Music Transcription System")
        self.setGeometry(100, 100, 1400, 900)
        
        # Set application style
        self.setStyleSheet("""
            QMainWindow {
                background-color: #2b2b2b;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #555;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
                background-color: #3c3c3c;
                color: #ffffff;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
                color: #4fc3f7;
            }
            QPushButton {
                background-color: #1976d2;
                color: white;
                border: none;
                padding: 8px 15px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1565c0;
            }
            QPushButton:pressed {
                background-color: #0d47a1;
            }
            QPushButton:disabled {
                background-color: #555;
                color: #888;
            }
            QTextEdit {
                background-color: #1e1e1e;
                color: #ffffff;
                border: 1px solid #555;
                border-radius: 3px;
                font-family: 'Consolas', 'Monospace';
            }
            QLabel {
                color: #ffffff;
            }
            QComboBox {
                background-color: #3c3c3c;
                color: white;
                border: 1px solid #555;
                padding: 5px;
                border-radius: 3px;
            }
            QProgressBar {
                border: 1px solid #555;
                border-radius: 3px;
                text-align: center;
                color: white;
            }
            QProgressBar::chunk {
                background-color: #4fc3f7;
                border-radius: 3px;
            }
        """)
        
        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Create top control bar
        top_bar = self.create_top_bar()
        main_layout.addWidget(top_bar)
        
        # Create splitter for main content
        splitter = QSplitter(Qt.Horizontal)
        
        # Left panel: Controls and results
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        # File selection group
        file_group = self.create_file_selection_group()
        left_layout.addWidget(file_group)
        
        # Recording group
        recording_group = self.create_recording_group()
        left_layout.addWidget(recording_group)
        
        # Settings group
        settings_group = self.create_settings_group()
        left_layout.addWidget(settings_group)
        
        # Results display
        results_group = self.create_results_group()
        left_layout.addWidget(results_group)
        
        # Confidence stats
        confidence_group = self.create_confidence_group()
        left_layout.addWidget(confidence_group)
        
        left_layout.addStretch()
        
        # Right panel: Visualizations
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        # Visualization tabs
        self.viz_tabs = QTabWidget()
        self.viz_tabs.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #555;
                background-color: #2b2b2b;
            }
            QTabBar::tab {
                background-color: #3c3c3c;
                color: white;
                padding: 8px 15px;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                background-color: #1976d2;
            }
            QTabBar::tab:hover {
                background-color: #1565c0;
            }
        """)
        
        # Waveform tab
        waveform_tab = QWidget()
        waveform_layout = QVBoxLayout(waveform_tab)
        
        self.waveform_canvas = FigureCanvas(Figure(figsize=(10, 4)))
        waveform_layout.addWidget(self.waveform_canvas)
        waveform_layout.addWidget(NavigationToolbar(self.waveform_canvas, self))
        
        # Spectrogram tab
        spectrogram_tab = QWidget()
        spectrogram_layout = QVBoxLayout(spectrogram_tab)
        
        self.spectrogram_canvas = FigureCanvas(Figure(figsize=(10, 4)))
        spectrogram_layout.addWidget(self.spectrogram_canvas)
        spectrogram_layout.addWidget(NavigationToolbar(self.spectrogram_canvas, self))
        
        self.viz_tabs.addTab(waveform_tab, "Waveform with Notes")
        self.viz_tabs.addTab(spectrogram_tab, "Spectrogram")
        
        right_layout.addWidget(self.viz_tabs)
        
        # Add panels to splitter
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([400, 1000])
        
        main_layout.addWidget(splitter)
        
        # Status bar
        self.status_bar = self.statusBar()
        self.status_bar.showMessage("Ready")
        
        # Progress bar in status bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximumWidth(200)
        self.progress_bar.setVisible(False)
        self.status_bar.addPermanentWidget(self.progress_bar)
        
    def create_top_bar(self):
        """Create top control bar"""
        top_bar = QFrame()
        top_bar.setFrameShape(QFrame.StyledPanel)
        top_bar.setStyleSheet("background-color: #1976d2; padding: 5px;")
        
        layout = QHBoxLayout(top_bar)
        
        # Title
        title = QLabel("🎵 Advanced Music Transcription System")
        title.setStyleSheet("color: white; font-size: 18px; font-weight: bold;")
        
        # Version
        version = QLabel("v2.0")
        version.setStyleSheet("color: #bbdefb;")
        
        layout.addWidget(title)
        layout.addStretch()
        layout.addWidget(version)
        
        return top_bar
    
    def create_file_selection_group(self):
        """Create file selection group with support for multiple formats"""
        group = QGroupBox("File Selection")
        layout = QVBoxLayout(group)
        
        # File path display
        self.file_path_label = QLabel("No file selected")
        self.file_path_label.setStyleSheet("padding: 5px; background-color: #1e1e1e; border-radius: 3px;")
        self.file_path_label.setWordWrap(True)
        
        # Browse button
        btn_browse = QPushButton("📁 Browse Audio/MIDI File")
        btn_browse.clicked.connect(self.browse_file)
        
        # Quick action buttons
        action_layout = QHBoxLayout()
        btn_play_file = QPushButton("▶ Preview")
        btn_play_file.clicked.connect(self.preview_audio)
        btn_clear = QPushButton("🗑 Clear")
        btn_clear.clicked.connect(self.clear_file)
        
        action_layout.addWidget(btn_play_file)
        action_layout.addWidget(btn_clear)
        
        layout.addWidget(QLabel("Selected File:"))
        layout.addWidget(self.file_path_label)
        layout.addWidget(btn_browse)
        layout.addLayout(action_layout)
        
        return group
    
    def create_recording_group(self):
        """Create recording controls with pause/resume"""
        group = QGroupBox("Live Recording")
        layout = QVBoxLayout(group)
        
        # Recording controls
        controls_layout = QHBoxLayout()
        
        self.btn_start_recording = QPushButton("● Start Recording")
        self.btn_start_recording.setStyleSheet("background-color: #d32f2f;")
        
        self.btn_pause_recording = QPushButton("⏸ Pause")
        self.btn_pause_recording.setEnabled(False)
        
        self.btn_resume_recording = QPushButton("▶ Resume")
        self.btn_resume_recording.setEnabled(False)
        
        self.btn_stop_recording = QPushButton("⏹ Stop")
        self.btn_stop_recording.setEnabled(False)
        
        controls_layout.addWidget(self.btn_start_recording)
        controls_layout.addWidget(self.btn_pause_recording)
        controls_layout.addWidget(self.btn_resume_recording)
        controls_layout.addWidget(self.btn_stop_recording)
        
        # Recording display
        self.recording_display = QLabel("00:00")
        self.recording_display.setAlignment(Qt.AlignCenter)
        self.recording_display.setStyleSheet("""
            font-size: 24px;
            font-weight: bold;
            color: #4fc3f7;
            padding: 10px;
            background-color: #1e1e1e;
            border-radius: 5px;
        """)
        
        # Recording level meter
        self.level_meter = QProgressBar()
        self.level_meter.setTextVisible(False)
        self.level_meter.setStyleSheet("""
            QProgressBar {
                border: 1px solid #555;
                border-radius: 3px;
                height: 20px;
            }
            QProgressBar::chunk {
                background-color: #4caf50;
                border-radius: 3px;
            }
        """)
        
        layout.addLayout(controls_layout)
        layout.addWidget(self.recording_display)
        layout.addWidget(QLabel("Recording Level:"))
        layout.addWidget(self.level_meter)
        
        return group
    
    def create_settings_group(self):
        """Create settings group for transcription parameters"""
        group = QGroupBox("Transcription Settings")
        layout = QVBoxLayout(group)
        
        # Max simultaneous notes
        notes_layout = QHBoxLayout()
        notes_layout.addWidget(QLabel("Max Simultaneous Notes:"))
        
        self.note_spinbox = QSpinBox()
        self.note_spinbox.setRange(1, 4)
        self.note_spinbox.setValue(2)
        self.note_spinbox.valueChanged.connect(self.update_max_notes)
        
        notes_layout.addWidget(self.note_spinbox)
        notes_layout.addStretch()
        
        # Confidence threshold
        threshold_layout = QHBoxLayout()
        threshold_layout.addWidget(QLabel("Confidence Threshold:"))
        
        self.confidence_slider = QSlider(Qt.Horizontal)
        self.confidence_slider.setRange(10, 90)
        self.confidence_slider.setValue(30)
        self.confidence_slider.setTickInterval(10)
        self.confidence_slider.setTickPosition(QSlider.TicksBelow)
        
        self.confidence_label = QLabel("0.3")
        
        threshold_layout.addWidget(self.confidence_slider)
        threshold_layout.addWidget(self.confidence_label)
        
        # Process button
        self.btn_process = QPushButton("🎵 Transcribe Audio")
        self.btn_process.setStyleSheet("background-color: #388e3c; padding: 10px;")
        
        layout.addLayout(notes_layout)
        layout.addLayout(threshold_layout)
        layout.addWidget(self.btn_process)
        
        return group
    
    def create_results_group(self):
        """Create results display area"""
        group = QGroupBox("Transcription Results")
        layout = QVBoxLayout(group)
        
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setMaximumHeight(200)
        
        # Export buttons
        export_layout = QHBoxLayout()
        btn_export_txt = QPushButton("📄 Export as Text")
        btn_export_csv = QPushButton("📊 Export as CSV")
        btn_export_midi = QPushButton("🎹 Export as MIDI")
        
        btn_export_txt.clicked.connect(lambda: self.export_results('txt'))
        btn_export_csv.clicked.connect(lambda: self.export_results('csv'))
        btn_export_midi.clicked.connect(lambda: self.export_results('midi'))
        
        export_layout.addWidget(btn_export_txt)
        export_layout.addWidget(btn_export_csv)
        export_layout.addWidget(btn_export_midi)
        
        layout.addWidget(self.results_text)
        layout.addLayout(export_layout)
        
        return group
    
    def create_confidence_group(self):
        """Create confidence statistics display"""
        group = QGroupBox("Confidence Statistics")
        layout = QVBoxLayout(group)
        
        self.confidence_text = QTextEdit()
        self.confidence_text.setReadOnly(True)
        self.confidence_text.setMaximumHeight(150)
        
        layout.addWidget(self.confidence_text)
        
        return group
    
    def setup_connections(self):
        """Setup signal-slot connections"""
        # Recording buttons
        self.btn_start_recording.clicked.connect(self.start_recording)
        self.btn_pause_recording.clicked.connect(self.pause_recording)
        self.btn_resume_recording.clicked.connect(self.resume_recording)
        self.btn_stop_recording.clicked.connect(self.stop_recording)
        
        # Process button
        self.btn_process.clicked.connect(self.process_audio)
        
        # Confidence slider
        self.confidence_slider.valueChanged.connect(self.update_confidence_label)
        
    def browse_file(self):
        """Browse for audio or MIDI file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Audio or MIDI File",
            "",
            "Audio Files (*.wav *.mp3 *.flac *.m4a);;"
            "MIDI Files (*.mid *.midi);;"
            "All Files (*.*)"
        )
        
        if file_path:
            self.current_file_path = file_path
            self.file_path_label.setText(os.path.basename(file_path))
            
            # Show success popup
            QMessageBox.information(
                self,
                "File Loaded",
                f"Successfully loaded: {os.path.basename(file_path)}",
                QMessageBox.Ok
            )
            
            # Load and visualize
            self.load_and_visualize_file(file_path)
    
    def load_and_visualize_file(self, file_path: str):
        """Load file and create initial visualization"""
        try:
            ext = os.path.splitext(file_path)[1].lower()
            
            if ext in ['.mid', '.midi']:
                # Handle MIDI file
                self.transcription_results = transcribe_midi_file(file_path)
                self.display_midi_results()
                self.audio_data = None  # MIDI has no audio data
            else:
                # Handle audio file
                self.audio_data, self.sample_rate, duration = load_audio_file(
                    file_path, target_sr=44100
                )
                
                # Update status
                self.status_bar.showMessage(
                    f"Loaded: {os.path.basename(file_path)} | "
                    f"Duration: {duration:.2f}s | "
                    f"Sample Rate: {self.sample_rate}Hz"
                )
                
                # Create initial waveform plot
                self.plot_waveform()
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load file: {str(e)}")
    
    def start_recording(self):
        """Start live recording"""
        try:
            if self.recording_manager.start_recording():
                self.btn_start_recording.setEnabled(False)
                self.btn_pause_recording.setEnabled(True)
                self.btn_stop_recording.setEnabled(True)
                
                # Start update timer
                self.recording_timer.start(100)  # Update every 100ms
                
                self.status_bar.showMessage("Recording...")
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to start recording: {str(e)}")
    
    def pause_recording(self):
        """Pause recording"""
        if self.recording_manager.pause_recording():
            self.btn_pause_recording.setEnabled(False)
            self.btn_resume_recording.setEnabled(True)
            self.status_bar.showMessage("Recording paused")
    
    def resume_recording(self):
        """Resume recording"""
        if self.recording_manager.resume_recording():
            self.btn_pause_recording.setEnabled(True)
            self.btn_resume_recording.setEnabled(False)
            self.status_bar.showMessage("Recording resumed")
    
    def stop_recording(self):
        """Stop recording and process"""
        if self.recording_manager.stop_recording():
            # Stop timer
            self.recording_timer.stop()
            
            # Reset button states
            self.btn_start_recording.setEnabled(True)
            self.btn_pause_recording.setEnabled(False)
            self.btn_resume_recording.setEnabled(False)
            self.btn_stop_recording.setEnabled(False)
            
            # Get recorded audio
            self.audio_data, self.sample_rate = self.recording_manager.get_recorded_audio()
            
            if self.audio_data is not None:
                duration = len(self.audio_data) / self.sample_rate
                self.status_bar.showMessage(
                    f"Recording complete: {duration:.2f}s recorded"
                )
                
                # Plot waveform
                self.plot_waveform()
                
                # Auto-transcribe
                self.process_audio()
            else:
                self.status_bar.showMessage("No audio recorded")
    
    def update_recording_display(self):
        """Update recording timer and level meter"""
        if self.recording_manager.is_recording:
            # Update timer
            duration = self.recording_manager.get_recording_duration()
            minutes = int(duration // 60)
            seconds = int(duration % 60)
            self.recording_display.setText(f"{minutes:02d}:{seconds:02d}")
            
            # Simulate level meter (in real app, would use actual audio levels)
            import random
            level = random.randint(20, 100)
            self.level_meter.setValue(level)
    
    def update_max_notes(self, value: int):
        """Update maximum simultaneous notes setting"""
        self.max_notes = value
        self.status_bar.showMessage(f"Max simultaneous notes set to {value}")
    
    def update_confidence_label(self, value: int):
        """Update confidence threshold label"""
        threshold = value / 100.0
        self.confidence_label.setText(f"{threshold:.2f}")
    
    def process_audio(self):
        """Process audio with transcription"""
        if self.audio_data is None:
            QMessageBox.warning(self, "Warning", "Please load or record audio first")
            return
        
        try:
            # Show progress
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)
            self.status_bar.showMessage("Processing audio...")
            
            # Get settings
            max_notes = self.note_spinbox.value()
            confidence_threshold = self.confidence_slider.value() / 100.0
            
            # Update progress
            self.progress_bar.setValue(30)
            
            # Perform transcription
            self.transcription_results = self.transcription_engine.transcribe_audio(
                self.audio_data,
                self.sample_rate,
                max_simultaneous_notes=max_notes
            )
            
            # Update progress
            self.progress_bar.setValue(70)
            
            # Filter by confidence
            filtered_results = []
            for event in self.transcription_results:
                filtered_notes = [
                    note for note in event['notes']
                    if note.get('confidence', 0) >= confidence_threshold
                ]
                if filtered_notes:
                    event['notes'] = filtered_notes
                    event['note_count'] = len(filtered_notes)
                    filtered_results.append(event)
            
            self.transcription_results = filtered_results
            
            # Update progress
            self.progress_bar.setValue(90)
            
            # Update visualizations
            self.plot_waveform_with_notes()
            self.plot_spectrogram()
            
            # Display results
            self.display_transcription_results()
            
            # Calculate confidence stats
            self.display_confidence_stats()
            
            # Complete progress
            self.progress_bar.setValue(100)
            self.status_bar.showMessage(
                f"Transcription complete: {len(self.transcription_results)} notes detected"
            )
            
            # Hide progress bar after delay
            QTimer.singleShot(1000, lambda: self.progress_bar.setVisible(False))
            
        except Exception as e:
            self.progress_bar.setVisible(False)
            QMessageBox.critical(self, "Error", f"Transcription failed: {str(e)}")
    
    def plot_waveform(self):
        """Plot basic waveform"""
        if self.audio_data is None:
            return
        
        ax = self.waveform_canvas.figure.subplots()
        ax.clear()
        
        times = np.linspace(0, len(self.audio_data) / self.sample_rate, len(self.audio_data))
        ax.plot(times, self.audio_data, linewidth=0.5, color='#1f77b4')
        
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        ax.set_title("Audio Waveform")
        ax.grid(True, alpha=0.3)
        
        self.waveform_canvas.draw()
    
    #def plot_waveform_with_notes(self):
        """Plot waveform with note regions and onsets"""
        if self.audio_data is None or not self.transcription_results:
            return
        
        # Clear existing plot
        ax = self.waveform_canvas.figure.subplots()
        ax.clear()
        
        # Plot waveform
        times = np.linspace(0, len(self.audio_data) / self.sample_rate, len(self.audio_data))
        ax.plot(times, self.audio_data, linewidth=0.5, color='#1f77b4', alpha=0.7)
        
        # Define colors for different note counts
        colors = {
            1: '#4caf50',  # Green for single notes
            2: '#2196f3',  # Blue for 2-note chords
            3: '#ff9800',  # Orange for 3-note chords
            4: '#f44336'   # Red for 4-note chords
        }
        
        # Plot note regions and onsets
        for i, event in enumerate(self.transcription_results):
            color = colors.get(event['note_count'], '#9e9e9e')
            
            # Shade note region
            ax.axvspan(event['onset'], event['offset'], 
                      alpha=0.2, color=color)
            
            # Mark onset
            ax.axvline(x=event['onset'], color='red', 
                      linestyle='--', linewidth=1, alpha=0.7)
            
            # Add note label
            notes = [n['note'] for n in event['notes']]
            confidences = [f"{n.get('confidence', 0):.1%}" for n in event['notes']]
            
            if len(notes) == 1:
                label = f"{notes[0]} ({confidences[0]})"
            else:
                label = "+".join(notes) + f"\n{'+'.join(confidences)}"
            
            mid_time = (event['onset'] + event['offset']) / 2
            ax.text(mid_time, np.max(self.audio_data) * 0.9, label,
                   ha='center', va='center', fontsize=8,
                   bbox=dict(boxstyle='round,pad=0.3', 
                           facecolor='white', alpha=0.8))
        
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        ax.set_title("Waveform with Note Regions and Onsets")
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, times[-1]])
        
        # Add legend
        import matplotlib.patches as mpatches
        patches = []
        for note_count, color in colors.items():
            if note_count <= self.note_spinbox.value():
                label = f"{note_count} note{'s' if note_count > 1 else ''}"
                patches.append(mpatches.Patch(color=color, alpha=0.2, label=label))
        
        if patches:
            ax.legend(handles=patches, loc='upper right')
        
        self.waveform_canvas.draw()
    
    #def plot_spectrogram(self):
        """Plot spectrogram with onsets"""
        if self.audio_data is None:
            return
        
        ax = self.spectrogram_canvas.figure.subplots()
        ax.clear()
        
        # Compute spectrogram
        stft = librosa.stft(self.audio_data)
        spectrogram = librosa.amplitude_to_db(np.abs(stft), ref=np.max)
        
        # Plot spectrogram
        img = librosa.display.specshow(spectrogram, sr=self.sample_rate,
                                      x_axis='time', y_axis='log',
                                      ax=ax, cmap='magma')
        
        # Add onset markers if available
        if self.transcription_results:
            for event in self.transcription_results:
                ax.axvline(x=event['onset'], color='cyan',
                          linestyle='--', linewidth=1.5, alpha=0.7)
        
        ax.set_title("Spectrogram with Onset Markers")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (Hz)")
        
        # Add colorbar
        cbar = self.spectrogram_canvas.figure.colorbar(img, ax=ax)
        cbar.set_label('Intensity (dB)')
        
        self.spectrogram_canvas.draw()
    
    # def plot_waveform_with_notes(self):
    #     """Plot waveform with note regions and onsets"""
    #     if self.audio_data is None or not self.transcription_results:
    #         return
        
    #     # Clear existing plot
    #     ax = self.waveform_canvas.figure.subplots()
    #     ax.clear()
        
    #     # Plot waveform
    #     times = np.linspace(0, len(self.audio_data) / self.sample_rate, len(self.audio_data))
    #     ax.plot(times, self.audio_data, linewidth=0.5, color='#1f77b4', alpha=0.7)
        
    #     # Define colors for different note counts
    #     colors = {
    #         1: '#4caf50',  # Green for single notes
    #         2: '#2196f3',  # Blue for 2-note chords
    #         3: '#ff9800',  # Orange for 3-note chords
    #         4: '#f44336'   # Red for 4-note chords
    #     }
        
    #     # Plot note regions and onsets/offsets
    #     for i, event in enumerate(self.transcription_results):
    #         color = colors.get(event['note_count'], '#9e9e9e')
            
    #         # Shade note region
    #         ax.axvspan(event['onset'], event['offset'], 
    #                   alpha=0.2, color=color)
            
    #         # Mark onset and offset
    #         ax.axvline(x=event['onset'], color='red', 
    #                   linestyle='--', linewidth=1, alpha=0.7)
    #         ax.axvline(x=event['offset'], color='red',
    #                   linestyle='--', linewidth=1, alpha=0.7)
            
    #         # Add note label
    #         notes = [n['note'] for n in event['notes']]
    #         confidences = [f"{n.get('confidence', 0):.1%}" for n in event['notes']]
            
    #         if len(notes) == 1:
    #             label = f"{notes[0]} ({confidences[0]})"
    #         else:
    #             label = "+".join(notes) + f"\n{'+'.join(confidences)}"
            
    #         mid_time = (event['onset'] + event['offset']) / 2
    #         ax.text(mid_time, np.max(self.audio_data) * 0.9, label,
    #                ha='center', va='center', fontsize=8,
    #                bbox=dict(boxstyle='round,pad=0.3', 
    #                        facecolor='white', alpha=0.8))
        
    #     ax.set_xlabel("Time (s)")
    #     ax.set_ylabel("Amplitude")
    #     ax.set_title("Waveform with Note Regions, Onsets and Offsets")
    #     ax.grid(True, alpha=0.3)
    #     ax.set_xlim([0, times[-1]])
        
    #     # Add legend
    #     import matplotlib.patches as mpatches
    #     patches = []
    #     for note_count, color in colors.items():
    #         if note_count <= self.note_spinbox.value():
    #             label = f"{note_count} note{'s' if note_count > 1 else ''}"
    #             patches.append(mpatches.Patch(color=color, alpha=0.2, label=label))
        
    #     if patches:
    #         ax.legend(handles=patches, loc='upper right')
        
    #     self.waveform_canvas.draw()
    def plot_waveform_with_notes(self):
        """Plot waveform with note regions, onsets and offsets"""
        if self.audio_data is None or not self.transcription_results:
            return
        
        # Clear existing plot
        ax = self.waveform_canvas.figure.subplots()
        ax.clear()
        
        # Plot waveform
        times = np.linspace(0, len(self.audio_data) / self.sample_rate, len(self.audio_data))
        ax.plot(times, self.audio_data, linewidth=0.5, color='#1f77b4', alpha=0.7)
        
        # Define colors for different note counts
        colors = {
            1: '#4caf50',  # Green for single notes
            2: '#2196f3',  # Blue for 2-note chords
            3: '#ff9800',  # Orange for 3-note chords
            4: '#f44336'   # Red for 4-note chords
        }
        
        # Plot note regions, onsets and offsets
        for i, event in enumerate(self.transcription_results):
            color = colors.get(event['note_count'], '#9e9e9e')
            
            # Shade note region
            ax.axvspan(event['onset'], event['offset'], 
                      alpha=0.15, color=color)
            
            # Mark onset with red line
            ax.axvline(x=event['onset'], color='red', 
                      linestyle='-', linewidth=1.5, alpha=0.8, label='Onset' if i == 0 else "")

            # Mark offset with red line
            ax.axvline(x=event['offset'], color='blue',
                      linestyle='-', linewidth=2.0, alpha=0.8, label='Offset' if i == 0 else "")
            
            # Add note label
            notes = [n['note'] for n in event['notes']]
            confidences = [f"{n.get('confidence', 0):.0%}" for n in event['notes']]
            
            if len(notes) == 1:
                label = f"{notes[0]}\n({confidences[0]})"
            else:
                label = f"{'+'.join(notes)}\n{'+'.join(confidences)}"
            
            mid_time = (event['onset'] + event['offset']) / 2
            ax.text(mid_time, np.max(self.audio_data) * 0.85, label,
                   ha='center', va='center', fontsize=9, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', 
                           facecolor='white', edgecolor=color, alpha=0.9))
        
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        ax.set_title(f"Waveform with Note Regions ({len(self.transcription_results)} events)")
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, times[-1]])
        
        # Add legend
        import matplotlib.patches as mpatches
        import matplotlib.lines as mlines
        
        # Create custom legend items
        legend_items = []
        
        # Add color patches for note counts
        for note_count, color in colors.items():
            if note_count <= self.note_spinbox.value():
                legend_items.append(mpatches.Patch(color=color, alpha=0.2, 
                                                  label=f"{note_count} note{'s' if note_count > 1 else ''}"))
        
        # Add line markers for onset/offset
        legend_items.append(mlines.Line2D([], [], color='red', linewidth=1.5,
                                         label='Onset'))
        legend_items.append(mlines.Line2D([], [], color='blue', linewidth=1.5,
                                         label='Offset'))
        
        if legend_items:
            ax.legend(handles=legend_items, loc='upper right', fontsize=9)
        
        self.waveform_canvas.draw()

    def plot_spectrogram(self):
        """Plot spectrogram with onsets and offsets"""
        if self.audio_data is None:
            return
        
        ax = self.spectrogram_canvas.figure.subplots()
        ax.clear()
        
        # Compute spectrogram
        stft = librosa.stft(self.audio_data)
        spectrogram = librosa.amplitude_to_db(np.abs(stft), ref=np.max)
        
        # Plot spectrogram
        img = librosa.display.specshow(spectrogram, sr=self.sample_rate,
                                      x_axis='time', y_axis='log',
                                      ax=ax, cmap='magma')
        
        # Add onset and offset markers if available
        if self.transcription_results:
            for event in self.transcription_results:
                ax.axvline(x=event['onset'], color='cyan',
                          linestyle='--', linewidth=1.5, alpha=0.7, label='Onset')
                ax.axvline(x=event['offset'], color='magenta',
                          linestyle='--', linewidth=1.5, alpha=0.7, label='Offset')
        
        ax.set_title("Spectrogram with Onset and Offset Markers")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (Hz)")
        
        # Add colorbar
        cbar = self.spectrogram_canvas.figure.colorbar(img, ax=ax)
        cbar.set_label('Intensity (dB)')
        
        # Remove duplicate labels
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        if by_label:
            ax.legend(by_label.values(), by_label.keys(), loc='upper right')
        
        self.spectrogram_canvas.draw()

    def display_transcription_results(self):
        """Display transcription results with confidence"""
        self.results_text.clear()
        
        if not self.transcription_results:
            self.results_text.append("No notes detected.")
            return
        
        # Header
        self.results_text.append("=" * 80)
        self.results_text.append("TRANSCRIPTION RESULTS")
        self.results_text.append("=" * 80)
        self.results_text.append("")
        
        # Summary
        total_notes = sum(event['note_count'] for event in self.transcription_results)
        self.results_text.append(f"Total Events: {len(self.transcription_results)}")
        self.results_text.append(f"Total Notes: {total_notes}")
        self.results_text.append("")
        
        # Detailed results
        for i, event in enumerate(self.transcription_results):
            self.results_text.append(f"Event {i+1}:")
            self.results_text.append(f"  Time: {event['onset']:.3f}s → {event['offset']:.3f}s "
                                   f"(Duration: {event['duration']:.3f}s)")
            
            for j, note in enumerate(event['notes']):
                confidence = note.get('confidence', 0)
                confidence_str = f"{confidence:.1%}"
                
                # Color code confidence
                if confidence > 0.7:
                    conf_color = "🟢"
                elif confidence > 0.4:
                    conf_color = "🟡"
                else:
                    conf_color = "🔴"
                
                self.results_text.append(f"  Note {j+1}: {note['note']:4s} "
                                       f"(Pitch: {note['pitch']:6.1f} Hz) "
                                       f"{conf_color} Confidence: {confidence_str}")
            
            self.results_text.append("")
    
    def display_confidence_stats(self):
        """Display confidence statistics"""
        if not self.transcription_results:
            self.confidence_text.clear()
            return
        
        stats = calculate_confidence_stats(self.transcription_results)
        
        self.confidence_text.clear()
        self.confidence_text.append("Confidence Statistics:")
        self.confidence_text.append("-" * 40)
        
        for key, value in stats.items():
            if isinstance(value, float):
                formatted = f"{value:.2%}" if 'confidence' in key else f"{value:.3f}"
            else:
                formatted = str(value)
            
            # Format key for display
            display_key = key.replace('_', ' ').title()
            self.confidence_text.append(f"{display_key}: {formatted}")
    
    def display_midi_results(self):
        """Display results from MIDI file"""
        if not self.transcription_results:
            return
        
        self.results_text.clear()
        self.results_text.append("=" * 80)
        self.results_text.append("MIDI TRANSCRIPTION RESULTS")
        self.results_text.append("=" * 80)
        self.results_text.append("")
        
        for i, event in enumerate(self.transcription_results):
            self.results_text.append(f"Note {i+1}:")
            self.results_text.append(f"  Time: {event['onset']:.3f}s → {event['offset']:.3f}s")
            self.results_text.append(f"  Note: {event['note']} (MIDI: {event['midi_note']})")
            self.results_text.append(f"  Velocity: {event['velocity']:.2f}")
            self.results_text.append("")
    
    def preview_audio(self):
        """Preview loaded audio"""
        if self.audio_data is None:
            QMessageBox.warning(self, "Warning", "No audio loaded to preview")
            return
        
        # Simple audio preview using sounddevice
        try:
            import sounddevice as sd
            sd.play(self.audio_data, self.sample_rate)
            self.status_bar.showMessage("Playing audio preview...")
        except ImportError:
            QMessageBox.warning(
                self,
                "Preview Unavailable",
                "Install sounddevice package for audio preview:\n"
                "pip install sounddevice"
            )
    
    def clear_file(self):
        """Clear current file and reset"""
        self.current_file_path = None
        self.audio_data = None
        self.transcription_results = []
        
        self.file_path_label.setText("No file selected")
        self.results_text.clear()
        self.confidence_text.clear()
        
        # Clear plots
        self.waveform_canvas.figure.clear()
        self.spectrogram_canvas.figure.clear()
        self.waveform_canvas.draw()
        self.spectrogram_canvas.draw()
        
        self.status_bar.showMessage("Cleared")
    
    def export_results(self, format_type: str):
        """Export results to file"""
        if not self.transcription_results:
            QMessageBox.warning(self, "Warning", "No results to export")
            return
        
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            if format_type == 'txt':
                file_path, _ = QFileDialog.getSaveFileName(
                    self,
                    "Export as Text",
                    f"transcription_{timestamp}.txt",
                    "Text Files (*.txt)"
                )
                
                if file_path:
                    with open(file_path, 'w') as f:
                        f.write(self.results_text.toPlainText())
                    
                    QMessageBox.information(self, "Success", 
                                          f"Results exported to {file_path}")
            
            elif format_type == 'csv':
                file_path, _ = QFileDialog.getSaveFileName(
                    self,
                    "Export as CSV",
                    f"transcription_{timestamp}.csv",
                    "CSV Files (*.csv)"
                )
                
                if file_path:
                    import csv
                    with open(file_path, 'w', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(['Event', 'Onset', 'Offset', 'Duration', 
                                       'Note', 'Pitch', 'Confidence', 'Method'])
                        
                        for i, event in enumerate(self.transcription_results):
                            for j, note in enumerate(event['notes']):
                                writer.writerow([
                                    i + 1,
                                    event['onset'],
                                    event['offset'],
                                    event['duration'],
                                    note['note'],
                                    note['pitch'],
                                    note.get('confidence', 0),
                                    note.get('detection_method', 'N/A')
                                ])
                    
                    QMessageBox.information(self, "Success", 
                                          f"Results exported to {file_path}")
            
            elif format_type == 'midi':
                file_path, _ = QFileDialog.getSaveFileName(
                    self,
                    "Export as MIDI",
                    f"transcription_{timestamp}.mid",
                    "MIDI Files (*.mid)"
                )
                
                if file_path:
                    self.export_to_midi(file_path)
                    QMessageBox.information(self, "Success", 
                                          f"MIDI exported to {file_path}")
        
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Export failed: {str(e)}")
    
    def export_to_midi(self, file_path: str):
        """Export transcription to MIDI file"""
        try:
            mid = mido.MidiFile()
            track = mido.MidiTrack()
            mid.tracks.append(track)
            
            # Add tempo track
            track.append(mido.MetaMessage('set_tempo', tempo=500000))
            
            current_tick = 0
            ticks_per_beat = 480  # Standard MIDI resolution
            
            for event in self.transcription_results:
                # Convert seconds to ticks
                onset_ticks = int(event['onset'] * 1000000 / 500000 * ticks_per_beat)
                offset_ticks = int(event['offset'] * 1000000 / 500000 * ticks_per_beat)
                
                # Calculate delta times
                delta_onset = onset_ticks - current_tick
                delta_duration = offset_ticks - onset_ticks
                
                # Add note on events
                for note in event['notes']:
                    # Convert note name to MIDI number
                    midi_note = self.note_name_to_midi(note['note'])
                    if midi_note:
                        track.append(mido.Message('note_on', 
                                                note=midi_note,
                                                velocity=64,
                                                time=delta_onset))
                        # Note off after duration
                        track.append(mido.Message('note_off',
                                                note=midi_note,
                                                velocity=64,
                                                time=delta_duration))
                
                current_tick = offset_ticks
            
            mid.save(file_path)
            
        except Exception as e:
            raise Exception(f"Failed to export MIDI: {str(e)}")
    
    def note_name_to_midi(self, note_name: str) -> Optional[int]:
        """Convert note name (e.g., C#4) to MIDI number"""
        try:
            # Simple conversion for common notes
            note_map = {
                'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3,
                'E': 4, 'F': 5, 'F#': 6, 'Gb': 6, 'G': 7, 'G#': 8,
                'Ab': 8, 'A': 9, 'A#': 10, 'Bb': 10, 'B': 11
            }
            
            # Parse note name (e.g., "C#4")
            import re
            match = re.match(r'([A-G][#b]?)(\d+)', note_name)
            if match:
                note, octave = match.groups()
                octave = int(octave)
                midi = note_map[note] + (octave + 1) * 12
                return midi if 0 <= midi <= 127 else None
            return None
        except:
            return None
    
    def closeEvent(self, event):
        """Clean up on close"""
        self.recording_manager.cleanup()
        event.accept()
