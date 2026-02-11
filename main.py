# main.py - Simple version
import sys
from PyQt5.QtWidgets import QApplication
from gui import AudioProcessorGUI

def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')  # Use Fusion style for better look
    
    # Create and show main window
    window = AudioProcessorGUI()
    window.show()
    
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()