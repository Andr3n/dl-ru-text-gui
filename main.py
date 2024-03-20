import sys
from PyQt5.QtWidgets import QApplication
from gui import LetterGenerationApp

def main():
    app = QApplication(sys.argv)
    window = LetterGenerationApp()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()