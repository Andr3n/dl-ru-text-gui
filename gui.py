from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton, QFileDialog, QTabWidget, QComboBox, QGroupBox, QGridLayout
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt

from ml_models import generate_letter,  initialize_generator
from ml_models import recognize_letter, recognize_word, initialize_recognizer

import gc

class LetterGenerationApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self._generator = None
        self._recognizer = None
        self.current_recognition_model = 'letter'

    def initUI(self):
        self.setWindowTitle('Распознавание и генерация рукописного русского текста')
        self.resize(600, 600)

        # Общий макет
        layout = QVBoxLayout()
        tab_widget = QTabWidget()

        ##########################
        # Вкладка "Распознавание" #
        ##########################
        recognition_tab = QWidget()
        layout_recog = QVBoxLayout(recognition_tab)

        groupbox = QGroupBox('Фотография для распознавания')
        grid_layout = QGridLayout(groupbox)

        inner_groupbox = QGroupBox()
        inner_groupbox.setFlat(True)
        inner_grid_layout = QGridLayout(inner_groupbox)
        inner_grid_layout.setVerticalSpacing(0)

        self.recognition_label = QLabel('')
        self.recognition_label.setAlignment(Qt.AlignCenter)
        self.recognized_text_label = QLabel('', alignment=Qt.AlignCenter)
        self.recognized_text_label.setVisible(False)

        inner_grid_layout.addWidget(self.recognition_label, 0, 0, 1, 2)
        inner_grid_layout.addWidget(self.recognized_text_label, 1, 0, 1, 2)
        inner_groupbox.setStyleSheet("border: none; background-color: transparent;")

        self.recognition_groupbox = inner_groupbox

        model_label = QLabel('Модель распознавания:')
        self.recognition_cb_model = QComboBox()
        self.recognition_cb_model.addItems(['Буквы', 'Слова'])

        self.recognition_btn_load = QPushButton('Загрузить фото')
        self.recognition_btn_load.clicked.connect(self.load_image_recognition)

        self.recognition_btn_recognize = QPushButton('Распознать!')
        self.recognition_btn_recognize.clicked.connect(self.recognize_text)

        grid_layout.addWidget(self.recognition_groupbox, 0, 0, 1, 2)
        grid_layout.addWidget(model_label, 1, 0, 1, 1)
        grid_layout.addWidget(self.recognition_cb_model, 1, 1, 1, 1)
        grid_layout.addWidget(self.recognition_btn_load, 2, 0, 1, 2)
        grid_layout.addWidget(self.recognition_btn_recognize, 3, 0, 1, 2)

        # Отрисовка границ для теста
        # for i in range(grid_layout.count()):
        #     w = grid_layout.itemAtPosition(i // grid_layout.columnCount(), i % grid_layout.columnCount()).widget()
        #     w.setStyleSheet('margin: 10px; border: 1px solid red;')

        layout_recog.addWidget(groupbox)
        tab_widget.addTab(recognition_tab, "Распознавание")

        #################
        # Вкладка "Генерация" #
        #################
        generation_tab = QWidget()
        layout_gen = QVBoxLayout(generation_tab)

        groupbox = QGroupBox('Генерация буквы')
        grid_layout = QGridLayout(groupbox)
        
        self.generation_label = QLabel('Выберите букву для генерации:')

        self.generation_combobox = QComboBox()
        self.generation_combobox.setStyleSheet("font-size: 12pt")
        self.generation_combobox.setFixedWidth(200)
        self.generation_combobox.addItems(['А', 'Б', 'В', 'Г', 'Д', 'Е', 'Ё', 'Ж', 'З', 'И', 'Й', 'К', 'Л', 'М', 'Н', 'О', 'П', 'Р', 'С', 'Т', 'У', 'Ф', 'Х', 'Ц', 'Ч', 'Ш', 'Щ', 'Ъ', 'Ы', 'Ь', 'Э', 'Ю', 'Я'])
        
        self.generation_btn_generate = QPushButton('Сгенерировать букву')
        self.generation_btn_generate.clicked.connect(self.generate_letter)
        
        self.generation_btn_save = QPushButton('Сохранить букву')
        self.generation_btn_save.clicked.connect(self.save_letter)
        
        self.generation_image_label = QLabel()
        self.generation_image_label.setAlignment(Qt.AlignCenter)

        grid_layout.addWidget(self.generation_label, 0, 0)
        grid_layout.addWidget(self.generation_combobox, 0, 1)
        grid_layout.addWidget(self.generation_btn_generate, 1, 0)
        grid_layout.addWidget(self.generation_btn_save, 1, 1)
        grid_layout.addWidget(self.generation_image_label, 2, 0, 1, 2)

        layout_gen.addWidget(groupbox)
        tab_widget.addTab(generation_tab, "Генерация")

        layout.addWidget(tab_widget)
        self.setLayout(layout)
        self.show()

    def load_image_recognition(self):
        file_path, _ = QFileDialog.getOpenFileName(self, 'Выберите фото', '', 'Изображения (*.png *.jpg *.jpeg)')
        if file_path:
            pixmap = self.create_display_pixmap(file_path)
            self.recognition_label.setPixmap(pixmap)
            self.file_to_recognize_path = file_path
            self.recognized_text_label.setVisible(False)

    def recognize_text(self):
        if self.recognition_cb_model.currentText() == 'Буквы':
            self.current_recognition_model = 'letter'
        elif self.recognition_cb_model.currentText() == 'Слова':
            self.current_recognition_model = 'word'

        if self._recognizer is None:
            self._r_model = self.current_recognition_model
        
        if self.current_recognition_model != self._r_model:
            del self._recognizer
            gc.collect()
            self._r_model = self.current_recognition_model

        self._recognizer = initialize_recognizer(self._r_model)

        print(f'Start recognizing with {self._r_model} model')
        if self._r_model == 'letter':
            predicted_class = recognize_letter(self._recognizer, self.file_to_recognize_path)
            photo_type = 'буква'
        elif self._r_model == 'word':
            predicted_class = recognize_word(self._recognizer, self.file_to_recognize_path)
            photo_type = 'слово'
        else:
            raise ValueError('Не указана модель для распознавания')
        
        original_text = f"<strong> {predicted_class} </strong>"
        new_text = f"<span style='color: blue; font-weight: bold; font-size: 18px;'>На фото {photo_type} </span>{original_text}"
        self.recognized_text_label.setText(new_text)
        self.recognized_text_label.setVisible(True)

    def create_display_pixmap(self, file_path, size=300):
        image = QImage(file_path)
        scaled_image = image.scaled(size, size, aspectRatioMode=Qt.KeepAspectRatio, transformMode=Qt.SmoothTransformation)
        pixmap = QPixmap.fromImage(scaled_image)
        return pixmap

    def generate_letter(self):
        # TODO: Add ability to change to other model in future
        if self._generator is None:
            self._generator = initialize_generator()

        selected_letter = self.generation_combobox.currentText()
        
        file_path = generate_letter(self._generator, selected_letter)
        pixmap = self.create_display_pixmap(file_path, size=128)
        self.generation_image_label.setPixmap(pixmap)
        self.generation_btn_save.setEnabled(True)

    def save_letter(self):
        pixmap = self.generation_image_label.pixmap()
        if pixmap:
            file_path, _ = QFileDialog.getSaveFileName(self, 'Сохранить изображение', '', 'Изображения (*.png *.jpg *.jpeg)')
            if file_path:
                pixmap.save(file_path)

